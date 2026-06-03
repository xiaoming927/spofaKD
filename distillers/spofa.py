import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from timm.models.vision_transformer import Block
from ._base import BaseDistiller
from .registry import register_distiller
from .utils import (
    GAP1d,
    get_module_dict,
    init_weights,
    is_cnn_model,
    PatchMerging,
    SepConv,
    set_module_dict,
    TokenFilter,
    TokenFnContext,
)


# --- 核心模块：EMA 动态缩放器 (原 PID 控制器重构) ---
class EMADynamicScaler:
    """
    EMA 动态缩放器 (EMA-Guided Dynamic Scaler)。

    核心思想:
    维护一个指标的指数移动平均 (EMA) 作为'历史基准' (Historical Baseline)。
    计算当前指标与历史基准的'偏差' (Deviation)，据此生成调节信号。

    Formula:
        Baseline_t = Momentum * Baseline_{t-1} + (1 - Momentum) * Current_t
        Deviation  = Baseline_t - Current_t
        Adjustment = Scale_Factor * Deviation
    """

    def __init__(self, scale_factor=2.0, momentum=0.99):
        """
        Args:
            scale_factor (float): 调节灵敏度因子 (对应原 PID 的 Kp)。
            momentum (float): EMA 的动量系数，决定历史记忆的长度。
        """
        self.scale_factor = scale_factor
        self.momentum = momentum
        self.ema_baseline = None  # 存储历史趋势基准

    def update(self, current_metric):
        """
        更新 EMA 基准并计算调节量。

        Args:
            current_metric: 当前步的观测值 (如相似度、一致性等)

        Returns:
            adjustment: 调节信号。
                        如果 Current < Baseline (表现下降)，Deviation > 0 -> Adjustment > 0
                        如果 Current > Baseline (表现上升)，Deviation < 0 -> Adjustment < 0
            ema_baseline: 当前更新后的基准值
        """
        # 1. 初始化或更新 EMA 基准线
        # if self.ema_baseline is None:
        #     self.ema_baseline = current_metric
        # else:
        #     self.ema_baseline = self.momentum * self.ema_baseline + (1 - self.momentum) * current_metric
        #
        # # 2. 计算相对偏差 (Deviation from History)
        # # Positive Deviation implies instable or degrading performance relative to history
        # deviation = self.ema_baseline - current_metric
        #
        # # 3. 计算调节幅度
        # adjustment = self.scale_factor * deviation
        #
        # return adjustment, self.ema_baseline

        # 1. 初始化情况
        if self.ema_baseline is None:
            self.ema_baseline = current_metric
            return 0.0, self.ema_baseline  # 初始时刻没有历史，偏差为0

        # 2. 先计算相对偏差 (使用旧的 ema_baseline 即 mu_{t-1})
        # Deviation = Baseline_{t-1} - Current_t
        deviation = self.ema_baseline - current_metric

        # 3. 再更新 EMA 基准线 (计算 mu_t)
        self.ema_baseline = self.momentum * self.ema_baseline + (1 - self.momentum) * current_metric

        # 4. 计算调节幅度
        adjustment = self.scale_factor * deviation

        return adjustment, self.ema_baseline


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.0, reduction='mean'):
    """
    OFA Loss，支持返回向量 (reduction='none')
    """
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps

    # 计算每个样本的 loss (向量)
    loss = torch.sum(-(prod - target_mask) * torch.log(pred_student), dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # 返回 [Batch_Size] 向量


@register_distiller
class SPOFA(BaseDistiller):
    """
    SPOFA: Momentum-based Adaptive Distillation

    基于动量引导的自适应蒸馏算法。利用 EMA 机制监测训练过程中的
    梯度冲突 (Gradient Conflict) 和 特征对齐度 (Feature Alignment)，
    动态调整不同层级和样本的蒸馏权重。
    """
    requires_feat = True

    def __init__(self, student, teacher, criterion, args,
                 # --- 自适应参数 (原 PID 参数) ---
                 output_scale_factor=2.0,  # 输出层调节灵敏度 (原 pid_kp)
                 feature_scale_factor=5.0,  # 特征层调节灵敏度 (原 stage_kp)
                 momentum=0.99,  # EMA 动量 (原 pid_momentum)
                 enable_adaptive=True,  # 是否启用自适应机制 (原 use_pid)
                 # --- 日志参数 ---
                 log_file="spofa_momentum_log.csv",
                 **kwargs):
        super(SPOFA, self).__init__(student, teacher, criterion, args)

        self.enable_adaptive = enable_adaptive

        # 1. 初始化 EMA 缩放器
        if self.enable_adaptive:
            # A. 输出层缩放器 (用于解决梯度冲突)
            self.output_scaler = EMADynamicScaler(scale_factor=output_scale_factor, momentum=momentum)

            # B. 特征层缩放器 (用于层级特征对齐)
            # 为每个 Stage 创建独立的 Scaler，捕捉不同层级的收敛动态
            self.feature_scalers = []
            for _ in args.ofa_stage:
                self.feature_scalers.append(EMADynamicScaler(scale_factor=feature_scale_factor, momentum=momentum))

        # 2. 初始化 Projector (含 LayerNorm 修复)
        if len(self.args.ofa_eps) == 1:
            eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
            self.args.ofa_eps = eps
        assert len(self.args.ofa_stage) + 1 == len(self.args.ofa_eps)

        self.projector = nn.ModuleDict()
        is_cnn_student = is_cnn_model(student)
        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        for stage in self.args.ofa_stage:
            _, size_s = self.student.stage_info(stage)
            if is_cnn_student:
                # CNN Projector
                in_chans, _, _ = size_s
                if stage != 4:
                    down_sample_blk_num = 4 - stage
                    down_sample_blks = []
                    for i in range(down_sample_blk_num):
                        if i == down_sample_blk_num - 1:
                            out_chans = max(feature_dim_s, feature_dim_t)
                        else:
                            out_chans = in_chans * 2
                        down_sample_blks.append(SepConv(in_chans, out_chans))
                        in_chans *= 2
                else:
                    down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]
                projector = nn.Sequential(
                    *down_sample_blks,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.LayerNorm(max(feature_dim_s, feature_dim_t)),  # LayerNorm 稳压器 对比nn.BatchNorm1d为什么不用这个
                    nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes)
                )
            else:
                # Transformer Projector
                patch_num, embed_dim = size_s
                token_num = getattr(student, "num_tokens", 0)
                final_patch_grid = 7
                patch_grid = int(patch_num ** 0.5)
                merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
                merger_modules = []
                for i in range(merge_num):
                    if i == 0:
                        merger_modules.append(
                            PatchMerging((patch_grid // 2 ** i, patch_grid // 2 ** i), embed_dim, feature_dim_s,
                                         nn.LayerNorm))
                    else:
                        merger_modules.append(
                            PatchMerging((patch_grid // 2 ** i, patch_grid // 2 ** i), feature_dim_s, feature_dim_s,
                                         nn.LayerNorm if i != merge_num - 1 else nn.Identity))

                patch_merger = nn.Sequential(*merger_modules)
                blocks = nn.Sequential(*[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))])
                get_feature = nn.Sequential(TokenFilter(token_num, False), nn.Flatten()) if token_num != 0 else GAP1d()
                projector = nn.Sequential(TokenFnContext(token_num, patch_merger), blocks, get_feature,
                                          nn.Linear(feature_dim_s, args.num_classes))
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)

        # 3. 日志初始化 (Lazy Init)
        self.log_filename = log_file
        self.log_path = None

    def _is_master_process(self):
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def set_log_dir(self, save_dir):
        """
        由 train.py 调用，设置日志路径
        """
        if self._is_master_process():
            self.log_path = os.path.join(save_dir, self.log_filename)
            if not os.path.exists(self.log_path):
                try:
                    with open(self.log_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        # 更新 CSV 表头，使用学术化术语
                        writer.writerow(['timestamp', 'grad_consistency', 'output_weight', 'feature_weight',
                                         'loss_gt', 'loss_kd', 'loss_spofa'])
                    print(f"[SPOFA] Log file initialized at: {self.log_path}")
                except Exception as e:
                    print(f"[SPOFA] Failed to init log file: {e}")

    def _log_to_csv(self, data_dict):
        if self.training and self._is_master_process() and self.log_path is not None:
            try:
                with open(self.log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.time(),
                        f"{data_dict.get('grad_consistency', 0):.4f}",
                        f"{data_dict.get('output_weight', 1.0):.4f}",
                        f"{data_dict.get('feature_weight', 1.0):.4f}",
                        f"{data_dict['loss_gt']:.4f}",
                        f"{data_dict['loss_kd']:.4f}",
                        f"{data_dict['loss_spofa']:.4f}"
                    ])
            except Exception as e:
                pass

    def forward(self, image, label, *args, **kwargs):
        # 1. 老师推断
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        # 2. 学生推断
        logits_student, feat_student = self.student(image, requires_feat=True)

        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        # --- 3. 动态特征层损失 (基于 EMA 偏差) ---
        ofa_losses = []
        feature_weights = []  # Log usage

        for i, (stage, eps) in enumerate(zip(self.args.ofa_stage, self.args.ofa_eps)):
            idx_s, _ = self.student.stage_info(stage)
            feat_s = feat_student[idx_s]
            logits_student_head = get_module_dict(self.projector, stage)(feat_s)

            # 基础 Loss
            loss_stage = ofa_loss(
                logits_student_head, logits_teacher, target_mask, eps, self.args.ofa_temperature
            )

            # --- Feature-wise EMA Dynamic Weighting ---
            dynamic_scale = 1.0
            if self.enable_adaptive and self.training:
                with torch.no_grad():
                    # 计算当前相似度 (Current Metric)
                    sim_current = F.cosine_similarity(logits_student_head, logits_teacher, dim=1).mean().item()

                    # 获取 EMA Scaler
                    scaler = self.feature_scalers[i]

                    # 更新 EMA 并获取调节信号 (Adjustment = Scale * Deviation)
                    adjustment, _ = scaler.update(sim_current)

                    # 逻辑:
                    # 若 Sim 下降 (Deviation > 0) -> Adjustment > 0 -> Weight 降低 (抑制噪声/等待追赶)
                    # 若 Sim 上升 (Deviation < 0) -> Adjustment < 0 -> Weight 增加 (鼓励优势)
                    dynamic_scale = 1.0 - adjustment

                    # 裁剪权重 [0.1, 2.0]
                    dynamic_scale = np.clip(dynamic_scale, 0.1, 2.0)

            feature_weights.append(dynamic_scale)

            # 应用动态权重
            weighted_loss_stage = loss_stage * dynamic_scale
            ofa_losses.append(weighted_loss_stage)

        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)
        avg_feature_weight = sum(feature_weights) / len(feature_weights) if feature_weights else 1.0

        # 4. GT Loss
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)

        # KD Loss 向量
        loss_kd_raw_vec = ofa_loss(
            logits_student, logits_teacher, target_mask,
            self.args.ofa_eps[-1], self.args.ofa_temperature, reduction='none'
        )

        # --- 5. 动态输出层权重 (基于梯度一致性 EMA) ---
        dynamic_weight = torch.ones(image.size(0), device=image.device)
        batch_sim_mean = 0.0

        if self.enable_adaptive and self.training:
            # A. 梯度探测 (Gradient Probing)
            with torch.no_grad():
                logits_probe = logits_student.detach()
            logits_probe.requires_grad = True

            if len(label.shape) == 1:
                loss_gt_vec = F.cross_entropy(logits_probe, label, reduction='none')
            else:
                loss_gt_vec = torch.sum(-label * F.log_softmax(logits_probe, dim=-1), dim=-1)

            loss_kd_vec = ofa_loss(
                logits_probe, logits_teacher.detach(), target_mask,
                self.args.ofa_eps[-1], self.args.ofa_temperature, reduction='none'
            )

            # B. 计算梯度一致性 (Gradient Consistency)
            g1_vec = torch.autograd.grad(loss_gt_vec.sum(), logits_probe, create_graph=False)[0]
            g2_vec = torch.autograd.grad(loss_kd_vec.sum(), logits_probe, create_graph=False)[0]

            if g1_vec is not None and g2_vec is not None:
                sim_vec = F.cosine_similarity(g1_vec, g2_vec, dim=1, eps=1e-8)

                # C. EMA 调节逻辑
                batch_sim_mean = sim_vec.mean().item()

                # 更新 Output Scaler
                _, _ = self.output_scaler.update(batch_sim_mean)

                # 计算样本级偏差 (Sample-wise Deviation)
                # 使用全局 EMA Baseline 对比每个样本的相似度
                ema_baseline_val = self.output_scaler.ema_baseline

                # Deviation > 0 意味着当前样本冲突比历史平均水平更严重
                deviation_vec = torch.clamp(ema_baseline_val - sim_vec, min=0.0)

                # Adjustment 越大 -> 惩罚越重 -> 权重越低
                # Formula: Weight = 1.0 - (Sensitivity * Adjustment)
                # 这里沿用之前的系数逻辑，保持数学等价性
                adjustment_vec = self.output_scaler.scale_factor * deviation_vec
                weight_adjustment = 1.0 - (0.1 * adjustment_vec)

                # 兜底
                dynamic_weight = torch.clamp(weight_adjustment, 0.5, 2.0)

        # 6. 最终 KD Loss
        loss_kd = (self.args.kd_loss_weight * dynamic_weight * loss_kd_raw_vec).mean()

        # --- 日志 ---
        log_data = {
            "grad_consistency": batch_sim_mean,
            "output_weight": dynamic_weight.mean().item(),
            "feature_weight": avg_feature_weight,
            "loss_gt": loss_gt.item(),
            "loss_kd": loss_kd.item(),
            "loss_spofa": loss_ofa.item() if isinstance(loss_ofa, torch.Tensor) else loss_ofa
        }
        self._log_to_csv(log_data)

        # --- 返回 ---
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_spofa": loss_ofa,
        }

        return logits_student, losses_dict