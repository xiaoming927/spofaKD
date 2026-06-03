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


class EMADynamicScaler:
    """
    EMA-Guided Dynamic Scaler.

    Maintains an Exponential Moving Average (EMA) of a metric as the baseline.
    Calculates the deviation of the current metric from this baseline to generate an adjustment signal.

    Formula:
        Baseline_t = Momentum * Baseline_{t-1} + (1 - Momentum) * Current_t
        Deviation  = Baseline_t - Current_t
        Adjustment = Scale_Factor * Deviation
    """

    def __init__(self, scale_factor=2.0, momentum=0.99):
        self.scale_factor = scale_factor
        self.momentum = momentum
        self.ema_baseline = None

    def update(self, current_metric):
        if self.ema_baseline is None:
            self.ema_baseline = current_metric
            return 0.0, self.ema_baseline

        deviation = self.ema_baseline - current_metric
        self.ema_baseline = self.momentum * self.ema_baseline + (1 - self.momentum) * current_metric
        adjustment = self.scale_factor * deviation

        return adjustment, self.ema_baseline


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.0, reduction='mean'):
    """
    OFA Loss, supports returning element-wise loss (reduction='none').
    """
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps

    loss = torch.sum(-(prod - target_mask) * torch.log(pred_student), dim=-1)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


@register_distiller
class SPOFA(BaseDistiller):
    """
    SPOFA: Momentum-based Adaptive Distillation

    Adaptive distillation algorithm based on momentum guidance. Utilizes the EMA mechanism
    to monitor Gradient Conflict and Feature Alignment during training, dynamically
    adjusting distillation weights for different layers and samples.
    """
    requires_feat = True

    def __init__(self, student, teacher, criterion, args,
                 output_scale_factor=2.0,
                 feature_scale_factor=5.0,
                 momentum=0.99,
                 enable_adaptive=True,
                 log_file="spofa_momentum_log.csv",
                 **kwargs):
        super(SPOFA, self).__init__(student, teacher, criterion, args)

        self.enable_adaptive = enable_adaptive

        if self.enable_adaptive:
            # Output layer scaler for resolving gradient conflicts
            self.output_scaler = EMADynamicScaler(scale_factor=output_scale_factor, momentum=momentum)

            # Feature layer scalers for hierarchical feature alignment
            self.feature_scalers = []
            for _ in args.ofa_stage:
                self.feature_scalers.append(EMADynamicScaler(scale_factor=feature_scale_factor, momentum=momentum))

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
                    nn.LayerNorm(max(feature_dim_s, feature_dim_t)),
                    nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes)
                )
            else:
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

        self.log_filename = log_file
        self.log_path = None

    def _is_master_process(self):
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def set_log_dir(self, save_dir):
        if self._is_master_process():
            self.log_path = os.path.join(save_dir, self.log_filename)
            if not os.path.exists(self.log_path):
                try:
                    with open(self.log_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
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
        # 1. Teacher inference
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        # 2. Student inference
        logits_student, feat_student = self.student(image, requires_feat=True)

        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        # 3. Dynamic Feature Layer Loss
        ofa_losses = []
        feature_weights = []

        for i, (stage, eps) in enumerate(zip(self.args.ofa_stage, self.args.ofa_eps)):
            idx_s, _ = self.student.stage_info(stage)
            feat_s = feat_student[idx_s]
            logits_student_head = get_module_dict(self.projector, stage)(feat_s)

            loss_stage = ofa_loss(
                logits_student_head, logits_teacher, target_mask, eps, self.args.ofa_temperature
            )

            dynamic_scale = 1.0
            if self.enable_adaptive and self.training:
                with torch.no_grad():
                    sim_current = F.cosine_similarity(logits_student_head, logits_teacher, dim=1).mean().item()
                    scaler = self.feature_scalers[i]
                    adjustment, _ = scaler.update(sim_current)

                    dynamic_scale = 1.0 - adjustment
                    dynamic_scale = np.clip(dynamic_scale, 0.1, 2.0)

            feature_weights.append(dynamic_scale)
            weighted_loss_stage = loss_stage * dynamic_scale
            ofa_losses.append(weighted_loss_stage)

        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)
        avg_feature_weight = sum(feature_weights) / len(feature_weights) if feature_weights else 1.0

        # 4. Ground Truth (GT) Loss
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)

        # 5. Dynamic Output Layer Weights (Gradient Consistency EMA)
        loss_kd_raw_vec = ofa_loss(
            logits_student, logits_teacher, target_mask,
            self.args.ofa_eps[-1], self.args.ofa_temperature, reduction='none'
        )

        dynamic_weight = torch.ones(image.size(0), device=image.device)
        batch_sim_mean = 0.0

        if self.enable_adaptive and self.training:
            # Gradient Probing
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

            g1_vec = torch.autograd.grad(loss_gt_vec.sum(), logits_probe, create_graph=False)[0]
            g2_vec = torch.autograd.grad(loss_kd_vec.sum(), logits_probe, create_graph=False)[0]

            if g1_vec is not None and g2_vec is not None:
                sim_vec = F.cosine_similarity(g1_vec, g2_vec, dim=1, eps=1e-8)
                batch_sim_mean = sim_vec.mean().item()

                _, _ = self.output_scaler.update(batch_sim_mean)
                ema_baseline_val = self.output_scaler.ema_baseline

                deviation_vec = torch.clamp(ema_baseline_val - sim_vec, min=0.0)
                adjustment_vec = self.output_scaler.scale_factor * deviation_vec
                weight_adjustment = 1.0 - (0.1 * adjustment_vec)

                dynamic_weight = torch.clamp(weight_adjustment, 0.5, 2.0)

        # 6. Final KD Loss
        loss_kd = (self.args.kd_loss_weight * dynamic_weight * loss_kd_raw_vec).mean()

        # Logging
        log_data = {
            "grad_consistency": batch_sim_mean,
            "output_weight": dynamic_weight.mean().item(),
            "feature_weight": avg_feature_weight,
            "loss_gt": loss_gt.item(),
            "loss_kd": loss_kd.item(),
            "loss_spofa": loss_ofa.item() if isinstance(loss_ofa, torch.Tensor) else loss_ofa
        }
        self._log_to_csv(log_data)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_spofa": loss_ofa,
        }

        return logits_student, losses_dict
