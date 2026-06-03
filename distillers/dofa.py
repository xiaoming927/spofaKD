import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from ._base import BaseDistiller
from .registry import register_distiller
from .utils import (
    GAP1d,
    get_module_dict,
    init_weights,
    is_cnn_model,
    SepConv,
    set_module_dict,
    TokenFilter,
    TokenFnContext,
)


def ofa_loss(logits_student, logits_teacher, temperature=1.):
    """Standard Distillation Loss (KL Divergence) - Fixed numerical stability"""
    temperature = max(temperature, 0.1)

    pred_student = F.log_softmax(logits_student / temperature + 1e-8, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    loss = - torch.sum(pred_teacher * pred_student, dim=1)

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        loss = torch.clamp(loss, min=-10.0, max=10.0)
        loss_mean = loss[~torch.isnan(loss)].mean() if (~torch.isnan(loss)).any() else 0.0
        loss = torch.where(torch.isnan(loss), torch.ones_like(loss) * loss_mean, loss)

    return loss


def detect_model_type(model):
    """Detect model type (Added Swin recognition)"""
    if hasattr(model, 'module'):
        real_model = model.module
    else:
        real_model = model

    model_name = real_model.__class__.__name__.lower()

    if 'mlp' in model_name or 'resmlp' in model_name or 'mixer' in model_name:
        return 'mlp'
    elif 'swin' in model_name:
        return 'swin'
    elif any(name in model_name for name in
             ['vit', 'deit', 't2t', 'crossvit', 'visiontransformer', 'transformer']):
        return 'vit'
    else:
        return 'cnn'


def compute_adaptive_weight(logits_s, logits_t, model_type="cnn",
                            temperature=1.0, weight_scale=1.0, max_w=5.0):
    """Adaptive weight computation"""
    temperature = max(temperature, 0.5)

    ps = F.softmax(logits_s / temperature, dim=1)
    pt = F.softmax(logits_t / temperature, dim=1)

    if model_type == "mlp":
        kl_st = F.kl_div(torch.log(ps + 1e-8), pt, reduction='none').sum(dim=1)
        kl_ts = F.kl_div(torch.log(pt + 1e-8), ps, reduction='none').sum(dim=1)
        beta = (kl_st + kl_ts) / 2.0
        weight = 1.0 + beta * weight_scale

    elif model_type == "vit" or model_type == "swin":
        m = 0.5 * (ps + pt)
        js = 0.5 * (F.kl_div(torch.log(ps + 1e-8), m, reduction='none').sum(dim=1) +
                    F.kl_div(torch.log(pt + 1e-8), m, reduction='none').sum(dim=1))
        beta = js
        weight = torch.exp(0.8 * beta * weight_scale)

    else:  # CNN
        beta = torch.norm(ps - pt, p=1, dim=1)
        beta = torch.clamp(beta, max=5.0)
        weight = torch.exp(weight_scale * beta)

    weight = torch.where(
        torch.isnan(weight) | torch.isinf(weight),
        torch.ones_like(weight),
        weight
    )
    weight = torch.clamp(weight, min=1.0, max=max_w)

    return weight


def compute_adaptive_cgr_scores(logits_student_head, w_beta, loss_vec, model_type="cnn"):
    """
    Adaptive CGR Gradient Computation   """
    original_requires_grad = logits_student_head.requires_grad
    logits_student_head.requires_grad_(True)
    w_beta.requires_grad_(True)

    try:
        # 1. Compute gradients
        grad_w = torch.autograd.grad(w_beta.sum(), logits_student_head, retain_graph=True, create_graph=True)[0]
        grad_pos = torch.autograd.grad(loss_vec.sum(), logits_student_head, retain_graph=True, create_graph=True)[0]

        # Construct original distillation gradient
        grad_neg = grad_w * loss_vec.unsqueeze(1)

        # 2. Compute original conflict degree (used for mask judgment)
        cos_sim = F.cosine_similarity(grad_neg, grad_pos, dim=1)
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0)

        # === Core Strategy: Scaling (Norm Scaling) ===
        # Scale the norm of the weight gradient to not exceed the main gradient's norm,
        # preventing the weighted gradient from dominating.
        if model_type in ["vit", "mlp", "swin", "cnn"]:
            g_pos_norm = grad_pos.norm(p=2, dim=1, keepdim=True) + 1e-6
            g_neg_norm = grad_neg.norm(p=2, dim=1, keepdim=True) + 1e-6

            # Compute scaling factor: min(1.0, |g_pos| / |g_neg|)
            scale_factor = torch.clamp(g_pos_norm / g_neg_norm, max=1.0)
            grad_neg = grad_neg * scale_factor

        # 3. Compute processed norm (used for mask threshold judgment)
        norm_neg = grad_neg.norm(p=2, dim=1)
        norm_neg = torch.nan_to_num(norm_neg, nan=1.0)

        return grad_w, grad_pos, grad_neg, norm_neg, cos_sim

    except RuntimeError as e:
        print(f"Warning: CGR calculation failed ({e}), using fallback.")
        batch_size = logits_student_head.size(0)
        device = logits_student_head.device
        return (torch.zeros_like(logits_student_head),) * 3 + (
            torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device))

    finally:
        logits_student_head.requires_grad_(original_requires_grad)


def compute_adaptive_retain_mask(cos_sim, norm_neg, epoch, total_epochs,
                                 model_type="cnn", method="hard", base_threshold=1.0):
    cos_sim = torch.nan_to_num(cos_sim, nan=0.0)
    norm_neg = torch.nan_to_num(norm_neg, nan=1.0)
    progress = epoch / max(total_epochs, 1)

    if model_type == "Acnn":
        current_threshold = max(base_threshold, 10.0 * (1 - progress))
        current_threshold = max(current_threshold, 1.0)
    else:
        current_threshold = base_threshold * (1.0 + 8.0 * (1 - progress))
        current_threshold = max(current_threshold, 2.0)

    if method == "hard" or model_type == "cnn":
        if model_type == "cnn":
            retain_mask = ((cos_sim > 0.0) & (norm_neg < current_threshold)).float()
        elif model_type == "vit" or model_type == "swin":
            retain_mask = ((cos_sim > 0.0) & (norm_neg < current_threshold)).float()
        else:
            # MLP
            retain_mask = ((cos_sim > 0.2) & (norm_neg < current_threshold)).float()
    else:
        retain_score = torch.sigmoid(cos_sim * 2.0) / (norm_neg + 0.1)
        retain_mask = (retain_score > 0.2).float()

    if retain_mask.sum() < 1e-6:
        retain_mask = torch.ones_like(retain_mask) * 0.5

    return retain_mask, current_threshold


@register_distiller
class DOFA(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(DOFA, self).__init__(student, teacher, criterion, args)

        self.student_type = detect_model_type(student)
        print(f"DOFA Init: Student Type = {self.student_type}")

        self._adjust_args_by_model_type()

        if len(self.args.ofa_eps) == 1:
            eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
            self.args.ofa_eps = eps

        # === Parameter Initialization ===
        self.args.ofa_weight_scale = getattr(self.args, 'ofa_weight_scale', 1.0)
        self.args.ofa_weight_max = getattr(self.args, 'ofa_weight_max', 5.0) 
        self.args.ofa_enable_screening = getattr(self.args, 'ofa_enable_screening', True)
        self.args.ofa_screening_method = getattr(self.args, 'ofa_screening_method', "hard")
        self.args.ofa_enable_stats = getattr(self.args, 'ofa_enable_stats', False)
        self.args.ofa_min_temperature = getattr(self.args, 'ofa_min_temperature', 0.5)

        # [Removed] CGR mode selection code, defaulting to Scaling

        self.projector = nn.ModuleDict()
        self.last_epoch = -1
        self.nan_counter = 0

        if self.args.ofa_enable_stats:
            self.weight_history = []
            self.retain_stat_list = []
        else:
            self.weight_history = None
            self.retain_stat_list = None

        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        for stage in self.args.ofa_stage:
            size_s = self.student.stage_info(stage)[1]
            _, size_s = self.student.stage_info(stage)

            use_cnn_projector = (self.student_type == "cnn") or (self.student_type == "swin")

            if use_cnn_projector:
                if len(size_s) == 3:
                    in_chans, _, _ = size_s
                else:
                    _, in_chans = size_s

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
                    nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes)
                )
            else:
                _, embed_dim = size_s
                token_num = getattr(student, "num_tokens", 0)

                target_dim = max(feature_dim_s, feature_dim_t)
                if embed_dim != target_dim:
                    align_layer = nn.Linear(embed_dim, target_dim)
                else:
                    align_layer = nn.Identity()

                if token_num > 0:
                    extract_layer = nn.Sequential(
                        TokenFilter(token_num, remove_mode=False),
                        nn.Flatten()
                    )
                else:
                    extract_layer = GAP1d()

                projector = nn.Sequential(
                    extract_layer,
                    align_layer,
                    nn.LayerNorm(target_dim),
                    nn.Linear(target_dim, args.num_classes)
                )

            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)

    def _adjust_args_by_model_type(self):
        dataset_name = getattr(self.args, 'dataset', '').lower()
        if 'imagenet' in dataset_name:
            print(f"DOFA: Detected ImageNet dataset ({dataset_name}). Setting temperature to 1.0 for all models.")
            self.args.ofa_temperature = 1.0
            if self.student_type == "mlp":
                if not hasattr(self.args, 'ofa_weight_scale'): self.args.ofa_weight_scale = 0.8
        else:
            print(f"DOFA: Detected dataset ({dataset_name}). Using architecture-specific temperatures.")
            if self.student_type == "mlp": # Swin-T -> ResMLP-S12 encountered loss NaN, so raised temperature 3->4 and adjusted weight_scale 0.8->0.5
                if not hasattr(self.args, 'ofa_temperature'): self.args.ofa_temperature = 6.0
                if not hasattr(self.args, 'ofa_weight_scale'): self.args.ofa_weight_scale = 0.8
            elif self.student_type == "vit" or self.student_type == "swin":
                if not hasattr(self.args, 'ofa_temperature'): self.args.ofa_temperature = 2.0

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student, feat_student = self.student(image, requires_feat=True)
        current_epoch = kwargs.get("epoch", 0)
        total_epochs = getattr(self.args, "epochs", 100)

        if hasattr(self.args, 'ofa_min_temperature'):
            self.args.ofa_temperature = max(self.args.ofa_temperature, self.args.ofa_min_temperature)

        ofa_losses = []

        for idx, stage in enumerate(self.args.ofa_stage):
            idx_s, _ = self.student.stage_info(stage)
            feat_s = feat_student[idx_s]

            if self.student_type == "swin":
                if len(self.student.stage_info(stage)[1]) == 3:
                    target_chans = self.student.stage_info(stage)[1][0]
                else:
                    target_chans = self.student.stage_info(stage)[1][1]

                if feat_s.dim() == 3:
                    B, L, C = feat_s.shape
                    if C == target_chans:
                        H = W = int(L ** 0.5)
                        feat_s = feat_s.permute(0, 2, 1).reshape(B, C, H, W)
                elif feat_s.dim() == 4:
                    if feat_s.shape[-1] == target_chans:
                        feat_s = feat_s.permute(0, 3, 1, 2)

            logits_student_head = get_module_dict(self.projector, stage)(feat_s)

            if torch.isnan(logits_student_head).any():
                logits_student_head = torch.nan_to_num(logits_student_head)

            loss_vec = ofa_loss(logits_student_head, logits_teacher, self.args.ofa_temperature)

            if torch.isnan(loss_vec).any():
                self.nan_counter += 1
                loss_vec = torch.nan_to_num(loss_vec, nan=loss_vec[~torch.isnan(loss_vec)].mean())

            if getattr(self.args, 'ofa_enable_screening', False):
                w_beta = compute_adaptive_weight(
                    logits_student_head, logits_teacher,
                    model_type=self.student_type,
                    temperature=self.args.ofa_temperature,
                    weight_scale=self.args.ofa_weight_scale,
                    max_w=self.args.ofa_weight_max
                )
                w_beta.requires_grad_(True)

                try:
                    # === Removed cgr_mode parameter, directly using Scaling strategy ===
                    grad_w, grad_pos, grad_neg, norm_neg, cos_sim = compute_adaptive_cgr_scores(
                        logits_student_head, w_beta, loss_vec,
                        model_type=self.student_type
                    )

                    retain_mask, dynamic_threshold = compute_adaptive_retain_mask(
                        cos_sim, norm_neg,
                        current_epoch, total_epochs,
                        model_type=self.student_type,
                        method=self.args.ofa_screening_method,
                        base_threshold=3.0
                    )

                    freeze_mask = 1.0 - retain_mask
                    w_beta_filtered = w_beta * retain_mask + w_beta.detach() * freeze_mask
                    loss_stage = (loss_vec * w_beta_filtered).mean()

                    if self.weight_history is not None:
                        self._record_statistics(current_epoch, stage, w_beta, retain_mask,
                                                cos_sim.mean().item(), norm_neg.mean().item(), dynamic_threshold)

                except RuntimeError as e:
                    print(f"Grad error: {e}")
                    loss_stage = loss_vec.mean()
            else:
                loss_stage = loss_vec.mean()

            ofa_losses.append(loss_stage)

        if current_epoch != self.last_epoch:
            self.last_epoch = current_epoch

        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * ofa_loss(logits_student, logits_teacher, self.args.ofa_temperature).mean()

        losses_dict = {"loss_gt": loss_gt, "loss_kd": loss_kd, "loss_dofa": loss_ofa}

        total_loss = sum(losses_dict.values())
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Critical: Total loss NaN. Counter: {self.nan_counter}")
            safe_loss = torch.tensor(0.0, device=logits_student.device, requires_grad=True)
            losses_dict = {k: safe_loss for k in losses_dict.keys()}

        return logits_student, losses_dict

    def _record_statistics(self, epoch, stage, weights, retain_mask, avg_cos_sim, avg_norm, threshold):
        if self.weight_history is None: return
        self.weight_history.append({
            "epoch": epoch, "stage": stage, "model_type": self.student_type,
            "w_mean": weights.mean().item(), "w_std": weights.std().item(),
            "retain_ratio": retain_mask.mean().item(), "avg_cos_sim": avg_cos_sim, "threshold": threshold
        })
        if len(self.weight_history) > 1000: self.weight_history = self.weight_history[-500:]
