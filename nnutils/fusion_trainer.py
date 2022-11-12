import torch
import numpy as np
from models.model_util import nn_gather
from utils.losses import get_rgb_loss
from utils.metrics import evaluate_3d_correspondances, evaluate_pose_Rt
from .trainer import BasicTrainer
from datasets.builder import get_dataloader



def batchify(x):
    return x.flatten(start_dim=1).mean(dim=1)



class Fusion_Trainer(BasicTrainer):
    def __init__(self, cfg):
        super(Fusion_Trainer, self).__init__(cfg)
        # Input setup
        self.use_gt_vp = cfg.use_gt_vp

        assert cfg.num_views == cfg.num_views
        self.num_views = cfg.num_views

        # set loss weights
        self.render_loss_weight = cfg.rgb_render_loss_weight
        self.corres_weight = cfg.correspondance_loss_weight
        self.depth_weight = cfg.depth_loss_weight
        self.train_loader, neighborhood = get_dataloader(cfg, split="train")
        self.valid_loader, _ = get_dataloader(cfg, split="valid", neighborhood_limits=neighborhood)

        # self.train_loader.dataset.__getitem__(0)
        # self.valid_loader.dataset.__getitem__(0)
        # print()

    def calculate_norm_dict(self):
        max_norm = 1e10
        norm_dict = {}
        modules = ["encode"]
        # modules = ["encode", "decode"]

        def grad_fn(name, module):
            try:
                p = module.parameters()
                _grad = torch.nn.utils.clip_grad_norm_(p, max_norm)
                norm_dict[name] = _grad.item()
            except RuntimeError:
                pass

        _model = self.model
        grad_fn("full_model", self.model)

        for m in modules:
            if hasattr(_model, m):
                grad_fn(m, getattr(_model, m))

        return norm_dict

    def forward_batch(self, batch):

        B, _, H, W = batch["rgb_0"].shape

        for k, v in batch.items():
            if type(v) == list:
                batch[k] = [item.to(self.device) for item in v]
            elif type(v) in [dict, float, type(None), np.ndarray]:
                pass
            else:
                batch[k] = v.to(self.device)

        gt_rgb = [batch[f"rgb_{i}"] for i in range(self.num_views)]
        gt_dep = [batch[f"depth_{i}"] for i in range(self.num_views)]
        gt_vps = [batch[f"Rt_{i}"] for i in range(self.num_views)]
        K = batch["K"].to(self.device)

        output = self.model(batch, gt_rgb, K, gt_dep, vps=gt_vps if self.use_gt_vp else None)

        loss, metrics = [], {}

        # calculate losses
        vis_loss = []
        geo_loss = []

        for i in range(self.num_views):
            cover_i = output[f"cover_{i}"]
            depth_i = output[f"ras_depth_{i}"]

            rgb_gt_i = gt_rgb[i]
            rgb_pr1_i = output[f"rgb_render_{i}"]

            # Appearance losses
            w1 = self.render_loss_weight
            vr1_loss_i, vr1_vis_i = get_rgb_loss(rgb_pr1_i, rgb_gt_i, cover_i)

            vr_vis_i = w1 * vr1_vis_i
            vr_loss_i = w1 * vr1_loss_i

            # depth loss - simple L1 loss
            depth_dif = (depth_i - gt_dep[i]).abs()
            depth_dif = depth_dif * (gt_dep[i] > 0).float()
            dc_loss_i = (cover_i * depth_dif).mean(dim=(1, 2, 3))

            # aggregate losses
            vis_loss.append(vr_loss_i)
            geo_loss.append(dc_loss_i)

            # Update some outputs
            output[f"rgb-l1_{i}"] = vr_vis_i.detach().cpu()

            # Add losses to metrics
            metrics[f"loss-rgb-render_{i}"] = vr1_loss_i.detach().cpu()
            metrics[f"loss-depth_{i}"] = dc_loss_i.detach().cpu()

            # Evaluate pose
            if f"vp_{i}" in output:
                p_metrics = evaluate_pose_Rt(output[f"vp_{i}"], gt_vps[i], scaled=False)
                for _k in p_metrics:
                    metrics[f"{_k}_{i}"] = p_metrics[_k].detach().cpu()

            # Evaluate correspondaces
            if f"corres_0{i}" in output:
                c_id0, c_id1, c_weight, _ = output[f"corres_0{i}"]
                input_pcs = self.model.generate_pointclouds(K, gt_dep)
                c_xyz_0 = nn_gather(input_pcs[0], c_id0)
                c_xyz_i = nn_gather(input_pcs[1], c_id1)
                vp_i = output[f"vp_{i}"]

                cor_eval, cor_pix = evaluate_3d_correspondances(
                    c_xyz_0, c_xyz_i, K, vp_i, (H, W)
                )

                output[f"corres_0{i}_pixels"] = (cor_pix[0], cor_pix[1], c_weight)

                for key in cor_eval:
                    metrics[f"{key}_{i}"] = cor_eval[key]

        # ==== Loss Aggregation ====
        vs_loss = sum(vis_loss)  # wighting already accounted for above
        dc_loss = sum(geo_loss) * self.depth_weight
        cr_loss = output["corr_loss"] * self.corres_weight
        loss = vs_loss + dc_loss + cr_loss

        # sum losses
        metrics["losses_appearance"] = vs_loss
        metrics["losses_geometric"] = dc_loss
        metrics["losses_correspondance"] = cr_loss
        metrics["losses_weight-sum"] = loss.detach().cpu()
        loss = loss.mean()

        return loss, metrics, output
