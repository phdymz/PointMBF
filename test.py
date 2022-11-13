import argparse
import os
import pickle
import random

import open3d  # noqa: F401
import torch
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import numpy as np
from models.kpfcn_config import *
from datasets.builder import build_loader, get_dataloader
from models.build_model import build_model
from models.model_util import get_grid
from utils.metrics import evaluate_correspondances, evaluate_pose_Rt
from models.kpfcn_config import architectures


seed = 77
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Set path for where to save the output dictionaries
RESULTS_DIR = '../test'



parser = argparse.ArgumentParser()
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

#dataset
DATASET = add_argument_group('Dataset')
DATASET.add_argument('--name', type=str, default='RGBD_3DMatch')
DATASET.add_argument('--RGBD_3D_ROOT', type=str, default='/media/ymz/软件/3dmatch_rgbd')     #need rewrite
DATASET.add_argument('--SCANNET_ROOT', type=str, default='/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/ScanNetRGBD')
DATASET.add_argument('--batch_size', type=int, default= 2)
DATASET.add_argument('--num_views', type=int, default= 2)
DATASET.add_argument('--view_spacing', type=int, default= 20)
DATASET.add_argument('--img_dim', type=int, default= 128)
DATASET.add_argument('--overfit', type=bool, default= False)
DATASET.add_argument('--num_workers', type=int, default= 4)
DATASET.add_argument('--input_type', type=str, default='hybrid')  #geo or hybrid
DATASET.add_argument('--filter', type=bool, default=True)
DATASET.add_argument('--pooling', type=str, default='mean', help=['max', 'mean'])
DATASET.add_argument('--complete', type=bool, default=True)
DATASET.add_argument('--voxelize', type=bool, default=False)
DATASET.add_argument('--voxel_size', type=float, default=0.025)
DATASET.add_argument('--processed', type=bool, default=False)


MODEL = add_argument_group('Model')
MODEL.add_argument('--model', type=str, default='PCReg_KPURes18_MSF')
MODEL.add_argument('--feat_dim', type=int, default=32)
MODEL.add_argument('--use_gt_vp', type=bool, default=False)

RENDER = add_argument_group('Render')
RENDER.add_argument('--render_size', type=int, default=128)
RENDER.add_argument('--points_per_pixel', type=int, default=16)
RENDER.add_argument('--radius', type=int, default=2.0)
RENDER.add_argument('--weight_calculation', type=str, default="exponential")
RENDER.add_argument('--compositor', type=str, default="norm_weighted_sum")
RENDER.add_argument('--pointcloud_source', type=str, default="other")

ALIGN = add_argument_group('Alignment')
ALIGN.add_argument('--algorithm', type=str, default="weighted_procrustes")
ALIGN.add_argument('--base_weight', type=str, default="nn_ratio")
ALIGN.add_argument('--num_correspodances', type=int, default=200)

SYSTEM = add_argument_group('System')
SYSTEM.add_argument('--RANDOM_SEED', type=int, default=8)
SYSTEM.add_argument('--NUM_WORKERS', type=int, default=6)
SYSTEM.add_argument('--TQDM', type=bool, default=True)

TRAIN = add_argument_group('Traning')
TRAIN.add_argument('--eval_step', type=int, default=5000)
TRAIN.add_argument('--num_epochs', type=int, default=12)
TRAIN.add_argument('--vis_step', type=int, default=500)
TRAIN.add_argument('--optimizer', type=str, default="Adam")
TRAIN.add_argument('--lr', type=float, default=1e-4)
TRAIN.add_argument('--momentum', type=float, default=0.9)
TRAIN.add_argument('--weight_decay', type=float, default=1e-6)
TRAIN.add_argument('--scheduler', type=str, default="constant")
TRAIN.add_argument('--rgb_render_loss_weight', type=float, default=1.0)
TRAIN.add_argument('--rgb_decode_loss_weight', type=float, default=0.0)
TRAIN.add_argument('--depth_loss_weight', type=float, default=1.0)
TRAIN.add_argument('--correspondance_loss_weight', type=float, default=0.1)
TRAIN.add_argument('--resume', type=str, default="")

EXPERIMENT = add_argument_group('Experiment')
EXPERIMENT.add_argument('--EXPname', type=str, default="URR3dmatch")
EXPERIMENT.add_argument('--rationale', type=str, default="")
EXPERIMENT.add_argument('--just_evaluate', type=bool, default=False)

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_PATH = os.path.dirname(PROJECT_PATH)
PATHS = add_argument_group('Paths')
PATHS.add_argument('--project_root', type=str, default=PROJECT_PATH)
PATHS.add_argument('--html_visual_dir', type=str, default="")
PATHS.add_argument('--tensorboard_dir', type=str, default=os.path.join(REPO_PATH, "logs", "tensor_logs"))
PATHS.add_argument('--experiments_dir', type=str, default=os.path.join(REPO_PATH, "logs", "experiments"))

KPFCN = add_argument_group('KPFCN')
num_layer = 3
if num_layer == 3:
    KPFCN.add_argument('--architectures', type=list, default=kpfcn_backbone3)
else:
    KPFCN.add_argument('--architectures', type=list, default=kpfcn_backbone4)
KPFCN.add_argument('--num_layers', type=int, default=num_layer)
KPFCN.add_argument('--deform_radius', type=float, default=5.0)
KPFCN.add_argument('--first_subsampling_dl', type=float, default=0.025)
KPFCN.add_argument('--in_feats_dim', type=int, default=1)
KPFCN.add_argument('--conv_radius', type=float, default=2.5)
KPFCN.add_argument('--num_kernel_points', type=int, default=15)
KPFCN.add_argument('--KP_extent', type=float, default=2.0)
KPFCN.add_argument('--KP_influence', type=str, default='linear')
KPFCN.add_argument('--aggregation_mode', type=str, default='sum')
KPFCN.add_argument('--fixed_kernel_points', type=str, default='center')
KPFCN.add_argument('--use_batch_norm', type=bool, default=True)
KPFCN.add_argument('--deformable', type=bool, default=False)
KPFCN.add_argument('--batch_norm_momentum', type=float, default=0.02)
KPFCN.add_argument('--use_padding', type=bool, default=True)
KPFCN.add_argument('--first_feats_dim', type=int, default=128)
KPFCN.add_argument('--in_points_dim', type=int, default=3)
KPFCN.add_argument('--modulated', type=bool, default=False)


FUSION = add_argument_group('')
FUSION.add_argument('--num_i2p', type=int, default=32)
FUSION.add_argument('--num_p2i', type=int, default=1)



TS = add_argument_group('Using teacher student to initial KPFCN')
TS.add_argument('--use_teacher', type=bool, default=False)
TS.add_argument('--rot_augment', type=bool, default=False)
TS.add_argument('--rot_factor', type=float, default=1.0)


UNET = add_argument_group('UNET')
UNET.add_argument('--num_downsample', type=int, default=3)


TRANS = add_argument_group('Transformer')
TRANS.add_argument('--use_patch_emb', type=bool, default=True)
TRANS.add_argument('--use_res', type=bool, default=True)
TRANS.add_argument('--depth', type=int, default=4)
TRANS.add_argument('--num_heads', type=int, default=4)




TEST = add_argument_group('Test')
# TEST.add_argument("model", type=str)
TEST.add_argument("--checkpoint", type=str, default='/home/ymz/下载/best_loss.pkl')  #rewrite
TEST.add_argument("--dataset", type=str, default="ScanNet", help=['ScanNet', 'RGBD_3DMatch'])      #RGBD_3DMatch
TEST.add_argument("--split", type=str, default="test")
TEST.add_argument("--boost_alignment", default=True)
TEST.add_argument("--save_dict", type=str, default='Fig38.pkl')   #rewrite
TEST.add_argument("--progress_bar", default=True)
TEST.add_argument("--no_ratio", default=False, action="store_true")
TEST.add_argument("--point_ratio", default=None, type=float)
TEST.add_argument("--num_seeds", default=None, type=int)




def evaluate_split(model, data_loader, args, dict_name=None, use_tqdm=True):
    all_metrics = {}
    all_outputs = {}

    for batch in tqdm(data_loader, disable=not use_tqdm, dynamic_ncols=True):
        batch_output, batch_metrics = forward_batch(model, batch)
        for metric in batch_metrics:
            b_metric = batch_metrics[metric].detach().cpu()
            if metric in all_metrics:
                all_metrics[metric] = torch.cat((all_metrics[metric], b_metric), dim=0)
            else:
                all_metrics[metric] = b_metric

        instances = batch_metrics["instance_id"]
        for ins in instances:
            all_outputs[ins] = {"Rt": batch_output["vp_1"].detach().cpu()}
            if "corres_01" in batch_output:
                _corres = batch_output["corres_01"]
                _corres = [_c.detach().cpu() for _c in _corres]
                all_outputs[ins]["corres"] = _corres

    # Save outputs
    if dict_name is not None:
        dict_path = os.path.join(RESULTS_DIR, dict_name)
        with open(dict_path, "wb") as handle:
            output_dict = {
                "metrics": all_metrics,
                "outputs": all_outputs,
                "args": args,
            }

            pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save metrics
    for metric in all_metrics:
        if metric == "instance_id":
            continue
        vals = all_metrics[metric]
        summary = f"{metric:30s}: {vals.mean():7.3f} +/- {vals.std():7.3f}   || "
        summary += f"median {vals.median():7.3f}"
        print(summary)

    # calculate percentage under errors
    r_acc = []
    t_acc = []
    c_acc = []
    r_err = all_metrics["vp-error_R"]
    t_err = all_metrics["vp-error_t"]
    c_err = all_metrics["chamfer"] * 1000

    for error in [5, 10, 45]:
        r_acc.append((r_err <= error).float().mean().item())

    for error in [5, 10, 25]:
        t_acc.append((t_err <= error).float().mean().item())

    for error in [1, 5, 10]:
        c_acc.append((c_err <= error).float().mean().item())

    r_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in r_acc])
    t_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in t_acc])
    c_acc_str = "  |  ".join([f"{x * 100:4.1f}" for x in c_acc])
    print(f"Rotation Accuracies:        {r_acc_str}")
    print(f"Translation Accuracies:     {t_acc_str}")
    print(f"Chamfer Accuracies:         {c_acc_str}")

    print("For latex: ")

    latex = f"{model.cfg.name} & "

    latex += f"{r_acc[0] * 100:4.1f} & "
    latex += f"{r_acc[1] * 100:4.1f} & "
    latex += f"{r_acc[2] * 100:4.1f} & "
    latex += f"{r_err.mean():4.1f} & "
    latex += f"{r_err.median():4.1f} & "

    latex += f"{t_acc[0] * 100:4.1f} & "
    latex += f"{t_acc[1] * 100:4.1f} & "
    latex += f"{t_acc[2] * 100:4.1f} & "
    latex += f"{t_err.mean():4.1f} & "
    latex += f"{t_err.median():4.1f} & "

    latex += f"{c_acc[0] * 100:4.1f} & "
    latex += f"{c_acc[1] * 100:4.1f} & "
    latex += f"{c_acc[2] * 100:4.1f} & "
    latex += f"{c_err.mean():4.1f} & "
    latex += f"{c_err.median():4.1f} & "

    print(latex)


def forward_batch(model, batch):
    num_views = 2

    for k, v in batch.items():
        if type(v) == list:
            batch[k] = [item.cuda() for item in v]
        elif type(v) in [dict, float, type(None), np.ndarray]:
            pass
        else:
            batch[k] = v.cuda()

    gt_rgb = [batch[f"rgb_{i}"].cuda() for i in range(num_views)]
    gt_dep = [batch[f"depth_{i}"].cuda() for i in range(num_views)]
    gt_vps = [batch[f"Rt_{i}"].cuda() for i in range(num_views)]
    K = batch["K"].cuda()

    output = model(batch, gt_rgb, K=K, deps=gt_dep)

    metrics = {"instance_id": batch["uid"]}

    # Model outputs
    vp_1 = output["vp_1"]
    pr_pc = output["joint_pointcloud"]
    gt_pc = model.generate_pointclouds(K, gt_dep, gt_vps)

    # Evaluate pose
    p_metrics = evaluate_pose_Rt(vp_1, gt_vps[1])
    for _k in p_metrics:
        metrics[f"{_k}"] = p_metrics[_k].detach().cpu()

    # get chamfer metrics
    cham = chamfer_distance(pr_pc.cuda(), gt_pc.cuda(), batch_reduction=None)[0].cpu()
    metrics["chamfer"] = cham

    # gather inputs
    if "corres_01" in output:
        id_c0, id_c1, c_ratio, _ = output["corres_01"]
        # Evaluate correspondaces -- should REALLY be factored out more
        depth_0 = gt_dep[0]
        B, _, H, W = depth_0.shape

        depth_0 = depth_0.view(B, 1, -1)
        id_01_0 = id_c0.unsqueeze(1)
        id_01_1 = id_c1.unsqueeze(1)

        grid = get_grid(B, H, W)
        grid = grid[:, :2].view(B, 2, -1).to(depth_0.device)
        dep01_0 = depth_0.gather(2, id_01_0)

        pix01_0 = grid.gather(2, id_01_0.repeat(1, 2, 1))
        pix01_1 = grid.gather(2, id_01_1.repeat(1, 2, 1))

        Rt_i = gt_vps[1]
        c_err_i = evaluate_correspondances(pix01_0, pix01_1, dep01_0, K, Rt_i)

        # errors cannot be larger than diagnonal (impossible .. )
        diag = (H ** 2 + W ** 2) ** 0.5
        c_err_i = c_err_i.clamp(max=diag)

        valid = (c_err_i >= 0).float()
        valid_denom = valid.sum(dim=1).clamp(min=1)
        error = (c_err_i * valid).sum(dim=1) / valid_denom
        metrics["corr-validDepth"] = valid.mean(dim=1) * 100.0
        metrics["corr-meanError"] = error

        for px_thresh in [2, 4, 10]:
            in_px = (c_err_i < px_thresh).float()
            in_px = (in_px * valid).sum(dim=1) / valid_denom
            metrics[f"corr-within{px_thresh}px"] = in_px * 100.0

    return output, metrics


if __name__ == "__main__":
    args = parser.parse_args()

    # Dataset configs to be decided
    # Dataset Parameters

    args.name = args.dataset
    args.batch_size = 4

    assert args.num_layers == args.num_downsample


    # Define model
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model_weights = checkpoint["model"]
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Step:  {checkpoint['step']}")

        # Load checkpoint
        model_cfg = checkpoint["cfg"]
        # model_cfg.defrost()
        print("===== Loaded Model Configs =====")
        print(model_cfg)
    # else:
    #     model_cfg = default_cfg.MODEL
    #     model_weights = None
    #     model_cfg.name = args.model

    # Set alignmnet performance
    if args.boost_alignment:
        assert not args.no_ratio
        assert args.num_seeds is None
        assert args.point_ratio is None

        # model_cfg.alignment.defrost()
        model_cfg.num_seeds = 100
        model_cfg.point_ratio = 0.05
        model_cfg.base_weight = "nn_ratio"

    # if args.no_ratio:
    #     model_cfg.alignment.base_weight = "uniform"

    # if args.num_seeds is not None:
    #     model_cfg.num_seeds = args.num_seeds

    # if args.point_ratio is not None:
    #     model_cfg.point_ratio = args.point_ratio

    if model_cfg.name == 'ScanNet':
        nei = np.array([30, 30, 35])
        args.voxelize = True
    else:
        nei = np.array([37, 18, 22])

    data_loader, _ = get_dataloader(args, split=args.split, neighborhood_limits = nei)

    model = build_model(model_cfg).cuda()

    if model_weights is not None:
        model.load_state_dict(model_weights)

    evaluate_split(model, data_loader, args, args.save_dict, use_tqdm=args.progress_bar)

