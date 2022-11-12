import torch
from torch import nn as nn
from models.block import UnaryBlock
from utils.transformations import transform_points_Rt
from .alignment import align
from .backbones import *
from .correspondence import get_correspondences
from .model_util import get_grid, grid_to_pointcloud, points_to_ndc
from .renderer import PointsRenderer
from monai.networks.nets import UNet
from pytorch3d.ops.knn import knn_points
from models.correspondence import calculate_ratio_test, get_topk_matches
import warnings
warnings.filterwarnings("ignore")

def project_rgb(pc_0in1_X, rgb_src, renderer):
    # create rgb_features
    B, _, H, W = rgb_src.shape
    rgb_src = rgb_src.view(B, 3, H * W)
    rgb_src = rgb_src.permute(0, 2, 1).contiguous()

    # Rasterize and Blend
    project_0in1 = renderer(pc_0in1_X, rgb_src)

    return project_0in1["feats"]


#baseline
class PCReg(nn.Module):
    def __init__(self, cfg):
        super(PCReg, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False
        self.encode = ResNetEncoder(chan_in, feat_dim, pretrained)
        self.decode = ResNetDecoder(feat_dim, 3, nn.Tanh(), pretrained)

        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.align_cfg = cfg

    def forward(self, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1 : n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1 : n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1 : n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def forward_pcreg(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        vps = []
        cor_loss = []
        for i in range(1, n_views):
            corr_i = get_correspondences(
                P1=pcs_F[0],
                P2=pcs_F[i],
                P1_X=pcs_X[0],
                P2_X=pcs_X[i],
                num_corres=self.num_corres,
                ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
            )
            Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"corres_0{i}"] = corr_i
            output[f"vp_{i}"] = Rt_i

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        return output

    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True,)
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
            feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


#replace recurrent resblock in baseline with 2 resnet blocks
from .backbones import ResNetEncoder_modified



class PCReg_KPURes18_MSF(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']
        pcs_F = self.map(feat_i, feat_p, batch)

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None






