import os.path

import numpy as np
import torch
from torchvision import transforms as transforms
from torch import nn as nn
from typing import List, Optional
from datasets.complete import fill_in_multiscale
from models.model_util import get_grid, grid_to_pointcloud, points_to_ndc
from utils.transformations import transform_points_Rt

from .abstract import AbstractDataset

class VideoDataset(AbstractDataset):
    """
        Dataset for video frames. It samples tuples of consecutive frames
    """

    def __init__(self, cfg, root_path, data_dict, split):
        name = cfg.name
        super(VideoDataset, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split
        self.num_views = cfg.num_views
        self.view_spacing = cfg.view_spacing
        self.image_dim = cfg.img_dim
        self.data_dict = data_dict
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
            ]
        )
        self.complete = cfg.complete

        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with.
        #  An example of strided vs non strided for a view spacing of 10:
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided)

        # Print out dataset stats
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        # Applies a center crop to the input image and resizes it to image_dim
        h, w = dep.shape
        left = int(round((w - h) / 2.0))
        right = left + h
        dep = dep[:, left:right]
        dep = torch.Tensor(dep[None, None, :, :]).float()
        dep = torch.nn.functional.interpolate(dep, (self.image_dim, self.image_dim))[0]
        return dep

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # Read in separate instances
        P_rel = s_instance[f_ids[0]]["extrinsic"]
        K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
        for i, id_i in enumerate(f_ids):
            rgb = self.get_rgb(s_instance[id_i]["rgb_path"])

            # calculate crop offset to rescale K; should be the same for all images
            smaller_dim = rgb.height
            crop_offset = (rgb.width - rgb.height) / 2

            # transform and save rgb
            rgb = self.rgb_transform(rgb)
            output["path_" + str(i)] = s_instance[id_i]["rgb_path"]
            output["rgb_" + str(i)] = rgb

            # Resize depth and scale to meters according to ScanNet Docs
            # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
            dep = self.get_img(s_instance[id_i]["dep_path"])
            cam_scale = 1000
            dpt = dep.copy() / cam_scale
            if self.complete:
                dpt, _ = fill_in_multiscale(
                    dpt, extrapolate=False, blur_type='bilateral',
                    show_process=False, max_depth=8.0
                )
            dep = dpt * cam_scale
            dep = self.dep_transform(dep)
            dep = dep / 1000.0
            output["depth_" + str(i)] = dep
            output["median_depth_" + str(i)] = dep[dep > 0].median()

            # Extract P from extrinsics; we can about P between the frame pair
            P = s_instance[id_i]["extrinsic"]
            P = torch.tensor(np.linalg.inv(P) @ P_rel).float()
            output["Rt_" + str(i)] = P[:3, :]

        # -- Transform K to handle image resize and crop
        K[0, 2] -= crop_offset  # handle cropped width
        K[:2, :] *= self.image_dim / smaller_dim  # handle resizing
        output["K"] = torch.tensor(K).float()

        return output

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances

class VideoDataset_geo(AbstractDataset):
    """
        Dataset for video frames. It samples tuples of consecutive frames
    """

    def __init__(self, cfg, root_path, data_dict, split):
        name = cfg.name
        super(VideoDataset_geo, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split
        self.use_padding = cfg.use_padding
        self.num_downsample = cfg.num_downsample
        self.num_views = cfg.num_views
        self.view_spacing = cfg.view_spacing
        self.image_dim = cfg.img_dim
        self.data_dict = data_dict
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
            ]
        )

        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with.
        #  An example of strided vs non strided for a view spacing of 10:
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided)

        # Print out dataset stats
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        # Applies a center crop to the input image and resizes it to image_dim
        h, w = dep.shape
        left = int(round((w - h) / 2.0))
        right = left + h
        dep = dep[:, left:right]
        dep = torch.Tensor(dep[None, None, :, :]).float()
        dep = torch.nn.functional.interpolate(dep, (self.image_dim, self.image_dim))[0]
        return dep

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # Read in separate instances
        P_rel = s_instance[f_ids[0]]["extrinsic"]
        K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
        for i, id_i in enumerate(f_ids):
            rgb = self.get_rgb(s_instance[id_i]["rgb_path"])

            # calculate crop offset to rescale K; should be the same for all images
            smaller_dim = rgb.height
            crop_offset = (rgb.width - rgb.height) / 2

            # transform and save rgb
            rgb = self.rgb_transform(rgb)
            output["path_" + str(i)] = s_instance[id_i]["rgb_path"]
            output["rgb_" + str(i)] = rgb

            # Resize depth and scale to meters according to ScanNet Docs
            # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
            dep = self.get_img(s_instance[id_i]["dep_path"])
            dep = self.dep_transform(dep)
            dep = dep / 1000.0
            output["depth_" + str(i)] = dep

            # Extract P from extrinsics; we can about P between the frame pair
            P = s_instance[id_i]["extrinsic"]
            P = torch.tensor(np.linalg.inv(P) @ P_rel).float()
            output["Rt_" + str(i)] = P[:3, :]

        # -- Transform K to handle image resize and crop
        K[0, 2] -= crop_offset  # handle cropped width
        K[:2, :] *= self.image_dim / smaller_dim  # handle resizing
        output["K"] = torch.tensor(K).float()

        src_pcd, tgt_pcd = self.generate_pointclouds(output["K"].unsqueeze(0), [output['depth_' + str(i)].unsqueeze(0) for i in range(2)])
        output['src_pcd'] = src_pcd
        output['tgt_pcd'] = tgt_pcd
        output['src_feats'] = torch.ones(src_pcd.shape[0],1)
        output['tgt_feats'] = torch.ones(tgt_pcd.shape[0],1)

        return output

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances

    def generate_pointclouds(self, K, deps):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        src_pcd, tgt_pcd= self.grid_to_pcd(K_inv = K_inv, depth = deps, grid = grid)

        # if not self.use_padding:
        #     mask = [pcs_X[i][:,2] >0  for i in range(n_views)]
        #     for i in range(n_views):
        #         pcs_X[i] = pcs_X[i][mask[i]]

        return src_pcd, tgt_pcd

    def grid_to_pcd(self, K_inv, depth, grid):

        B, _, H, W = depth[0].shape

        # Apply inverse projection
        src_raw_points = depth[0] * grid
        tgt_raw_points = depth[1] * grid

        src_points = src_raw_points.view(B, 3, H * W)
        tgt_points = tgt_raw_points.view(B, 3, H * W)
        src_points = K_inv.bmm(src_points)
        tgt_points = K_inv.bmm(tgt_points)

        src_pcd = src_points.permute(0, 2, 1)
        tgt_pcd = tgt_points.permute(0, 2, 1)

        return src_pcd.squeeze(), tgt_pcd.squeeze()


class VideoDataset_hybrid(AbstractDataset):
    """
        Dataset for video frames. It samples tuples of consecutive frames
    """

    def __init__(self, cfg, root_path, data_dict, split):
        name = cfg.name
        super(VideoDataset_hybrid, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split
        self.use_padding = cfg.use_padding
        self.num_downsample = cfg.num_downsample
        self.num_views = cfg.num_views
        self.view_spacing = cfg.view_spacing
        self.image_dim = cfg.img_dim
        self.data_dict = data_dict
        self.processed = cfg.processed
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
            ]
        )
        self.complete = cfg.complete
        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with.
        #  An example of strided vs non strided for a view spacing of 10:
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided)

        # Print out dataset stats
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        # Applies a center crop to the input image and resizes it to image_dim
        h, w = dep.shape
        left = int(round((w - h) / 2.0))
        right = left + h
        dep = dep[:, left:right]
        dep = torch.Tensor(dep[None, None, :, :]).float()
        dep = torch.nn.functional.interpolate(dep, (self.image_dim, self.image_dim))[0]
        return dep

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # Read in separate instances
        P_rel = s_instance[f_ids[0]]["extrinsic"]
        K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
        for i, id_i in enumerate(f_ids):
            rgb = self.get_rgb(s_instance[id_i]["rgb_path"])

            # calculate crop offset to rescale K; should be the same for all images
            smaller_dim = rgb.height
            crop_offset = (rgb.width - rgb.height) / 2

            # transform and save rgb
            rgb = self.rgb_transform(rgb)
            output["path_" + str(i)] = s_instance[id_i]["rgb_path"]
            output["rgb_" + str(i)] = rgb

            # Resize depth and scale to meters according to ScanNet Docs
            # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
            if self.processed:
                # dep = np.load(os.path.join(self.root, s_instance[id_i]["dep_path"].replace('.png', '.npy')))
                if self.name == 'RGBD_3DMatch':
                    dep = self.get_img(os.path.join(self.root, s_instance[id_i]["dep_path"].replace('.depth', '.complete')))
                else:
                    dep = self.get_img(os.path.join(self.root, s_instance[id_i]["dep_path"].replace('.png', '_complete.png')))
                dep[dep > 8000] = 0
            else:
                dep = self.get_img(s_instance[id_i]["dep_path"])
                cam_scale = 1000
                dpt = dep.copy() / cam_scale
                if self.complete:
                    dpt, _ = fill_in_multiscale(
                        dpt, extrapolate=False, blur_type='bilateral',
                        show_process=False, max_depth=8.0
                    )
                dep = dpt * cam_scale
            dep = self.dep_transform(dep)
            dep = dep / 1000.0
            output["depth_" + str(i)] = dep

            # Extract P from extrinsics; we can about P between the frame pair
            P = s_instance[id_i]["extrinsic"]
            P = torch.tensor(np.linalg.inv(P) @ P_rel).float()
            output["Rt_" + str(i)] = P[:3, :]

        # -- Transform K to handle image resize and crop
        K[0, 2] -= crop_offset  # handle cropped width
        K[:2, :] *= self.image_dim / smaller_dim  # handle resizing
        output["K"] = torch.tensor(K).float()

        src_pcd_list, tgt_pcd_list = self.generate_pointclouds(output["K"].unsqueeze(0), [output['depth_' + str(i)].unsqueeze(0) for i in range(2)])
        output['src_pcd_list'] = src_pcd_list
        output['tgt_pcd_list'] = tgt_pcd_list
        # output['src_feats'] = torch.ones(src_pcd_list['src_pcd_0'].shape[1],1)
        # output['tgt_feats'] = torch.ones(tgt_pcd_list['tgt_pcd_0'].shape[1],1)

        return output

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances

    def generate_pointclouds(self, K, deps):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        src_pcd_list, tgt_pcd_list = self.multi_scale_grid_to_pcd(K_inv = K_inv, depth = deps, grid = grid)

        # if not self.use_padding:
        #     mask = [pcs_X[i][:,2] >0  for i in range(n_views)]
        #     for i in range(n_views):
        #         pcs_X[i] = pcs_X[i][mask[i]]

        return src_pcd_list, tgt_pcd_list

    def multi_scale_grid_to_pcd(self, K_inv, depth, grid):

        B, _, H, W = depth[0].shape

        # Apply inverse projection
        src_raw_points = depth[0] * grid
        tgt_raw_points = depth[1] * grid

        src_pcd_list = {}
        tgt_pcd_list = {}

        for i in range(self.num_downsample):
            src_points = src_raw_points.view(B, 3, H * W)
            tgt_points = tgt_raw_points.view(B, 3, H * W)
            src_points = K_inv.bmm(src_points)
            tgt_points = K_inv.bmm(tgt_points)

            src_pcd_list['src_pcd_' + str(i)] = src_points.permute(0, 2, 1)
            tgt_pcd_list['tgt_pcd_' + str(i)] = tgt_points.permute(0, 2, 1)

            if i < (self.num_downsample - 1):
                if self.cfg.pooling == 'max':
                    src_raw_points = nn.functional.max_pool2d(src_raw_points, 2, 2)
                    tgt_raw_points = nn.functional.max_pool2d(tgt_raw_points, 2, 2)
                elif self.cfg.pooling == 'mean':
                    src_raw_points = nn.functional.avg_pool2d(src_raw_points, 2, 2)
                    tgt_raw_points = nn.functional.avg_pool2d(tgt_raw_points, 2, 2)

                H = H // 2
                W = W // 2


        return src_pcd_list, tgt_pcd_list