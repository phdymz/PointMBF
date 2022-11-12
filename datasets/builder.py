import os
import pickle
import torch
import numpy as np
import open3d as o3d
from functools import partial
from torch.utils.data import DataLoader
from .video_dataset import VideoDataset, VideoDataset_hybrid, VideoDataset_geo
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from scipy.spatial.transform import Rotation

# Define some important paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RGBD_3D_ROOT = None  # -- You need to define those --
SCANNET_ROOT = None  # -- You need to define those --






def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def build_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """

    if cfg.name == "ScanNet":
        root_path = cfg.SCANNET_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"make_dataset/scannet_{split}.pkl")
    elif cfg.name == "RGBD_3DMatch":
        root_path = cfg.RGBD_3D_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"make_dataset/3dmatch_{split}.pkl")
    else:
        raise ValueError("Dataset name {} not recognized.".format(cfg.name))

    with open(dict_path, "rb") as f:
        data_dict = pickle.load(f)
    dataset = VideoDataset(cfg, root_path, data_dict, split)

    # Reduce ScanNet validation size to allow for more frequent validation
    if cfg.name == "ScanNet" and split == "valid":
        dataset.instances = dataset.instances[::10]

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def build_loader(cfg, split, overfit=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    dataset = build_dataset(cfg, split, overfit)
    shuffle = (split == "train") and (not overfit)
    batch_size = cfg.batch_size

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    return loader


def get_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """

    if cfg.name == "ScanNet":
        root_path = cfg.SCANNET_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"make_dataset/scannet_{split}.pkl")
    elif cfg.name == "RGBD_3DMatch":
        root_path = cfg.RGBD_3D_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"make_dataset/3dmatch_{split}.pkl")
    else:
        raise ValueError("Dataset name {} not recognized.".format(cfg.name))

    with open(dict_path, "rb") as f:
        data_dict = pickle.load(f)

    if cfg.input_type == 'hybrid':
        dataset = VideoDataset_hybrid(cfg, root_path, data_dict, split)
    elif cfg.input_type == 'geo':
        dataset = VideoDataset_geo(cfg, root_path, data_dict, split)

    # Reduce ScanNet validation size to allow for more frequent validation
    if cfg.name == "ScanNet" and split == "valid":
        dataset.instances = dataset.instances[::10]

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def get_dataloader(cfg, split, overfit=None, neighborhood_limits=None):
    if cfg.input_type == 'geo':
        collate_fn = collate_fn_geo
    elif cfg.input_type == 'hybrid':
        collate_fn = collate_fn_hybrid

    dataset = get_dataset(cfg, split, overfit)
    shuffle = (split == "train") and (not overfit)
    batch_size = cfg.batch_size

    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, cfg, collate_fn=collate_fn)
    print("neighborhood:", neighborhood_limits)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=partial(collate_fn, config=cfg, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )

    return dataloader, neighborhood_limits



def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits



def collate_fn_geo(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    batched_rgbs_0_list = []
    batched_rgbs_1_list = []
    batched_deps_0_list = []
    batched_deps_1_list = []
    batched_Rt_0_list = []
    batched_Rt_1_list = []
    batched_K_list = []
    batched_uid = []

    src_pcd_list = []
    tgt_pcd_list = []

    for ind, batch_i in enumerate(list_data):

        src_pcd = batch_i['src_pcd']
        tgt_pcd = batch_i['tgt_pcd']

        src_pcd_list.append( src_pcd.unsqueeze(0) )
        tgt_pcd_list.append( tgt_pcd.unsqueeze(0) )

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(batch_i['src_feats'])
        batched_features_list.append(batch_i['tgt_feats'])
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

        batched_rgbs_0_list.append(batch_i['rgb_0'].unsqueeze(0))
        batched_rgbs_1_list.append(batch_i['rgb_1'].unsqueeze(0))
        batched_deps_0_list.append(batch_i['depth_0'].unsqueeze(0))
        batched_deps_1_list.append(batch_i['depth_1'].unsqueeze(0))
        batched_Rt_0_list.append(batch_i['Rt_0'].unsqueeze(0))
        batched_Rt_1_list.append(batch_i['Rt_1'].unsqueeze(0))
        batched_K_list.append(batch_i['K'].unsqueeze(0))

        batched_uid.append(batch_i['uid'])


    batched_features = torch.cat(batched_features_list)
    batched_points = torch.cat(batched_points_list)
    batched_lengths = torch.tensor(batched_lengths_list).int()
    batched_uid = torch.tensor(batched_uid)

    src_pcd_list = torch.vstack(src_pcd_list)
    tgt_pcd_list = torch.vstack(tgt_pcd_list)
    batched_rgbs_0_list = torch.vstack(batched_rgbs_0_list)
    batched_rgbs_1_list = torch.vstack(batched_rgbs_1_list)
    batched_deps_0_list = torch.vstack(batched_deps_0_list)
    batched_deps_1_list = torch.vstack(batched_deps_1_list)
    batched_Rt_0_list = torch.vstack(batched_Rt_0_list)
    batched_Rt_1_list = torch.vstack(batched_Rt_1_list)
    batched_K_list = torch.vstack(batched_K_list)


    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    # construt kpfcn inds
    for block_i, block in enumerate(config.architectures):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []


    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'rgb_0': batched_rgbs_0_list,
        'rgb_1': batched_rgbs_1_list,
        'depth_0': batched_deps_0_list,
        'depth_1': batched_deps_1_list,
        'Rt_0': batched_Rt_0_list,
        'Rt_1': batched_Rt_1_list,
        'K': batched_K_list,
        'uid': batched_uid
    }

    return dict_inputs


def filter_holes(point):
    idx = (point != torch.zeros(3)).sum(-1) > 2
    return point[idx]

def voxel_downsample(point, voxel_size):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point.numpy())
    point_cloud = point_cloud.voxel_down_sample(voxel_size)

    return torch.from_numpy(np.array(point_cloud.points))


def collate_fn_hybrid(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    batched_points_list_img = [ [] for i in range(config.num_layers)]
    batched_lengths_list_img = [ [] for i in range(config.num_layers)]

    batched_rgbs_0_list = []
    batched_rgbs_1_list = []
    batched_deps_0_list = []
    batched_deps_1_list = []
    batched_Rt_0_list = []
    batched_Rt_1_list = []
    batched_K_list = []
    batched_uid = []
    batched_src_pcd_img = []
    batched_tgt_pcd_img = []

    for ind, batch_i in enumerate(list_data):
        src_pcd = batch_i['src_pcd_list']['src_pcd_0']
        tgt_pcd = batch_i['tgt_pcd_list']['tgt_pcd_0']
        if config.filter:
            src_pcd = filter_holes(src_pcd)
            tgt_pcd = filter_holes(tgt_pcd)
        else:
            src_pcd = src_pcd.squeeze()
            tgt_pcd = tgt_pcd.squeeze()
        if config.voxelize:
            src_pcd = voxel_downsample(src_pcd, config.voxel_size)
            tgt_pcd = voxel_downsample(tgt_pcd, config.voxel_size)

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(torch.ones(len(src_pcd),1))
        batched_features_list.append(torch.ones(len(tgt_pcd),1))
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))
        batched_uid.append(batch_i['uid'])
        batched_src_pcd_img.append(batch_i['src_pcd_list']['src_pcd_0'])
        batched_tgt_pcd_img.append(batch_i['tgt_pcd_list']['tgt_pcd_0'])

        for i in range(config.num_layers):
            batched_points_list_img[i].append(batch_i['src_pcd_list']['src_pcd_{}'.format(i)].squeeze(0))
            batched_points_list_img[i].append(batch_i['tgt_pcd_list']['tgt_pcd_{}'.format(i)].squeeze(0))
            batched_lengths_list_img[i].append((batch_i['src_pcd_list']['src_pcd_{}'.format(i)]).shape[-2])
            batched_lengths_list_img[i].append((batch_i['tgt_pcd_list']['tgt_pcd_{}'.format(i)]).shape[-2])


        batched_rgbs_0_list.append(batch_i['rgb_0'].unsqueeze(0))
        batched_rgbs_1_list.append(batch_i['rgb_1'].unsqueeze(0))
        batched_deps_0_list.append(batch_i['depth_0'].unsqueeze(0))
        batched_deps_1_list.append(batch_i['depth_1'].unsqueeze(0))
        batched_Rt_0_list.append(batch_i['Rt_0'].unsqueeze(0))
        batched_Rt_1_list.append(batch_i['Rt_1'].unsqueeze(0))
        batched_K_list.append(batch_i['K'].unsqueeze(0))


    batched_features = torch.cat(batched_features_list)
    batched_points = torch.cat(batched_points_list)
    batched_lengths = torch.tensor(batched_lengths_list).int()
    batched_uid = torch.tensor(batched_uid)
    batched_src_pcd_img = torch.cat(batched_src_pcd_img)
    batched_tgt_pcd_img = torch.cat(batched_tgt_pcd_img)

    for i in range(config.num_layers):
        batched_points_list_img[i] = torch.cat(batched_points_list_img[i])
        batched_lengths_list_img[i] = torch.tensor(batched_lengths_list_img[i]).int()


    batched_rgbs_0_list = torch.vstack(batched_rgbs_0_list)
    batched_rgbs_1_list = torch.vstack(batched_rgbs_1_list)
    batched_deps_0_list = torch.vstack(batched_deps_0_list)
    batched_deps_1_list = torch.vstack(batched_deps_1_list)
    batched_Rt_0_list = torch.vstack(batched_Rt_0_list)
    batched_Rt_1_list = torch.vstack(batched_Rt_1_list)
    batched_K_list = torch.vstack(batched_K_list)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    p2i_list = []
    i2p_list = []


    # construt kpfcn inds
    for block_i, block in enumerate(config.architectures):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architectures) - 1 and not ('upsample' in config.architectures[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])
            p2i_list.append(batch_neighbors_kpconv(batched_points_list_img[layer], batched_points,
                                   batched_lengths_list_img[layer], batched_lengths,  r, config.num_p2i).long())
            i2p_list.append(batch_neighbors_kpconv(batched_points, batched_points_list_img[layer],
                                                   batched_lengths, batched_lengths_list_img[layer], r, config.num_i2p).long())

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])


        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []


    rots = []
    trans = []
    for i in range(len(input_batches_len[0])):
        euler_ab = np.random.rand(3) * np.pi * 2 / config.rot_factor  # anglez, angley, anglex
        rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix().astype(np.float32)
        rots.append(torch.from_numpy(rot_ab))

    if config.rot_augment:
        for i in range(config.num_layers):
            start_length = 0
            end_length = 0
            for j, batch in enumerate(input_batches_len[i]):
                end_length += batch
                point_raw = input_points[i][start_length:end_length]
                point_raw = torch.matmul(rots[j], point_raw.T).T
                input_points[i][start_length:end_length] = point_raw
                start_length += batch

    dict_inputs = {
        'points_img': [batched_src_pcd_img, batched_tgt_pcd_img],
        'p2i_list': p2i_list,
        'i2p_list': i2p_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'rgb_0': batched_rgbs_0_list,
        'rgb_1': batched_rgbs_1_list,
        'depth_0': batched_deps_0_list,
        'depth_1': batched_deps_1_list,
        'Rt_0': batched_Rt_0_list,
        'Rt_1': batched_Rt_1_list,
        'K': batched_K_list,
        'uid': batched_uid
    }

    return dict_inputs