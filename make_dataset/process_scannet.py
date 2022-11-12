import numpy as np
import torch
from torchvision import transforms as transforms
from torch import nn as nn
from typing import List, Optional
from datasets.complete import fill_in_multiscale
import os
import pickle
from tqdm import tqdm
from PIL import Image, ImageFile
import png
root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/ScanNetRGBD'

def dict_to_instances(data_dict, strided):
    view_spacing = 20
    num_views = 2
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
                frames = frames[:: view_spacing]
                stride = 1
            else:
                stride = view_spacing

            num_frames = len(frames)

            for i in range(num_frames - num_views * stride):
                f_ids = []
                for v in range(num_views):
                    f_ids.append(frames[i + v * stride])

                # Hacky way of getting source to be in the middle for triplets
                mid = num_views // 2
                f_ids = f_ids[mid:] + f_ids[:mid]
                instances.append([cls_id, s_id, tuple(f_ids)])

    return instances

def get_img(path, bbox=None):
    path = os.path.join(root, path)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            if bbox is not None:
                img = img.crop(box=bbox)
            return np.array(img)

def save_png(save_root, im):
    im[im < 0] = 0
    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(save_root, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


if __name__ == "__main__":

    split = 'valid'
    data_dict = dict_path = os.path.join(f"scannet_{split}.pkl")
    with open(dict_path, "rb") as f:
        data_dict = pickle.load(f)
    strided = split in ["valid", "test"]

    instances = dict_to_instances(data_dict, strided)

    for index in tqdm(range((len(instances)))):
        cls_id, s_id, f_ids = instances[index]
        s_instance = data_dict[cls_id][s_id]["instances"]

        for i, id_i in enumerate(f_ids):
            dep = get_img(s_instance[id_i]["dep_path"])
            cam_scale = 1000
            dpt = dep.copy() / cam_scale
            dpt, _ = fill_in_multiscale(
                    dpt, extrapolate=False, blur_type='bilateral',
                    show_process=False, max_depth=8.0
            )
            dep = dpt * cam_scale

            # save_root = os.path.join(root, s_instance[id_i]["dep_path"].replace('.png', '.npy'))
            # if not os.path.exists(save_root):
            #     np.save(save_root, dep)
            save_root = os.path.join(root, s_instance[id_i]["dep_path"].replace('.png', '_complete.png'))
            if not os.path.exists(save_root):
                save_png(save_root, dep)





