from __future__ import annotations
from pathlib import Path
import jax
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List
import lz4.frame
import pickle
import torch
import json, math, random, zipfile, io
from torch.utils.data import DataLoader

import json, numpy as np, cv2
import os
from pathlib import Path
from pycocotools import mask as mutils  # pip install pycocotools
from tqdm import tqdm

import random
DATA_DIR = Path('/mnt/ssd/hojin/efficient_planning/dataset/estimation/20_5_data')
OUTDIR = Path('/mnt/ssd/hojin/efficient_planning/dataset/estimation/20_5_data_coco')
# OUTDIR = Path('20_5_data_coco_test')
OUTDIR.mkdir(exist_ok=True, parents=True)

TRAIN_ZIP       = OUTDIR / "train2017.zip"
VAL_ZIP         = OUTDIR / "val2017.zip"
ANNS_ZIP        = OUTDIR / "panoptic_annotations_trainval2017.zip"

IMG_TRAIN_DIR   = Path("./train2017")             # inside ZIP
IMG_VAL_DIR     = Path("./val2017")
PAN_TRAIN_DIR   = Path("panoptic_train2017")
PAN_VAL_DIR     = Path("panoptic_val2017")
ANNS_DIR        = Path("annotations")          # no leading ./

CATEGORIES = [
    {"id": 1, "name": "environment", "supercategory": "scene", "isthing": 0},
    {"id": 2, "name": "object", "supercategory": "scene", "isthing": 1},
]

BITS_LOCAL = 5                      # 5 bits → 0-31
MAX_LOCAL  = (1 << BITS_LOCAL) - 1  # 31


random.seed(0)

def pytree_collate(batch: List[Dict]):
  """Simple collation for numpy pytree instances"""
  data = jax.tree_util.tree_map(lambda *x: np.stack(x, 0), *batch)
  return data
def del_if_exist(data, key):
    if key in data:
        del data[key]
def preprocess_datapoint(data):
    extended = False
    del_if_exist(data, "RobotInfo_urdf_path")
    del_if_exist(data, "RobotInfo_q")
    del_if_exist(data, "RobotInfo_posquat")
    del_if_exist(data, "RobitLinkInfo_mesh_names")
    del_if_exist(data, "RobotLinkInfo_mesh_names")
    del_if_exist(data, "RobotLinkInfo_scales")
    del_if_exist(data, "RobotLinkInfo_posquat")
    del_if_exist(data, "RobotLinkInfo_posquats")
    del_if_exist(data, "EnvInfo_mesh_name")
    del_if_exist(data, "EnvInfo_obj_posquats")
    del_if_exist(data, "EnvInfo_scale")
    del_if_exist(data, "EnvInfo_uid_list")
    if len(data['rgbs'].shape) == 4:
        extended = True
        data = jax.tree_util.tree_map(lambda x: x[None], data)
    if 'obj_info' not in data:
        obj_info = {k.replace('ObjInfo_',''):data[k] for k in data.keys() if 'ObjInfo' in k}
        data['obj_info'] = obj_info
        data = {k:data[k] for k in data.keys() if 'ObjInfo' not in k}
    if 'cam_info' not in data:
        cam_info_key = ['cam_posquats', 'cam_intrinsics']
        cam_info = {k:data[k] for k in cam_info_key}
        data['cam_info'] = cam_info
        for ck in cam_info_key:
            del data[ck]
    ## this is from nvisii render function error
    res = np.abs(data['cam_info']['cam_intrinsics'][...,4:6] - data['cam_info']['cam_intrinsics'][...,:2] * 0.5).max()
    assert res < 1e-5, f"cam_intrinsics is not correct! {res}"
    data['cam_info']['cam_intrinsics'][...,4:6] = data['cam_info']['cam_intrinsics'][...,:2] * 0.5
    del_if_exist(data, "depths")
    del_if_exist(data, "table_params")
    del_if_exist(data, "robot_params")
    del_if_exist(data, "uid_clss")
    del_if_exist(data, "nvren_info")
    del_if_exist(data['obj_info'], "obj_cvx_verts_padded")
    del_if_exist(data['obj_info'], "obj_cvx_faces_padded")
    del_if_exist(data['obj_info'], "uid_list")
    if 'env_info' in data:
        del_if_exist(data['env_info'], "uid_list")
    if extended:
        data = jax.tree_util.tree_map(lambda x: x.squeeze(0), data)
    return data

class EstimatorDatasetsplit(Dataset):
    """Estimator dataset tailored for FLAX"""
    def __init__(
            self,
            data_dir_path:Path,
            ds_obj_no:int,
            ds_type='train',
            filter_data:int = -1,
    ):
        """Entire data is already loaded in the memory"""
        self.filter_data = filter_data # 20250625
        self.ds_type = ds_type
        dataset_filenames = list(sorted(data_dir_path.glob(f'*.lz4')))
        # filter max num
        dataset_filenames_ = []
        for df in dataset_filenames:
            base_name = str(df.name).split('.')[0]
            if base_name.split('_')[-1] == 'robot':
                ds_max_obj_no = int(base_name.split('_')[-2])
            else:
                ds_max_obj_no = int(base_name.split('_')[-1])
            if ds_max_obj_no == ds_obj_no:
                dataset_filenames_.append(df)
        dataset_filenames = dataset_filenames_
        if self.ds_type == 'single_ds':
            ## test with single dataset!!
            self.dataset_filenames = [dataset_filenames[2] for df in range(16*100)]
            print('single dataset test')
        elif self.ds_type == 'single_ds_test':
            self.dataset_filenames = [dataset_filenames[2] for df in range(16*2)]
            print('single dataset test')
        else:
            # random.shuffle(dataset_filenames)
            # shuffle with numpy
            np.random.default_rng(0).shuffle(dataset_filenames)
            if self.ds_type == 'test':
                self.dataset_filenames = dataset_filenames[-len(dataset_filenames)//20:]
            elif self.ds_type == 'debug':
                self.dataset_filenames = dataset_filenames[:10000]
            elif self.ds_type == 'debug_2':
                self.dataset_filenames = dataset_filenames[-10000:]
            else:
                self.dataset_filenames = dataset_filenames[:-len(dataset_filenames)//20]
            shelf_ds_no = len([df for df in self.dataset_filenames if 'shelf' in str(df.name).split('_')])
            table_ds_no = len(self.dataset_filenames) - shelf_ds_no
            print(f"ds type: {self.ds_type} // fn loaded: {len(self.dataset_filenames)} // shelf no: {shelf_ds_no} // table no: {table_ds_no}")
    def __len__(self):
        """Dataset size"""
        return len(self.dataset_filenames)
    def __getitem__(self, index):
        """All operations will be based on tree_map"""
        # Index an item (squeeze batch dim)
        fname = self.dataset_filenames[index]
        if self.filter_data > 0 and int(Path(fname).name.split("_")[0]) < self.filter_data:
            # print(f"Skipping old dataset {fname}")
            return self.__getitem__(index+1)
        # with lz4.frame.open(str(fname), "r") as fp:
        #     bytes_data = fp.read()
        # data = pickle.loads(bytes_data)
        # data = preprocess_datapoint(data)
        try:
            with lz4.frame.open(str(fname), "r") as fp:
                bytes_data = fp.read()
            data = pickle.loads(bytes_data)
            data = preprocess_datapoint(data)
        except:
            print(f"Error in processing datapoint from {fname}")
            # remove file from directory
            if fname.exists():
                os.remove(fname)
            return self.__getitem__(index+1)
        return data

def encode_id_to_rgb(segment_id: int) -> tuple[int,int,int]:
    """COCO panoptic RGB encoding (24-bit big-endian)."""
    r =  segment_id          & 255
    g = (segment_id >>  8)   & 255
    b = (segment_id >> 16)   & 255
    return r, g, b

def write_png_from_id_map(id_map: np.ndarray) -> bytes:
    """uint32 id_map → PNG bytes in memory (RGB, uint8)."""
    h, w = id_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = ((id_map >> (8 * c)) & 255).astype(np.uint8)
    is_ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert is_ok
    return buf.tobytes()

def bbox_from_mask(binary: np.ndarray) -> list[float]:
    """xmin, ymin, width, height (COCO floats)."""
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return [float(xmin), float(ymin), float(xmax - xmin + 1),
            float(ymax - ymin + 1)]

def convert_split(loader, img_zip_path, pan_dir_in_zip):
    """
    Iterate through loader, write RGB into train/val ZIP,
    write PNGs into an *inner* ZIP, then place that inner ZIP
    under ./annotations/ in the outer panoptic archive.
    """
    images_json, ann_json = [], []
    next_global_sid = 1

    # ---- prepare the inner ZIP in memory ---------------------------------
    inner_buf  = io.BytesIO()
    inner_zip  = zipfile.ZipFile(inner_buf, "w", compression=zipfile.ZIP_STORED)

    with zipfile.ZipFile(img_zip_path, "w", compression=zipfile.ZIP_STORED) as zip_img, \
         zipfile.ZipFile(ANNS_ZIP,  "a", compression=zipfile.ZIP_STORED)   as zip_pan:

        img_id = 0
        for i, sample in tqdm(enumerate(loader), total=len(loader),
                              desc=f"converting {img_zip_path.stem}"):

            rgbs = sample["rgbs"][0].astype(np.uint8)
            segs = np.asarray(sample["seg"][0], dtype=np.int64)

            for rgb, seg in zip(rgbs, segs):
                img_id += 1

                # check if there is any segmentation, if not, skip
                if seg.max() == -2:
                    print(f"Skipping image {img_id} with no segmentation")
                    continue


                file_name  = f"{img_id:012d}.jpg"
                png_name   = pan_dir_in_zip / f"{img_id:012d}.png"   # e.g. panoptic_train2017/…

                # ---------- RGB → outer train/val ZIP ----------------------
                h, w = rgb.shape[:2]
                _, jpg_bytes = cv2.imencode(
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                arcname = (IMG_TRAIN_DIR / file_name) if "train" in pan_dir_in_zip.name \
                                                     else (IMG_VAL_DIR  / file_name)
                zip_img.writestr(arcname.as_posix(), jpg_bytes)

                # ---------- build id_map & segments_info -------------------
                id_map = np.zeros_like(seg, dtype=np.uint32)
                segments_info = []

                env_mask = seg == -1
                local_idx = 0

                if env_mask.any():
                    # sid = next_global_sid;  next_global_sid += 1
                    # sid = (img_id << 8) | instance_cnt   # img_id in high 16 bits, inst# in low 8
                    # instance_cnt += 1

                    sid = (img_id << BITS_LOCAL) | local_idx
                    local_idx += 1
                    id_map[env_mask] = sid
                    segments_info.append({
                        "id": sid, "category_id": 1,
                        "area": int(env_mask.sum()),
                        "bbox": bbox_from_mask(env_mask), "iscrowd": 0,
                    })

                for inst_id in np.unique(seg[seg > -1]):
                    obj_mask = seg == inst_id
                    if local_idx >= MAX_LOCAL:
                        raise RuntimeError("more than 255 instances in one image")

                    sid = (img_id << BITS_LOCAL) | local_idx
                    local_idx += 1

                    id_map[obj_mask] = sid
                    segments_info.append({
                        "id": sid, "category_id": 2,
                        "area": int(obj_mask.sum()),
                        "bbox": bbox_from_mask(obj_mask), "iscrowd": 0,
                    })

                # ---------- PNG → inner ZIP --------------------------------
                inner_zip.writestr(png_name.as_posix(),
                                   write_png_from_id_map(id_map))

                # ---------- JSON blocks ------------------------------------
                images_json.append({
                    "id": img_id, "file_name": file_name,
                    "height": h, "width": w,
                })
                ann_json.append({
                    "image_id": img_id,
                    "file_name": png_name.as_posix(),
                    "segments_info": segments_info,
                })

        # close inner ZIP to flush bytes, then drop it into the outer ZIP
        inner_zip.close()
        inner_arcname = f"annotations/{pan_dir_in_zip.name}.zip"
        zip_pan.writestr(inner_arcname, inner_buf.getvalue())

    return images_json, ann_json


def dump_json(zip_pan: zipfile.ZipFile, path_in_zip: Path, data: dict):
    buf = io.BytesIO(json.dumps(data, indent=2).encode())
    zip_pan.writestr(str(path_in_zip.as_posix()), buf.getvalue())

if __name__ == '__main__':
    train_set = EstimatorDatasetsplit(DATA_DIR, 20, "train")
    val_set = EstimatorDatasetsplit(DATA_DIR, 20, "test")
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        num_workers=0,
        collate_fn=pytree_collate,
        pin_memory=False, # Only for torch
        shuffle=False
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        num_workers=0,
        collate_fn=pytree_collate,
        pin_memory=False, # Only for torch
        shuffle=False
    )
    print("Pass 1/2 - TRAIN")
    train_images, train_anns = convert_split(train_loader, TRAIN_ZIP, PAN_TRAIN_DIR)

    print("Pass 2/2 - VAL")
    val_images, val_anns = convert_split(val_loader, VAL_ZIP, PAN_VAL_DIR)

    with zipfile.ZipFile(ANNS_ZIP, "a") as zip_pan:
        dump_json(zip_pan, ANNS_DIR / "panoptic_train2017.json", {
            "images": train_images,
            "annotations": train_anns,
            "categories": CATEGORIES,
        })
        dump_json(zip_pan, ANNS_DIR / "panoptic_val2017.json", {
            "images": val_images,
            "annotations": val_anns,
            "categories": CATEGORIES,
        })

    print("✓ Conversion complete - files ready for your COCOPanoptic dataloader.")
