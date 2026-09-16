import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
class VOC_Dataset(Dataset):
    def __init__(self,root, image_size=(512, 512), use_npy=True,  rgb=True, normalize=True,):
        super().__init__()
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.object_npy_dir = os.path.join(root, "SegmentationObjectNpy")
        self.object_png_dir = os.path.join(root, "SegmentationObject")
        self.image_size = image_size
        self.use_npy = use_npy
        self.rgb = rgb
        self.normalize = normalize
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])
        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in: {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, image_path):
        if self.rgb:
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(image_path).convert("L")
        image = image.resize(self.image_size,resample=Image.BILINEAR,)
        image = TF.to_tensor(image)
        if self.normalize:
            if self.rgb:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
            else:
                image = (image - 0.5) / 0.5
        return image.float()

    def _load_instance_mask(self, stem):
        if self.use_npy:
            mask_path = os.path.join(self.object_npy_dir, stem + ".npy")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Instance npy mask not found: {mask_path}")
            mask = np.load(mask_path)
            if mask.ndim == 3:
                if mask.shape[-1] == 1:
                    mask = mask[..., 0]
                else:
                    raise ValueError(
                        f"Instance mask should be [H, W] or [H, W, 1], got {mask.shape}"
                    )
            mask = Image.fromarray(mask.astype(np.int32), mode="I")
        else:
            mask_path = os.path.join(self.object_png_dir, stem + ".png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Instance png mask not found: {mask_path}")
            mask = Image.open(mask_path)
        mask = mask.resize(self.image_size, resample=Image.NEAREST,)
        mask = np.array(mask).astype(np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = torch.from_numpy(mask).long()
        return mask

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        stem = os.path.splitext(image_file)[0]
        image_path = os.path.join(self.image_dir, image_file)
        image = self._load_image(image_path)
        mask = self._load_instance_mask(stem)
        sample = {
            "image": image,
            "mask": mask,
            "name": stem,
        }
        return sample
