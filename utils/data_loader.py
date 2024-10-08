import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from python_patch_extractor import PatchExtractor as PE


ImageFile.LOAD_TRUNCATED_IMAGES = True

class BasicLoader(Dataset):
    def __init__(self, db: pd.DataFrame, subsample: float = None, transformer: A.BasicTransform = ToTensorV2(),
                 aug_transformers: List[A.BasicTransform] = None, patch_size: int = 32, patch_number: int = 5,
                 pe_stride: int = 4):
        super(BasicLoader, self).__init__()

        if subsample is not None:
            db = db.sample(frac=subsample)
        self.db = db
        self.transformer = transformer
        self.aug_transformers = aug_transformers
        self.patch_size = patch_size
        self.patch_number = patch_number
        self.pe_stride = pe_stride
        self.pe = PE.PatchExtractor(dim=(self.patch_size, self.patch_size, 3), stride=(self.pe_stride, self.pe_stride, 3))

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        # note: we are considering progressive indexing of db, not actual index
        record = self.db.iloc[idx]
        cropper = A.RandomCrop(width=self.patch_size, height=self.patch_size, always_apply=True, p=1.)
        pe = PE.PatchExtractor((self.patch_size, self.patch_size, 3))


        # open the image in RGB
        img = np.asarray(Image.open(record.image))
        # Consider only original dimensions (top-left corner) of tampered image with inpainting (dall-e)
        if record.tamper_method == 'dall_e':
            img = img[:256, :256]

        if img.ndim < 3:
            # print('Gray scale image, converting to RGB')
            img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img2[:, :, 0] = img
            img2[:, :, 1] = img
            img2[:, :, 2] = img
            img = img2.copy()

        if img.shape[2] > 3:
            # print('Omitting alpha channel')
            img = img[:, :, :3]

        # extract patches from RGB image
        # patches = [cropper(image=img)['image'] for x in range(self.patch_number)]

        if self.aug_transformers is None:
            trans = A.Compose(self.transformer)
        else:
            trans = A.Compose(self.aug_transformers + self.transformer)

        transf_patch_list = []
        patch_label_list = []
        patches_ext = self.pe.extract(img)
        number_of_patches = patches_ext.shape[0]
        patches = patches_ext.reshape((patches_ext.shape[0]*patches_ext.shape[1], self.patch_size, self.patch_size, 3))
        for patch in patches:
            transf_patch_list.append(trans(image=patch)['image'])
            patch_label_list.append(record['mask'])

        # create batch:
        trans_img = torch.stack(transf_patch_list, dim=0)
        # target = torch.tensor(patch_label_list)
        target = patch_label_list

        return trans_img, target, number_of_patches
