import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
from deepbrain import Extractor
from skimage import measure
import itertools
from PIL import Image, ImageEnhance
from skimage.util import random_noise


def all_np(arr):
    # 拼接数组函数
    List = list(itertools.chain.from_iterable(arr))
    arr = np.array(List)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


class AdniDataSet(Dataset):

    def __init__(self, data_path, img_path, sets):
        self.data_path = data_path
        self.img_path = img_path
        self.subjects = pd.read_csv(data_path)
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.subjects['index'])

    def __getitem__(self, idx):

        # img = nibabel.load(self.img_path + str(self.subjects['Subject ID'][
        #                                            idx]) + '/corr/001_processed.nii')
        img = nibabel.load(self.img_path + str(self.subjects['index'][
                                                               idx]) + '.nii')
        assert img is not None
        # group = self.subjects['Research Group'][idx]
        # score = self.subjects['scaled_score'][idx]
        group = self.subjects['group'][idx]
        score = self.subjects['score'][idx]
        assert group is not None
        if group == 'MCI':
            label = 1
        else:
            label = 0

        img_array = self.__training_data_process__(img)
        # mask_array = self.__training_mask_process__(img, mask)

        # 2 tensor array
        img_array = self.__nii2tensorarray__(img_array)
        # mask_array = self.__nii2tensorarray__(mask_array)
        return img_array, img_array, label, score

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __scaler__(self, image):
        img_f = image.flatten()
        # find the range of the pixel values
        i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
        # clear the minus pixel values in images
        image = image - img_f[np.argmin(img_f)]
        img_normalized = np.float32(image / i_range)
        return img_normalized

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data):
        # random center crop
        data = self.__random_center_crop__(data)

        return data

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        # data = self.__scaler__(data)

        return data

    def __training_mask_process__(self, img, mask):
        mask = mask.get_data()
        mask = self.__resize_data__(mask)
        values = [v for v in all_np(mask).keys()]
        mask[mask == values[1]] = 1
        mask[mask == values[2]] = 1
        mask[mask == values[3]] = 0
        img = img.get_data()
        img = self.__resize_data__(img)
        img = self.__scaler__(img)
        data = mask * img
        data = self.__drop_invalid_range__(data)
        data = self.__resize_data__(data)
        return data

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]