import math
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
import itertools


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

    def __init__(self, data_path1, data_path2, img_path, sets):
        self.img_path = img_path
        self.subjects1 = pd.read_csv(data_path1)
        self.subjects2 = pd.read_csv(data_path2)
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
        return len(self.subjects1['index'])

    def __getitem__(self, idx):
        img1 = nibabel.load(self.img_path + str(self.subjects1['index'][
                                                    idx]) + '.nii')
        img2 = nibabel.load(self.img_path + str(self.subjects2['index'][
                                                    idx]) + '.nii')
        assert img1 is not None
        assert img2 is not None

        group1 = self.subjects1['group'][idx]
        group2 = self.subjects2['group'][idx]

        assert group1 is not None
        assert group2 is not None

        if group1 == 'AD':
            label1 = 1
        else:
            label1 = 0

        if group2 == 'AD':
            label2 = 1
        else:
            label2 = 0

        if group1 == group2:
            label3 = 0
        else:
            label3 = 1

        img_array1 = self.__training_data_process__(img1, self.subjects1['index'][
            idx])
        img_array2 = self.__training_data_process__(img2, self.subjects2['index'][
            idx])
        # 2 tensor array
        img_array1 = self.__nii2tensorarray__(img_array1)
        img_array2 = self.__nii2tensorarray__(img_array2)

        return img_array1, img_array2, label1, label2, label3

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

    def __training_data_process__(self, data, id):
        # crop data according net input size
        # print(id)
        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        # data = self.__itensity_normalize_one_volume__(data)

        data = self.__scaler__(data)

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

