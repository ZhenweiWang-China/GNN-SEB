import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tools import cal_patch_index
import glob

def get_pair_path(cur_date, ref_date, LAN_path, MOD_path):
    file_LAN = np.load(LAN_path)
    file_MOD = np.load(MOD_path)

    size = file_MOD.shape

    paths = np.zeros((4, 1, size[1], size[2]))

    paths[0] = np.expand_dims(file_MOD[cur_date - 1, :, :], axis=0)  # 目标时刻MODIS路径

    paths[1] = np.expand_dims(file_LAN[cur_date - 1, :, :], axis=0)  # 目标时刻LANDSAT路径

    paths[2] = np.expand_dims(file_MOD[ref_date - 1, :, :], axis=0)  # 参考时刻MODIS路径

    paths[3] = np.expand_dims(file_LAN[ref_date - 1, :, :], axis=0)  # 参考时刻LANDSAT路径

    return paths


def load_image_pair(cur_date, ref_date, LAN_path, MOD_path):
    paths = get_pair_path(cur_date, ref_date, LAN_path, MOD_path)
    images = []
    for p in range(4):
        images.append(paths[p])

    return images  # 返回[0,1,2,3]分别对应目标时刻MODIS、LANDSAT，参考时刻MODIS、LANDSAT


def transform_image(image, flip_num, rotate_num0, rotate_num):
    image_mask = np.ones(image.shape)

    negtive_mask = np.where(image < 250)
    inf_mask = np.where(image > 350)

    image = (image - 250) / 100  # 对温度执行归一化

    image_mask[negtive_mask] = 0.0
    image_mask[inf_mask] = 0.0
    image[negtive_mask] = 0.0
    image[inf_mask] = 0.0
    image = image.astype(np.float32)

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape
    if rotate_num0 == 1:
        # -90
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        # 90
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        # 180
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask

class PatchSet2(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        self.npy_filenames = self._get_npy_filenames()  # 获取所有 .npy 文件名
        self.total_index = len(self.npy_filenames)  # 文件的数量

    def __getitem__(self, item):
        # 使用文件名列表中的文件名来加载数据
        file_name = self.npy_filenames[item]
        im = np.load(os.path.join(self.root_dir, file_name))
        im = np.array(im)
        im = im.reshape(10, 16, 16)
        images = im[:9, :, :]  # 提取前9个波段 (shape: 9, H, W)
        lab = im[9, :, :]  # 提取第10个波段 (shape: H, W)
        return images, lab

    def _get_npy_filenames(self):
        npy_files = []
        with os.scandir(self.root_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".npy"):
                    npy_files.append(entry.name)  # 添加文件名到列表
        return npy_files

    def __len__(self):
        return self.total_index  # 返回文件的数量

# Data Augment, flip、rotate

def replace_nan_in_pixel(tensor):
    # 获取 tensor 的形状
    batch_size, channels, height, width = tensor.shape

    # 对每个像元位置检查是否有 NaN 值，如果有，将该像元的所有值置为 0
    for i in range(height):
        for j in range(width):
            # 检查在该位置 (i, j) 是否有 NaN 值
            if torch.isnan(tensor[:, :, i, j]).any():
                # 将该像元所有通道的值置为 0
                tensor[:, :, i, j] = 0

    return tensor


def custom_band_normalize(arr, custom_mins, custom_maxs):
    """
    多波段数组自定义极值归一化（超出范围强制设为0或1）

    参数:
        arr: 输入数组，形状为 [波段数, 高度, 宽度]
        custom_mins: 各波段自定义最小值列表（长度=9）
        custom_maxs: 各波段自定义最大值列表（长度=9）

    返回:
        归一化后的数组，范围严格限定在[0,1]
    """
    arr = arr.astype(np.float32)
    normalized = np.zeros_like(arr)

    for band_idx in range(arr.shape[0]):
        min_val = custom_mins[band_idx]
        max_val = custom_maxs[band_idx]

        # 计算归一化值
        band_data = arr[band_idx]
        normalized_band = (band_data - min_val) / (max_val - min_val)

        # 强制超出范围的值设为0或1
        normalized_band = np.where(band_data < min_val, 0.0, normalized_band)
        normalized_band = np.where(band_data > max_val, 1.0, normalized_band)

        normalized[band_idx] = normalized_band

    return normalized


class PatchSet(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.root_dir = root_dir
        #h_list, w_list = cal_patch_index(patch_size, image_size)
        self.total_index = self._count_npy_files()
        #self.total_index = 773
        #self.total_index = len(image_dates) * len(h_list) * len(w_list)

    def __getitem__(self, item):
        #images = []
        #lab = []
        #k = 1
        im = np.load(os.path.join(self.root_dir, str(item) + '.npy'))
        #im = sorted(glob.glob(os.path.join(self.root_dir, '*.npy')))
        im = replace_nan_in_pixel(im)#将nan值变为0

        #最大值最小值归一化
        # 自定义每个波段的归一化范围（示例值）
        #albedo, DSR, DLR,emiss, B, airdensity, Ta, ra, β, ERA5LST
        custom_mins = [0, 0, 0, 0, 0, 0, 200, 0, 0, 200]
        custom_maxs = [1, 1400, 500, 1, 20, 2, 350, 200, 1, 350]
        im = custom_band_normalize(im, custom_mins, custom_maxs)
        im = np.array(im)
        #print(im.shape)
        im = im.reshape(10, 16, 16)
        #print(im.shape)
        #try:
        '''for i in range(9):
                # print(k)
                band = np.expand_dims(im[i, :, :], axis=0)
                print(band.shape)
                images.append(band)'''

        images = im[:9, :, :]  # 提取前9个波段 (shape: 9, H, W)
        #images = np.array(images, )
        lab = im[9, :, :]  # 提取第10个波段 (shape: H, W)
        #print(images.shape)
        #print(lab.shape)
        #lab = np.array(lab)
        # 创建一个与 images 列表长度相同的列表 patches，并将所有元素初始化为 None
        '''patches = [None] * len(images)
        lab = [None] * len(1)
        for i in range(len(patches)-1):
                im = images[i]
                patches[i] = im'''

        '''ima = images[10]
        print('ima', ima)
        lab[0] = ima'''

        #except:

            #band2 = np.expand_dims(im[9, :, :], axis=0)
            #print(k)
            #k = k+1


        '''patches = [None] * len(images)
        masks = [None] * len(images)

        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)
        for i in range(len(patches)):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask

        # 将参考图像对差异过大的地方去除，差异值可调整
        image_mask = np.ones(patches[2].shape)
        ref_diff = abs(patches[2] - patches[3])
        diff_indx = np.where(ref_diff > 0.05)
        image_mask[diff_indx] = 0.0

        gt_mask = masks[0] * masks[1] * masks[2] * masks[3] * image_mask'''

        return images, lab
    def _count_npy_files(self):
        count = 0
        with os.scandir(self.root_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".npy"):
                    count += 1
        #print(count)
        return count

    def __len__(self):
        return self.total_index
if __name__ == '__main__':
    # 保证初始化一致
    #train_set = PatchSet('H:\preprocessing\train', 1, 16, 16)

    x = torch.randn((1, 9, 16, 16))
    y = torch.randn((1, 9, 16, 16))
    #x = model(x)

    print(x.shape)


