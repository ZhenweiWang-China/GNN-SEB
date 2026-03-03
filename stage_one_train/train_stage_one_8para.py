import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
#from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
#from sewar import rmse
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datasets import PatchSet, load_image_pair, transform_image
from tools import cal_patch_index, ReconstructionLoss, Average, ssim_numpy
from tools.loss import ssim, calculate2_ssim
from basicsr.stage_one_8parameter import first_GNN
#from stage_one import stage_one
#from base_models.swin_transformer import
import os
from torch.utils.data import Dataset
import numpy as np


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
    #print(custom_mins[0], custom_maxs[0])
    #print(arr.shape)

    for band_idx in range(arr.shape[0]):
        #print(custom_mins[band_idx], custom_maxs[band_idx])
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

def replace_nan_in_pixel(tensor):
    # 获取 tensor 的形状
    #print(tensor.shape)
    #batch_size, channels, height, width = tensor.shape
    height, width = tensor.shape
    # 对每个像元位置检查是否有 NaN 值，如果有，将该像元的所有值置为 0
    for i in range(height):
        for j in range(width):
            # 检查在该位置 (i, j) 是否有 NaN 值
            if torch.isnan(tensor[i, j]).any():
                # 将该像元所有通道的值置为 0
                tensor[i, j] = 0

    return tensor

def replace_nan_vectorized(arr):
    # 默认将NaN替换为0，正/负无穷保留原值
    #用numpy将nan变为0
    return np.nan_to_num(arr, nan=0)

class PatchSet2(Dataset):
    def __init__(self, root_dir, image_dates, image_size, patch_size):
        super().__init__()  # 使用Python3的super()方式初始化父类
        self.root_dir = root_dir
        self.npy_filenames = self._get_npy_filenames()  # 获取所有 .npy 文件名
        self.total_index = len(self.npy_filenames)  # 文件的数量

    def __getitem__(self, item):
        # 使用文件名列表中的文件名来加载数据
        file_name = self.npy_filenames[item]
        im = np.load(os.path.join(self.root_dir, file_name))
        #print(im.shape)
        im = im.reshape(10, 32, 32)
        #im = replace_nan_in_pixel(im)  # 将nan值变为0
        im = replace_nan_vectorized(im)

        # 最大值最小值归一化
        # 自定义每个波段的归一化范围（示例值）
        # albedo, DSR, DLR,emiss, B, airdensity, Ta, ra, β, ERA5LST
        custom_mins = [0, 0, 0, 0, 0, 0, 200, 0, 0, 200]
        custom_maxs = [1, 1400, 500, 1, 20, 2, 350, 200, 1, 350]
        im = custom_band_normalize(im, custom_mins, custom_maxs)

        im = np.array(im)
        mask=np.isnan(im)
        im[mask]=0

        # 即排除 index 6, 11, 14, 15
        keep_indices = [i for i in range(10) if i not in [6, 9]]  # 从0开始，

        images = im[keep_indices, :, :]  # 提取前12个波段 (shape: 9, H, W),提取数字-1个波段

        #images = im[:9, :, :]  # 提取前9个波段 (shape: 9, H, W)
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

def compute_rmse_gpu(image1, image2):
    # 确保输入图像是numpy数组并转换为torch tensor
    image1 = torch.tensor(image1, dtype=torch.float32).cuda()  # 迁移到GPU
    image2 = torch.tensor(image2, dtype=torch.float32).cuda()  # 迁移到GPU

    # 计算RMSE
    rmse = torch.sqrt(torch.mean((image1 - image2) ** 2))

    return rmse.item()  # 返回标量值

def get_args_parser():
    parser = argparse.ArgumentParser(description='Train stage one')
    parser.add_argument('--image_size', default=[1918, 3941], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=32, type=int, help='training sample size')
    parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
    #1e-4   1e-5    1e-6   1e-7   1e-8
    parser.add_argument('--lr', default=1e-4, type=float, help='parameters learning rate')
    parser.add_argument('--data_number', default=43, type=int, help='The number of images for train and test')
    parser.add_argument('--test_index', default=17, type=int, help='The index of image for test')
    parser.add_argument('--LAN_path', default=r'D:\SwinT_FYMOLA\landsat\lansat_fine.npy',
                        help='All landsat images storage location')
    parser.add_argument('--MOD_path', default=r'D:\SwinT_FYMOLA\MODIS\modis_fine.npy',
                        help='All MODIS images storage location')
    parser.add_argument('--train_dir', default=r'J:\stage_one_sample\ERA_var_zhouLST32', help='Sample storage location')
    parser.add_argument('--test_dir', default=r'J:\stage_one_sample\ERA_var_zhouLST32\test', help='Sample storage location')
    parser.add_argument('--save_dir', default=r'stage_one_8parameter', help='Save training parameters')
    return parser


def draw_fig(list, name, epoch, save_dir):
    print('draw_fig', list)
    x1 = range(1, epoch + 1)
    y1 = list
    if name == "loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'r-.d')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.savefig(save_dir + '/Train_loss_stage1.png')
        plt.show()
    elif name == "rmse":
        plt.cla()
        plt.title('RMSE vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'g-.+')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.savefig(save_dir + '/Train_RMSE_stage1.png')
        plt.show()
    elif name == "ssim":
        plt.cla()
        plt.title('SSIM  vs. epoch', fontsize=20)
        plt.plot(x1, y1, 'b-.*')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train SSIM', fontsize=20)
        plt.savefig(save_dir + '\Train_SSIM_stage1.png')
        plt.show()


def train(opt, train_dates, test_dates):
    train_set = PatchSet2(opt.train_dir, train_dates, opt.image_size, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batch_size, shuffle=True)
    #print(train_loader)
    #print(train_set)
    model = first_GNN(opt.patch_size, opt.patch_size)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('There are %d trainable parameters in stage one.' % n_params)

    loss_function = ReconstructionLoss()

    model.cuda()
    loss_function.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0)  # transformer使用该学习率
    scheculer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_RMSE = 100.0
    best_epoch = -1
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_loss = []
    epoch_rmse = []
    epoch_ssim = []  # 存储每次迭代的平均loss，以及验证集的RMSE和SSIM
    for epoch in tqdm(range(opt.num_epochs)):
        model.train()
        g_loss, batch_time = Average(), Average()
        batches = len(train_loader)

        for item, (images, lable) in tqdm(enumerate(train_loader)):

            t_start = timer()
            #print('images', images.shape)
            #print('lable',lable.shape)
            #images = torch.stack(images)
            #lable = torch.stack(lable)

            #images = images.cuda().to(torch.float64)
            images = images.cuda().to(torch.float32)
            lable = lable.cuda()
            #print('images', images)
            #lable = lable.unsqueeze(dim=1).to(torch.float64)
            lable = lable.unsqueeze(dim=1).to(torch.float32)
            #lable = lable.unsqueeze(dim=1)
            '''print('images', images)
            print('images', images.shape)
            print('lable', lable)
            print('lable', lable.shape)'''
            '''data = data.cuda()
            target = target.cuda()
            ref_lr = ref_lr.cuda()
            ref_target = ref_target.cuda()
            gt_mask = gt_mask.float().cuda()'''

            predict_LST = model(images)
            #print('predict_LST', predict_LST.shape)
            #print('lable', lable.shape)
            #predict_LST = predict_LST*150+200
            #lable = lable*150+200
            #print('predict_LST = model(images)', predict_LST)
            #predict_LST = predict_LST.squeeze(1)

            optimizer.zero_grad()
            #print('predict_LST', predict_LST.shape)
            #print('lable', lable.shape)
            l_total = loss_function(predict_LST, lable)

            l_total.backward()
            optimizer.step()
            # optimizer.zero_grad()

            g_loss.update(l_total.cpu().item())

            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))

            if item % 100 == 99:
                print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
                    epoch + 1, opt.num_epochs, item + 1, batches, g_loss.avg, batch_time.avg,
                ))
        print('[%d/%d][%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, batches, batches, g_loss.avg, batch_time.avg,
        ))

        final_ssim, final_rmse = test(model, test_dates, opt)
        #print(g_loss, final_rmse, final_ssim)
        epoch_loss.append(g_loss.avg)
        epoch_rmse.append(final_rmse)
        epoch_ssim.append(final_ssim)

        scheculer.step(final_rmse)
        if final_rmse < best_RMSE:
            best_RMSE = final_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir + '/stage_one_parameters.pth')

        torch.save(model.state_dict(), save_dir + '/epoch_%d.pth' % (epoch + 1))
        print('Best Epoch is %d' % (best_epoch + 1), 'SSIM is %.4f' % best_RMSE)
        print('------------------')

    data_save = np.zeros([len(epoch_loss), 3])
    data_save[:, 0] = np.array(epoch_loss)
    data_save[:, 1] = np.array(epoch_rmse)
    data_save[:, 2] = np.array(epoch_ssim)

    result_save = pd.DataFrame(data=data_save, index=None, columns=['loss', 'rmse', 'ssim'])
    result_save.to_csv(save_dir + '/stage_one_result.csv')

    draw_fig(epoch_loss, 'loss', opt.num_epochs, save_dir)
    draw_fig(epoch_ssim, 'ssim', opt.num_epochs, save_dir)
    draw_fig(epoch_rmse, 'rmse', opt.num_epochs, save_dir)


def test(model, test_dates, opt):
    model.eval()
    train_set = PatchSet2(opt.test_dir, train_dates, opt.image_size, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batch_size, shuffle=True)

    epoch_rmse = []
    epoch_ssim = []  # 存储每次迭代的平均loss，以及验证集的RMSE和SSIM

    for item, (images, lable) in tqdm(enumerate(train_loader)):
        images = images.cuda().to(torch.float32)
        lable = lable.cuda().unsqueeze(dim=1).to(torch.float32)
        #lable = lable.cuda().to(torch.float32)
        #print('images', images.shape)
        #print('lable', lable.shape)
        predict_LST = model(images)
        predict_LST = predict_LST * 150 + 200
        lable = lable * 150 + 200
        #print('predict_LST', predict_LST.shape)
        #l_total = loss_function(predict_LST, lable)
        rmse_value = compute_rmse_gpu(predict_LST, lable)
        epoch_rmse.append(rmse_value)

        # 计算SSIM
        #ssim_index = ssim(predict_LST, lable)
        ssim_index =ssim(predict_LST, lable)
        epoch_ssim.append(ssim_index)
    #h_list, w_list = cal_patch_index(opt.patch_size, opt.image_size)
    #print('epoch_ssim', epoch_ssim)
    #print('epoch_rmse', epoch_rmse)
    #epoch_ssim = epoch_ssim.cpu()  # 将CUDA张量移至CPU
    #final_ssim = np.mean(epoch_ssim)
    #epoch_rmse = epoch_rmse.cpu()  # 将CUDA张量移至CPU
    #final_rmse =np.mean(epoch_rmse)
    final_ssim =sum(epoch_ssim) / len(epoch_ssim)
    final_ssim = final_ssim.item()
    final_rmse = sum(epoch_rmse) / len(epoch_rmse)
    #final_rmse = final_rmse.item()
    #print('final_ssim', final_ssim)
    #print('final_rmse', final_rmse)

    try:
        f2 = open(r'train_stage_one_8para.txt', 'a+')
        # new_line2 = ''.join(i)
        f2.write(str(final_rmse) + ', ' + str(final_ssim) + '\r\n')
        f2.close()

    except:
        pass

    '''print('rmse_value', rmse_value, 'ssim_index', ssim_index)
    print('epoch_rmse', epoch_rmse, 'epoch_rmse', epoch_rmse)
    print('final_ssim', final_ssim, 'final_rmse', final_rmse)'''
    '''final_ssim = 0.0
    final_rmse = 0.0'''
    '''for cur_date in test_dates:
        ref_date = cur_date - 1
        images = load_image_pair(cur_date, ref_date, opt.LAN_path, opt.MOD_path)

        output_image = np.zeros(images[1].shape)
        image_mask = np.ones(images[1].shape)
        for i in range(4):
            negtive_mask = np.where(images[i] < 250)
            inf_mask = np.where(images[i] > 350)
            image_mask[negtive_mask] = 0
            image_mask[inf_mask] = 0

        for i in range(len(h_list)):
            for j in range(len(w_list)):
                h_start = h_list[i]
                w_start = w_list[j]

                input_lr = images[0][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]
                ref_lr = images[2][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]
                ref_hr = images[3][:, h_start: h_start + opt.patch_size, w_start: w_start + opt.patch_size]

                flip_num = 0
                rotate_num0 = 0
                rotate_num = 0  # 预测时不进行增强操作
                input_lr, _ = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                ref_lr, _ = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                ref_hr, _ = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                input_lr = input_lr.unsqueeze(0).cuda()
                ref_lr = ref_lr.unsqueeze(0).cuda()
                ref_hr = ref_hr.unsqueeze(0).cuda()

                down, up, output = model(ref_lr, ref_hr, input_lr)
                output = output.squeeze()

                fill_in, patch_in = test_fill_index(i, j, h_start, w_start, h_list, w_list, opt.patch_size)

                output_image[:, fill_in[0]: fill_in[1], fill_in[2]: fill_in[3]] = \
                    output[patch_in[0]: patch_in[1], patch_in[2]: patch_in[3]].cpu().detach().numpy()

        output_image = (output_image) * 100 + 250
        real_im = images[1] * image_mask
        real_output = output_image * image_mask
        # plt.imshow(real_output[0],cmap='jet')
        # plt.show()
        final_ssim = ssim_numpy(real_im[0] - 270, real_output[0] - 270, val_range=40)
        final_rmse = rmse(real_im[0], real_output[0])
        print('[%s/%s] RMSE: %.4f SSIM: %.4f' % (
            cur_date, ref_date, final_rmse, final_ssim))'''

    return final_ssim, final_rmse


if __name__ == '__main__':
    # 保证初始化一致
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.deterministic = True
    opt = get_args_parser().parse_args()

    train_dates = []
    test_dates = []
    all_index = [i + 1 for i in range(opt.data_number)]  # 训练与测试的图像总数,下标从1开始
    for i in range(len(all_index)):
        if all_index[i] not in [opt.test_index]:  # 挑选一张作为测试，与样本裁剪时一致
            train_dates.append(all_index[i])  # 待训练的时刻
        else:
            test_dates.append(all_index[i])  # 待预测的个时刻

    train(opt, train_dates, test_dates)
    '''x = torch.randn((1, 9, 16, 16))
    y = torch.randn((1, 9, 16, 16))
    #x = model(x)

    print(compute_rmse_gpu(x, y))'''