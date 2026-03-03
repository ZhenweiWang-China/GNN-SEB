import torch
import torch.nn as nn
import torch.nn.functional as F
from .pytorch_ssim import msssim, ssim
#from skimage.metrics import structural_similarity as ssim
import numpy as np
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.pixel_cri = CharbonnierLoss()

    def forward(self, pred, target):
        #pred = pred.astype(np.float32)
        #target = target.astype(np.float32)
        #loss = self.pixel_cri(pred, target) + (1 - msssim(pred, target, val_range=1, normalize='relu'))
        #loss = self.pixel_cri(pred, target)
        loss = self.pixel_cri(pred, target) + (1-ssim(pred, target))
        return loss


def calculate2_ssim(img1, img2):
    """
    计算两幅图像的 SSIM（结构相似性指数）。

    参数:
        img1: numpy.ndarray or torch.Tensor, 第一张图像 (H, W) 或 (H, W, C)
        img2: numpy.ndarray or torch.Tensor, 第二张图像 (H, W) 或 (H, W, C)

    返回:
        ssim_value: float, SSIM 值
    """
    img1 = img1.permute(1,0, 2, 3)
    img1 = img1.squeeze(0)
    #img2 = img2.permute(1,0, 2, 3)
    img2 = img2.squeeze(0)
    print(img1.shape, img2.shape)
    # 如果输入是 PyTorch Tensor，则转换为 NumPy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # 确保图像为浮点数类型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算 SSIM
    #ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min(), multichannel=True)
    ssim_value = ssim(img1, img2,
                      #data_range= 1.0,
                      #channel_axis=2,
                      win_size=16)
    return ssim_value


'''def ssim(x, y, window_size=11, size_average=True, full=False):
    # Create Gaussian window
    def gaussian_window(window_size, sigma):
        _1D = torch.linspace(-window_size // 2 + 1, window_size // 2, window_size)
        x, y = torch.meshgrid(_1D, _1D)
        d = x ** 2 + y ** 2
        window = torch.exp(-d / (2 * sigma ** 2))
        window = window / window.sum()
        return window

    def calculate_ssim(x, y, window):
        # Constants
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.conv2d(x, window, padding=window.size(2) // 2, groups=1)
        mu_y = F.conv2d(y, window, padding=window.size(2) // 2, groups=1)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=window.size(2) // 2, groups=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=window.size(2) // 2, groups=1) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window.size(2) // 2, groups=1) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map

    # Create window
    window = gaussian_window(window_size, 1.5).view(1, 1, window_size, window_size).to(x.device)

    # Ensure the window is in float32 for consistency
    window = window.float()

    # Compute SSIM map
    ssim_map = calculate_ssim(x, y, window)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])'''


class CharbonnierLoss(nn.Module):
    """Charbonnier损失函数的深度网络可以更好地处理异常值，比L2损失函数高能提高超分辨率(SR)性能 """
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(target, -pred)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
if __name__ == '__main__':
    # 保证初始化一致
    #train_set = PatchSet('H:\preprocessing\train', 1, 16, 16)

    x = torch.randn(2, 1, 16, 16).to(torch.float32).cuda()  # Ensure input tensor is float32
    y = torch.randn(2, 1, 16, 16).to(torch.float32).cuda()  # Ensure input tensor is float32
    #x = model(x)
    #z= calculate_ssim(x, y)
    z=ssim(x, y)
    print(z)