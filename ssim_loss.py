import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# class SSIMLoss(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.create_window(window_size, self.channel)
#
#     def create_window(self, window_size, channel):
#         _1D_window = torch.exp(-torch.linspace(-(window_size // 2), window_size // 2, window_size) ** 2 / 2)
#         _1D_window /= _1D_window.sum()
#         _2D_window = _1D_window.unsqueeze(1).mm(_1D_window.unsqueeze(1).t()).float().unsqueeze(0).unsqueeze(0)
#         self.window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#
#     def ssim(self, img1, img2):
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
#
#         # Gaussian kernel
#         mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
#         mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
#
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#
#         sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
#         sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
#
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#         return ssim_map.mean() if self.size_average else ssim_map.mean([1, 2, 3])
#
#     def forward(self, img1, img2):
#         if img1.size() != img2.size():
#             raise ValueError('Input images must have the same dimensions.')
#
#         _, channel, _, _ = img1.size()
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             self.create_window(self.window_size, channel)
#             window = self.window
#             window = window.to(img1.device)
#             self.channel = channel
#
#         return 1 - self.ssim(img1, img2)
#
#
# # Example usage
# if __name__ == "__main__":
#     ssim_loss = SSIMLoss(window_size=11, size_average=True)
#
#     # Two example time series, shape：(batch_size, channel, height, width)
#     img1 = torch.rand((1, 1, 256, 256))  # 单通道灰度图像
#     img2 = torch.rand((1, 1, 256, 256))
#
#     # Calculate SSIM loss
#     loss = ssim_loss(img1, img2)
#     print(f'SSIM Loss (Single Channel): {loss.item()}')
#
#     img1_rgb = torch.rand((1, 3, 256, 256))  # 三通道 RGB 图像
#     img2_rgb = torch.rand((1, 3, 256, 256))
#
#     # Calculate SSIM loss
#     loss_rgb = ssim_loss(img1_rgb, img2_rgb)
#     print(f'SSIM Loss (RGB): {loss_rgb.item()}')

import torch
import torch.nn.functional as F
from torch import nn


class SSIMLossForSequence(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLossForSequence, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.create_window(window_size)

    def create_window(self, window_size):
        _1D_window = torch.exp(-torch.linspace(-(window_size // 2), window_size // 2, window_size) ** 2 / 2)
        _1D_window /= _1D_window.sum()
        self.window = _1D_window.float().unsqueeze(0).unsqueeze(0).cuda()  # shape: (1, 1, window_size)

    def ssim(self, seq1, seq2,device='cpu',gamma=1.0, conf=None):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        padding = self.window_size // 2

        # Ensure the input sequences are 3D for conv1d operation
        seq1 = seq1.unsqueeze(1)  # shape: (batch_size, 1, sequence_length)
        seq2 = seq2.unsqueeze(1)  # shape: (batch_size, 1, sequence_length)

        mu1 = F.conv1d(seq1, self.window, padding=padding, groups=1).to(device)
        mu2 = F.conv1d(seq2, self.window, padding=padding, groups=1).to(device)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv1d(seq1 * seq1, self.window, padding=padding, groups=1) - mu1_sq
        sigma2_sq = F.conv1d(seq2 * seq2, self.window, padding=padding, groups=1) - mu2_sq
        sigma12 = F.conv1d(seq1 * seq2, self.window, padding=padding, groups=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if conf is not None:
            # Interpolation of conf to the same length as ssim_map
            if conf.size(-1) != ssim_map.size(-1):
                conf = F.interpolate(conf.unsqueeze(1).float(),
                                     size=ssim_map.size(-1),
                                     mode='linear',
                                     align_corners=False).squeeze(1)
            omega_r = 1.0 / (1.0 + torch.exp(-gamma * conf))  # (B, L')
            # Normalize omega_r to prevent batch scale inconsistency
            omega_r = omega_r / omega_r.sum(dim=-1, keepdim=True)
            score = (ssim_map * omega_r.unsqueeze(1)).sum(dim=-1)  # 加权求和
        else:
            score = ssim_map.mean(-1) if not self.size_average else ssim_map.mean()

        return score

    def forward(self, seq1, seq2,device):
        if seq1.size() != seq2.size():
            raise ValueError('Input sequences must have the same dimensions.')

        _, len_seq = seq1.size()
        if len_seq < self.window_size:
            raise ValueError('Sequence length must be greater than window size.')

        return 1 - self.ssim(seq1, seq2,device)


# Example usage
if __name__ == "__main__":
    ssim_loss = SSIMLossForSequence(window_size=11, size_average=True)

    # Two example time series, shape:(batch_size, sequence_length)
    seq1 = torch.rand((4, 100)).cuda()  # batch大小为4，序列长度为100
    seq2 = torch.rand((4, 100)).cuda()
    device = torch.device("cuda")
    # Calculate SSIM loss
    loss = ssim_loss(seq1, seq2,device)
    print(f'SSIM Loss for Sequence: {loss.item()}')
