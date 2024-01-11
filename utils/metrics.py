import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.spatial_color_alignment as sca_utils
from utils.spatial_color_alignment import get_gaussian_kernel, match_colors
import lpips
from utils.ssim import cal_ssim
from utils.warp import warp
from math import exp

class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse + 1e-6


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, spatial_out=False):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if spatial_out:
        ret = ssim_map
    elif size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, spatial_out=False):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.spatial_out = spatial_out

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        window = window.to(img1.device)
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,
                    spatial_out=self.spatial_out)


class SSIM(nn.Module):
    def __init__(self, boundary_ignore=None, use_for_loss=True):
        super().__init__()
        self.ssim = MSSSIM(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex')
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg')
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()
    
#################################################################################
# Compute aligned L1 loss
#################################################################################

class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        l1 = l1 + eps
        elem_ratio = l1.numel() / valid.numel()
        l1 = (l1 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return l1


class AlignedL1_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l1 = AlignedL1(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L1_all = [self.l1(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L1_loss = sum(L1_all) / len(L1_all)
        return L1_loss


def make_patches(output, labels, burst, patch_size=48):    

    stride = patch_size-(burst.size(-1)%patch_size)
    
    burst1 = burst[0].unfold(2,patch_size,stride).unfold(3,patch_size,stride).contiguous()
    burst1 = burst1.view(14,4,burst1.size(2)*burst1.size(3),patch_size,patch_size).permute(2,0,1,3,4)            
 
    output1 = output.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    output1 = output1.view(3,output1.size(2)*output1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)

    labels1 = labels.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
    labels1 = labels1[0].view(3,labels1.size(2)*labels1.size(3),patch_size*8,patch_size*8).permute(1,0,2,3)
    
    return output1, labels1, burst1

#################################################################################
# Compute aligned PSNR, LPIPS, and SSIM
#################################################################################


class AlignedPred(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        return pred_warped_m, gt, valid


class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True) * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear', align_corners=True)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = sca_utils.match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')
        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse

class AlignedL2_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        
    def forward(self, pred, gt, burst_input):
        L2_all = [self.l2(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        L2_loss = sum(L2_all) / len(L2_all)
        return L2_loss
        
class AlignedSSIM_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :]
        pred_warped_m = pred_warped_m[0, 0, :, :]

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid)

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return 1 - ssim

    def forward(self, pred, gt, burst_input):
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS_loss(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_vgg(var1, var2)
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips

class AlignedPSNR(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value    

    def psnr(self, pred, gt, burst_input):
        
        #### PSNR
        mse = self.l2(pred, gt, burst_input) + 1e-12
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input, patch_size=16)
        psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class AlignedSSIM(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        

    def ssim(self, pred, gt, burst_input):
        
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)

        gt = gt[0, 0, :, :].cpu().numpy()
        pred_warped_m = pred_warped_m[0, 0, :, :].cpu().numpy()

        mssim,ssim_map = cal_ssim(pred_warped_m*255, gt*255)
        ssim_map = torch.from_numpy(ssim_map).float()
        valid = torch.squeeze(valid.cpu())

        eps = 1e-12
        elem_ratio = ssim_map.numel() / valid.numel()
        ssim = (ssim_map * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return ssim

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input, patch_size=16)
        ssim_all = [self.ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        ssim = sum(ssim_all) / len(ssim_all)
        return ssim

class AlignedLPIPS(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.pred_warped = AlignedPred(alignment_net=alignment_net, sr_factor=sr_factor, boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def lpips(self, pred, gt, burst_input):

        #### PSNR
        pred_warped_m, gt, valid = self.pred_warped(pred, gt, burst_input)
        var1 = 2*pred_warped_m-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_alex(var1.cpu(), var2.cpu())
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt, burst_input):

        pred, gt, burst_input = make_patches(pred, gt, burst_input, patch_size=16)
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)) for p, g, bi in zip(pred, gt, burst_input)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips


###########################################################################################################
############ LPIPS ########################################################################################
def compute_lpips(pred, gt, net_type='vgg', use_gpu=True):
    """
    Compute the LPIPS score between two images.
    Args:
        pred (torch.Tensor): The predicted image tensor.
        gt (torch.Tensor): The ground truth image tensor.
        net_type (str): The network type to use for LPIPS (default is 'vgg').
        use_gpu (bool): Whether to use GPU for computation.
    Returns:
        float: The calculated LPIPS score.
    """
    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net=net_type)
    if use_gpu:
        lpips_model.cuda()
        pred = pred.cuda()
        gt = gt.cuda()

    # Normalize input images to range [-1, 1]
    pred = (pred - 0.5) * 2.0
    gt = (gt - 0.5) * 2.0

    # Compute LPIPS score
    lpips_score = lpips_model(pred, gt).item()
    return lpips_score
