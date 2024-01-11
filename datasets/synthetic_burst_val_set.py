import torch
import cv2
import numpy as np
import pickle as pkl

from data_processing.postprocessing_functions import gamma_compression

class SyntheticBurstVal(torch.utils.data.Dataset):
    """ Synthetic burst validation set. The validation burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """
    def __init__(self, root, burst_size = None, crop_size = None, data_type="burstsr",):
        self.root = root
        self.burst_list = list(range(300))
        if burst_size:
            self.burst_size = burst_size
        else:
            self.burst_size = 14
        self.crop_size = crop_size
        self.data_type = data_type

    def __len__(self):
        return len(self.burst_list)

    def _read_burst_image(self, index, image_id):
        im = cv2.imread('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(self.root, index, image_id), cv2.IMREAD_UNCHANGED)
        if self.crop_size:   #crop for 256 sampling
            crop_size = int(self.crop_size)
            assert crop_size%2==0, "crop size must be divisible by 2"
            im = im[im.shape[0]//2-crop_size//2:im.shape[0]//2+crop_size//2, im.shape[1]//2-crop_size//2:im.shape[1]//2+crop_size//2, :]
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (2**14)
        return im_t

    def _read_gt_image(self, index):
        gt = cv2.imread('{}/gt/{:04d}/im_rgb.png'.format(self.root, index), cv2.IMREAD_UNCHANGED)
        if self.crop_size:   #crop for 256 sampling
            crop_size = int(self.crop_size)*8
            gt = gt[gt.shape[0]//2-crop_size//2:gt.shape[0]//2+crop_size//2, gt.shape[1]//2-crop_size//2:gt.shape[1]//2+crop_size//2, :]
        gt_t = (torch.from_numpy(gt.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
        return gt_t

    def _read_meta_info(self, index):
        with open('{}/gt/{:04d}/meta_info.pkl'.format(self.root, index), "rb") as input_file:
            meta_info = pkl.load(input_file)

        return meta_info

    def __getitem__(self, index):
        """ Generates a synthetic burst
            args:
                index: Index of the burst

            returns:
                burst: LR RAW burst, a torch tensor of shape
                        [14, 4, 48, 48]
                        The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
            seq_name: Name of the burst sequence
        """
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info['burst_name'] = burst_name
        # bgr -> rgb
        gt = gt * 2 - 1
        gt_bgr = gt.clone().detach()
        gt_bgr[0,...], gt_bgr[2,...] = gt[2,...], gt[0,...]

        return burst, gt_bgr, meta_info
        # return burst, gt, meta_info



class SyntheticBurstValMultiGPU(SyntheticBurstVal):
    """ Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """
    def __init__(self, root, burst_size=None, crop_size=None, data_type="burstsr", shard=0, num_shards=1,):
        self.root = root
        self.burst_list = list(range(300))
        if burst_size:
            self.burst_size = burst_size
        else:
            self.burst_size = 14
        self.crop_size = crop_size
        self.shard = shard
        self.num_shards = num_shards
        self.data_type = data_type

    def __len__(self):
        return len(self.burst_list)//self.num_shards

    def __getitem__(self, index):
        index = self.shard + index*self.num_shards
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info['burst_name'] = burst_name
        # bgr -> rgb
        gt = gt * 2 - 1
        gt_bgr = gt.clone().detach()
        # gtのch変えないほうがいい気がする
        gt_bgr[0,...], gt_bgr[2,...] = gt[2,...], gt[0,...]

        return burst, gt_bgr, meta_info


#############################################################################
############## use pre_pred #################################################


class SyntheticBurstValGT(torch.utils.data.Dataset):
    """ Synthetic burst validation set. The validation burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """
    def __init__(self, root):
        self.root = root
        self.burst_list = list(range(300))
        self.burst_size = 14

    def __len__(self):
        return len(self.burst_list)

    def _read_burst_image(self, index, image_id):
        im = cv2.imread('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(self.root, index, image_id), cv2.IMREAD_UNCHANGED)
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (2**14)
        return im_t

    def _read_gt_image(self, index):
        gt = cv2.imread('{}/gt/{:04d}/im_rgb.png'.format(self.root, index), cv2.IMREAD_UNCHANGED)
        gt_t = (torch.from_numpy(gt.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
        return gt_t

    def _read_meta_info(self, index):
        with open('{}/gt/{:04d}/meta_info.pkl'.format(self.root, index), "rb") as input_file:
            meta_info = pkl.load(input_file)
        return meta_info


    def __getitem__(self, index):
        """ Generates a synthetic burst
                args:
                    index: Index of the burst

                returns:
                    burst: LR RAW burst, a torch tensor of shape
                           [14, 4, 48, 48]
                           The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
                    seq_name: Name of the burst sequence
                """
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info['burst_name'] = burst_name

        return burst, gt, meta_info