from mpi4py import MPI
from torch.utils.data import DataLoader

from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGBMultiGPU
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.burstsr_dataset import BurstSRDataset

def load_data_bsr(
    *,
    data_dir,
    batch_size,
    burst_size=8,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    train_zurich_raw2rgb = ZurichRAW2RGBMultiGPU(root=data_dir,  split='train', shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size(),)
    train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=burst_size, crop_sz=256)

    if deterministic:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True
            )
    while True:
        yield from train_loader



def load_data_bsr_real(
    *,
    data_dir,
    batch_size,
    burst_size=8,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    train_dataset = BurstSRDataset(data_dir, burst_size=burst_size, crop_sz=32, split='train')


    if deterministic:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True
            )
    while True:
        yield from train_loader

