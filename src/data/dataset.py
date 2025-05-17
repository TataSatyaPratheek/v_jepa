import torch
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.data import LabeledVideoDataset

class VideoDataset(Dataset):
    def __init__(self, paths, clip_sampler, decode_audio=False):
        self.dataset = LabeledVideoDataset(
            paths,
            clip_sampler,
            transform=self._transform,
            decode_audio=decode_audio,
            decoder="pyav",
            multithreaded_io=True
        )
        
    def _transform(self, x):
        return x["video"].to(torch.float16)  # Half-precision

def create_loader(dataset, batch_size=2, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="mps",
        prefetch_factor=2,
        multiprocessing_context="fork" if num_workers > 0 else None
    )
