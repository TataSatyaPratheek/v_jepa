import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
import logging
import av
import threading
import queue
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipSampler

from ..utils.device import get_device_manager
from ..utils.memory import empty_cache

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VideoDatasetConfig:
    """Configuration for video dataset."""
    path: str
    clip_duration: float = 2.0  # seconds
    frame_rate: float = 16.0  # fps
    clip_sampler: str = "random"  # "random", "uniform", or "constant_clips_per_video"
    decode_audio: bool = False
    decoder: str = "pyav"  # "pyav" or "decord"
    transform_backend: str = "torchvision"  # "torchvision" or "cv2"
    use_half_precision: bool = True
    clip_overlap: float = 0.0  # Overlap between clips [0.0, 1.0]
    multithreaded_io: bool = True
    preload_to_memory: bool = False  # Preload videos to memory for faster access
    preload_max_videos: int = 100  # Max videos to preload to memory
    # Memory optimization properties
    optimize_for_m1: bool = True  # Apply M1-specific optimizations
    prefetch_factor: int = 2
    multiprocessing_context: str = "fork"  # "fork" or "spawn"


class AsyncVideoLoader:
    """
    Asynchronous video loader for improved throughput.
    Prefetches video frames in background threads to reduce IO bottlenecks.
    """
    
    def __init__(self, 
                 max_queue_size: int = 8,
                 num_workers: int = 2):
        """
        Initialize async loader.
        
        Args:
            max_queue_size: Maximum size of prefetch queue
            num_workers: Number of worker threads
        """
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = threading.Event()
    
    def start(self, 
              video_paths: List[str], 
              load_fn: Callable,
              load_args: Optional[List[Dict]] = None):
        """
        Start async loading of videos.
        
        Args:
            video_paths: List of video paths to load
            load_fn: Function to load video
            load_args: Optional arguments for load function per video
        """
        # Initialize stop event and workers
        self.stop_event.clear()
        self.workers = []
        
        # Set up task queue
        tasks = []
        for i, path in enumerate(video_paths):
            args = load_args[i] if load_args and i < len(load_args) else {}
            tasks.append((path, args))
        
        # Start worker threads
        for _ in range(min(self.num_workers, len(tasks))):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(tasks, load_fn),
                daemon=True
            )
            self.workers.append(worker)
            worker.start()
    
    def _worker_loop(self, tasks: List[Tuple[str, Dict]], load_fn: Callable):
        """Worker thread function."""
        # Distribute tasks among workers using thread ID
        worker_id = threading.get_ident()
        worker_idx = next((i for i, w in enumerate(self.workers) if w.ident == worker_id), 0)
        
        for i, (path, args) in enumerate(tasks):
            if self.stop_event.is_set():
                break
                
            # Simple task distribution - each worker takes every nth task
            if i % len(self.workers) != worker_idx:
                continue
                
            try:
                # Load video
                result = load_fn(path, **args)
                # Put result in queue
                self.queue.put((path, result))
            except Exception as e:
                logger.error(f"Error loading video {path}: {e}")
                # Put None to indicate error
                self.queue.put((path, None))
        
        # Put None to indicate end of tasks
        self.queue.put((None, None))
    
    def get(self) -> Tuple[str, Any]:
        """Get next loaded video."""
        return self.queue.get()
    
    def stop(self):
        """Stop all workers."""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers = []


class MemoryEfficientVideoDataset(Dataset):
    """
    Memory-efficient video dataset optimized for M1 Macs.
    Supports both PyTorchVideo and custom video loading.
    """
    
    def __init__(self, 
                 config: VideoDatasetConfig,
                 transform: Optional[Callable] = None):
        """
        Initialize video dataset.
        
        Args:
            config: Dataset configuration
            transform: Optional transform function
        """
        self.config = config
        self.transform = transform
        
        # Set up paths
        self.paths = self._get_video_paths()
        
        # Set up clip sampler
        self.clip_sampler = self._create_clip_sampler()
        
        # Initialize labeled video dataset
        self.dataset = LabeledVideoDataset(
            self.paths,
            self.clip_sampler,
            transform=self._transform,
            decode_audio=config.decode_audio,
            decoder=config.decoder,
            multithreaded_io=config.multithreaded_io
        )
        
        # Set up memory cache if preloading
        self.video_cache = {}
        if config.preload_to_memory:
            self._preload_videos()
    
    def _get_video_paths(self) -> List[Dict]:
        """Get list of video paths."""
        video_paths = []
        
        # Handle directory or file list
        if os.path.isdir(self.config.path):
            # Find all video files in directory
            for root, _, files in os.walk(self.config.path):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        path = os.path.join(root, file)
                        video_paths.append({
                            "path": path,
                            "label": os.path.basename(os.path.dirname(path))
                        })
        else:
            # Assume it's a text file with paths
            with open(self.config.path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        path = parts[0]
                        label = parts[1] if len(parts) > 1 else "unknown"
                        video_paths.append({
                            "path": path,
                            "label": label
                        })
        
        logger.info(f"Found {len(video_paths)} videos")
        return video_paths
    
    def _create_clip_sampler(self) -> ClipSampler:
        """Create clip sampler based on configuration."""
        from pytorchvideo.data.clip_sampling import (
            RandomClipSampler,
            UniformClipSampler,
            ConstantClipsPerVideoSampler
        )
        
        clip_duration = self.config.clip_duration
        clip_overlap = self.config.clip_overlap
        
        if self.config.clip_sampler == "random":
            return RandomClipSampler(clip_duration=clip_duration)
        elif self.config.clip_sampler == "uniform":
            return UniformClipSampler(
                clip_duration=clip_duration,
                clips_per_video=5,
                clip_stride=max(1, math.ceil(clip_duration * (1 - clip_overlap)))
            )
        elif self.config.clip_sampler == "constant_clips_per_video":
            return ConstantClipsPerVideoSampler(
                clip_duration=clip_duration,
                clips_per_video=5
            )
        else:
            raise ValueError(f"Unknown clip sampler: {self.config.clip_sampler}")
    
    def _transform(self, x: Dict) -> torch.Tensor:
        """Default transform function for video clips."""
        # Extract video tensor
        video = x["video"]
        
        # Apply custom transform if provided
        if self.transform is not None:
            video = self.transform(video)
        
        # Convert to half precision if configured
        if self.config.use_half_precision:
            video = video.to(torch.float16)
        
        return video
    
    def _preload_videos(self):
        """Preload videos to memory for faster access."""
        if not self.config.preload_to_memory:
            return
            
        logger.info(f"Preloading up to {self.config.preload_max_videos} videos to memory...")
        
        # Determine how many videos to preload
        num_to_preload = min(len(self.paths), self.config.preload_max_videos)
        paths_to_preload = self.paths[:num_to_preload]
        
        # Create async loader
        loader = AsyncVideoLoader(
            max_queue_size=min(16, num_to_preload),
            num_workers=4
        )
        
        # Define load function
        def load_video(path, **kwargs):
            try:
                container = av.open(path)
                frames = []
                for frame in container.decode(video=0):
                    # Convert frame to tensor
                    img = torch.from_numpy(
                        np.array(frame.to_image().convert('RGB'))
                    ).permute(2, 0, 1)
                    frames.append(img)
                return torch.stack(frames)
            except Exception as e:
                logger.error(f"Error preloading video {path}: {e}")
                return None
        
        # Start async loading
        loader.start(
            [p["path"] for p in paths_to_preload],
            load_fn=load_video
        )
        
        # Collect results
        for _ in range(num_to_preload):
            path, video = loader.get()
            if path is None:  # End of tasks
                break
            if video is not None:
                self.video_cache[path] = video
        
        # Stop loader
        loader.stop()
        
        logger.info(f"Preloaded {len(self.video_cache)} videos to memory")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item from dataset."""
        # Check if we have it cached
        if self.config.preload_to_memory:
            video_path = self.paths[idx]["path"]
            if video_path in self.video_cache:
                video = self.video_cache[video_path]
                # Apply transforms
                if self.transform is not None:
                    video = self.transform(video)
                return video
        
        # Otherwise get from pytorchvideo dataset
        return self.dataset[idx]


def create_video_transforms(mode: str = "train", 
                           size: int = 128, 
                           backend: str = "torchvision") -> Callable:
    """
    Create video transforms based on mode.
    
    Args:
        mode: "train" or "val"
        size: Size of video frames
        backend: "torchvision" or "cv2"
        
    Returns:
        Transform function
    """
    if backend == "torchvision":
        import torchvision.transforms as T
        
        if mode == "train":
            return T.Compose([
                T.RandomResizedCrop(size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(size),
                T.CenterCrop(size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif backend == "cv2":
        # Custom OpenCV-based transforms (more memory efficient)
        import cv2
        
        def transform(video: torch.Tensor) -> torch.Tensor:
            # Assuming video is [T, C, H, W] or [C, T, H, W]
            if video.shape[0] == 3:  # [C, T, H, W]
                video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
            
            T, C, H, W = video.shape
            resized = torch.zeros(T, C, size, size, dtype=video.dtype, device=video.device)
            
            # Process each frame
            for t in range(T):
                # Convert to numpy
                frame = video[t].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                
                # Apply transforms
                if mode == "train":
                    # Random crop
                    scale = np.random.uniform(0.8, 1.0)
                    crop_size = int(min(H, W) * scale)
                    start_h = np.random.randint(0, H - crop_size + 1)
                    start_w = np.random.randint(0, W - crop_size + 1)
                    frame = frame[start_h:start_h+crop_size, start_w:start_w+crop_size]
                    
                    # Random flip
                    if np.random.random() > 0.5:
                        frame = cv2.flip(frame, 1)  # horizontal flip
                    
                    # Color jitter
                    if np.random.random() > 0.5:
                        # Brightness
                        factor = np.random.uniform(0.7, 1.3)
                        frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
                        
                        # Contrast
                        factor = np.random.uniform(0.7, 1.3)
                        mean = np.mean(frame, axis=(0, 1), keepdims=True)
                        frame = (frame - mean) * factor + mean
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                else:
                    # Simple resize for validation
                    if H != size or W != size:
                        frame = cv2.resize(frame, (size, size))
                
                # Normalize
                frame = frame.astype(np.float32) / 255.0
                frame -= np.array([0.485, 0.456, 0.406])
                frame /= np.array([0.229, 0.224, 0.225])
                
                # Back to tensor
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
                resized[t] = frame_tensor
            
            return resized
        
        return transform
    else:
        raise ValueError(f"Unknown transform backend: {backend}")


def create_loader(dataset: Dataset, 
                 batch_size: int = 2, 
                 num_workers: int = 2,
                 optimize_for_m1: bool = True,
                 persistent_workers: bool = True,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2,
                 drop_last: bool = False) -> DataLoader:
    """
    Create optimized data loader for video dataset.
    
    Args:
        dataset: Video dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        optimize_for_m1: Whether to apply M1-specific optimizations
        persistent_workers: Whether to use persistent workers
        pin_memory: Whether to pin memory
        prefetch_factor: Prefetch factor
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader instance
    """
    # Get device manager
    device_manager = get_device_manager()
    
    # Set pin memory device if using M1
    pin_memory_device = None
    if optimize_for_m1 and device_manager.device_str == "mps":
        pin_memory_device = "mps"
    
    # Determine multiprocessing context - 'fork' is faster but less stable
    # Only use 'fork' on Unix systems with num_workers > 0
    mp_context = None
    if num_workers > 0:
        import multiprocessing
        if hasattr(multiprocessing, 'get_all_start_methods'):
            if 'fork' in multiprocessing.get_all_start_methods():
                mp_context = 'fork'
            else:
                mp_context = 'spawn'
    
    # Create DataLoader with optimized settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context,
        drop_last=drop_last
    )