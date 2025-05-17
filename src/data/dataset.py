import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import os
import math
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
import logging
import av
import threading
import queue

# Try to import PyTorchVideo classes
try:
    from pytorchvideo.data import LabeledVideoDataset
    from pytorchvideo.data.clip_sampling import ClipSampler
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False

from ..utils.device import get_device_manager
from ..utils.memory import empty_cache

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VideoDatasetConfig:
    """Configuration for video dataset."""
    path: str = "data/videos"  # Make path optional with a default value
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
        
        # Initialize self.dataset to None
        self.dataset = None 
        if PYTORCHVIDEO_AVAILABLE:
            try:
                self.dataset = LabeledVideoDataset(
                    labeled_video_paths=self.paths,
                    clip_sampler=self.clip_sampler,
                    transform=self._transform, # This is your internal wrapper
                    decode_audio=self.config.decode_audio,
                    decoder=self.config.decoder
                )
                
                # Correctly test the IterableDataset functionality
                if self.dataset is not None and len(self.paths) > 0:
                    logger.info("Attempting to fetch a test item from LabeledVideoDataset...")
                    try:
                        # For IterableDataset, get an iterator and try to fetch one item
                        _ = next(iter(self.dataset)) # This correctly tests iteration
                        logger.info("Successfully fetched a test item from LabeledVideoDataset.")
                    except StopIteration:
                        logger.warning("LabeledVideoDataset is empty or exhausted on the first test item.")
                        # This might be okay if the dataset can be empty, or an issue if not.
                    except Exception as e_iter:
                        logger.warning(f"Could not fetch a test item from LabeledVideoDataset during initial test: {e_iter}. "
                                       f"PytorchVideo path might have issues for some items. Will rely on fallback if needed.")
                        # Depending on how critical this test is, you might decide to set self.dataset = None here
                        # to force fallback for all items if this initial check fails.
                        # For now, we'll allow it to proceed and let __getitem__ handle individual item errors.
            except Exception as e_init:
                logger.error(f"Error during LabeledVideoDataset initialization: {e_init}. "
                               f"Will use fallback _load_video_directly for all items.")
                self.dataset = None # Ensure self.dataset is None if initialization itself fails
        else:
            logger.info("PyTorchVideo is not available. Using fallback _load_video_directly for all items.")
            # self.dataset is already None, which is correct

        # Pre-load all items from PyTorchVideo dataset for random access
        self.dataset_items = []
                
        # Set up memory cache if preloading
        self.video_cache = {}
        if config.preload_to_memory:
            self._preload_videos()
        
        # Pre-load PyTorchVideo items for faster random access if dataset is available
        self._preload_pytorchvideo_items()
    
    def _get_video_paths(self) -> List[Tuple[str, Dict]]:
        """Get list of video paths."""
        video_path_tuples = []
        
        # Handle directory or file list
        if os.path.isdir(self.config.path):
            # Find all video files in directory
            for root, _, files in os.walk(self.config.path):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        path = os.path.join(root, file)
                        label = os.path.basename(os.path.dirname(path))
                        # Include video_path in metadata for compatibility with PyTorchVideo
                        video_path_tuples.append(
                            (path, {"label": label, "video_path": path})
                        )
        else:
            # Assume it's a text file with paths
            with open(self.config.path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        path = parts[0]
                        label = parts[1] if len(parts) > 1 else "unknown"
                        video_path_tuples.append(
                            (path, {"label": label})
                        )
        
        logger.info(f"Found {len(video_path_tuples)} videos")
        return video_path_tuples
    
    def _preload_pytorchvideo_items(self):
        """Pre-load all items from PyTorchVideo dataset for random access."""
        if self.dataset is not None:
            try:
                logger.info("Pre-loading items from PyTorchVideo dataset (optimized for M1)...")
                # Process items in smaller batches to avoid memory pressure
                self.dataset_items = []
                batch_size = 10  # Process 10 items at a time
                dataset_iter = iter(self.dataset)
                try:
                    while True:
                        self.dataset_items.extend([next(dataset_iter) for _ in range(batch_size)])
                        empty_cache()  # Clear cache after each batch
                except StopIteration:
                    logger.info(f"Pre-loaded {len(self.dataset_items)} items from PyTorchVideo dataset")
            except Exception as e:
                logger.error(f"Error pre-loading items from PyTorchVideo dataset: {e}. Will use fallback loading.")
                self.dataset_items = []
    
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

        # Force garbage collection before preloading PyTorchVideo items
        gc.collect()
        empty_cache()
        self._preload_pytorchvideo_items()

    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item from dataset."""
        # Check if we have it cached
        if self.config.preload_to_memory:
            # self.paths is now a list of tuples (path_str, info_dict)
            video_path_str = self.paths[idx][0]
            if video_path_str in self.video_cache:
                video = self.video_cache[video_path_str]
                # Apply transforms
                if self.transform is not None:
                    video = self.transform(video)
                return video
        
        # Try to use pre-loaded items if available
        if len(self.dataset_items) > idx:
            # Get item from preloaded cache
            item = self.dataset_items[idx]
            
            # Apply transforms if needed
            if self.transform is not None and 'video' in item:
                item['video'] = self.transform(item['video'])
            
            return item
        # Otherwise try to get from pytorchvideo dataset directly (this path may fail)
        elif self.dataset is not None:
            try:
                # This might be slow if LabeledVideoDataset is an IterableDataset and not a MapDataset
                # Try direct indexing (may not work with IterableDataset)
                item = self.dataset[idx]
                return item
            except Exception as e:
                logger.error(f"Error accessing dataset at index {idx}: {e}")
                # Fall back to direct loading - clear memory first
                empty_cache()
                return self._load_video_directly(idx)
        else:
            # If dataset initialization failed, load video directly
            return self._load_video_directly(idx)
            
    def _load_video_directly(self, idx: int) -> torch.Tensor:
        """Fallback to load video directly using PyAV if PyTorchVideo fails or is unavailable."""
        video_path_str = self.paths[idx][0] # self.paths stores tuples of (path_str, info_dict)
        # Clear memory before loading
        empty_cache()
        logger.info(f"Directly loading video from {video_path_str} using PyAV fallback.")

        try:
            container = av.open(video_path_str)
            stream = container.streams.video[0]
            
            video_fps = float(stream.average_rate if stream.average_rate else self.config.frame_rate)
            
            frames = []
            # Calculate the number of frames to sample for the desired clip duration and frame rate
            num_frames_to_sample = int(self.config.clip_duration * self.config.frame_rate)
            logger.info(f"Fallback load: Frames to sample: {num_frames_to_sample}")
            
            # This is a simple sequential frame sampling strategy for the fallback.
            # It aims to extract 'num_frames_to_sample' at roughly the target 'self.config.frame_rate'.
            
            # Determine the interval to pick frames from the original video stream
            # to match the target self.config.frame_rate.
            frame_interval = video_fps / self.config.frame_rate
            if frame_interval <= 0: # Avoid division by zero or negative interval
                frame_interval = 1.0 

            output_frames_collected = 0
            # Represents the "ideal" floating point index in the original video stream
            current_target_frame_original_idx = 0.0 

            for frame_idx, frame in enumerate(container.decode(video=0)):
                if output_frames_collected >= num_frames_to_sample:
                    break
                
                # Select the frame if its actual index is at or after the current target index
                if frame_idx >= int(round(current_target_frame_original_idx)): # Round to nearest whole frame index
                    # Convert PyAV frame to RGB Pillow Image, then to NumPy array, then to PyTorch tensor
                    img = torch.from_numpy(np.array(frame.to_image().convert('RGB'))).permute(2, 0, 1)  # Shape: (C, H, W)
                    # Using print as per user's debugging example for these specific lines
                    print(f"Fallback load: Shape of img after frame conversion: {img.shape}") 
                    frames.append(img)
                    output_frames_collected += 1
                    current_target_frame_original_idx += frame_interval
            
            container.close()

            if not frames:
                logger.warning(f"Fallback: No frames extracted from {video_path_str}. Returning zeros.")
                # Use a consistent placeholder shape (C, T, H, W)
                return torch.zeros((3, num_frames_to_sample, 128, 128), 
                                   dtype=torch.float16 if self.config.use_half_precision else torch.float32)

            # Stack frames to form a video tensor (T, C, H, W)
            video_tensor = torch.stack(frames, dim=0)
            print(f"Fallback load: Shape of video_tensor before transform: {video_tensor.shape}")

            if self.transform:
                video_tensor = self.transform(video_tensor)
                print(f"Fallback load: Shape of video_tensor after transform: {video_tensor.shape}")
            
            # Ensure output tensor is in (C, T, H, W) format.
            # Assumes C=3. If video_tensor is (T, 3, H, W), permute it.
            if video_tensor.ndim == 4:
                if video_tensor.shape[1] == 3: # Likely (T, 3, H, W)
                    logger.info(f"Fallback load: Permuting video_tensor from {video_tensor.shape} to (C,T,H,W).")
                    video_tensor = video_tensor.permute(1, 0, 2, 3) # New shape: (3, T, H, W)
                elif video_tensor.shape[0] == 3: # Likely already (3, T, H, W)
                    logger.info(f"Fallback load: video_tensor shape {video_tensor.shape} is already (C,T,H,W)-like.")
                else:
                    logger.warning(f"Fallback load: video_tensor has shape {video_tensor.shape} after transform. Cannot determine C and T for standard (C,T,H,W) permutation. Expected 3 channels.")
            else:
                logger.warning(f"Fallback load: video_tensor has unexpected ndim {video_tensor.ndim} after transform. Expected 4 dimensions.")
            
            if self.config.use_half_precision:
                video_tensor = video_tensor.to(torch.float16)
            
            print(f"Fallback load: Final shape of video_tensor being returned: {video_tensor.shape}")
            return video_tensor

        except Exception as e:
            logger.error(f"Error in fallback video loading for {video_path_str}: {e}")
            num_dummy_frames = int(self.config.frame_rate * self.config.clip_duration)
            return torch.zeros((3, num_dummy_frames, 128, 128), 
                               dtype=torch.float16 if self.config.use_half_precision else torch.float32)


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
    
    actual_num_workers = num_workers # type: ignore
    actual_pin_memory = pin_memory
    actual_pin_memory_device = None  # Default to None
    actual_persistent_workers = persistent_workers
    actual_prefetch_factor = prefetch_factor
    # For M1 with limited memory, further restrict the settings
    if optimize_for_m1:
        actual_prefetch_factor = min(prefetch_factor, 1)  # Limit prefetch factor to reduce memory pressure
        drop_last = True  # Drop last incomplete batch for consistent memory usage
    mp_context = None
    
    if optimize_for_m1 and device_manager.device_str == "mps":
        # This log message was missing in your output, ensure logger for 'src.data.dataset' is configured if you want to see it.
        logger.info("Applying M1-specific DataLoader optimizations: num_workers=0, pin_memory=False, pin_memory_device=''")
        actual_num_workers = 0
        actual_pin_memory = False  # Disable pin_memory with num_workers=0 on MPS
        actual_pin_memory_device = "" # Ensure it's a string
    elif device_manager.device_str == "mps" and actual_pin_memory:
        # If not M1 optimized path but still on MPS and pinning memory
        actual_pin_memory_device = "mps" # type: ignore

    if actual_num_workers > 0:
        # persistent_workers and prefetch_factor are relevant if num_workers > 0
        actual_persistent_workers = persistent_workers
        actual_prefetch_factor = prefetch_factor
        import multiprocessing
        if hasattr(multiprocessing, 'get_all_start_methods'):
            # Prefer 'fork' on Unix-like systems (not Windows) for speed, if available
            if 'fork' in multiprocessing.get_all_start_methods() and os.name != 'nt':
                mp_context = 'fork'
            elif 'spawn' in multiprocessing.get_all_start_methods(): # 'spawn' is safer and cross-platform
                mp_context = 'spawn'
            # If neither, PyTorch will use its default.
    else:  # actual_num_workers is 0
        actual_persistent_workers = False # persistent_workers requires num_workers > 0
        actual_prefetch_factor = None     # prefetch_factor is not used if num_workers is 0
        mp_context = None

    # Ensure pin_memory_device is a string if it's still None (e.g. CUDA path with pin_memory=True)
    if actual_pin_memory_device is None:
        actual_pin_memory_device = ""
    
    # Create DataLoader with optimized settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        persistent_workers=actual_persistent_workers,
        pin_memory=actual_pin_memory,
        pin_memory_device=actual_pin_memory_device, # Will be None if actual_pin_memory is False
        prefetch_factor=actual_prefetch_factor,
        multiprocessing_context=mp_context,
        drop_last=drop_last
    )