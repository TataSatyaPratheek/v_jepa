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
import queue # Keep existing queue import
import time # Add time import
from queue import Queue, Empty # Add specific imports from queue

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
    fast_start: bool = True             # Skip preloading for faster startup
    max_preload_videos: int = 100       # Limit videos to preload (Note: similar to preload_max_videos)
    max_preload_time: int = 30          # Max seconds to spend preloading
    verify_structure: bool = True       # Verify directory structure
    max_dataset_size: Optional[int] = None  # Limit total dataset size for testing
    lazy_loading: bool = True           # Load videos on-demand instead of preloading
    log_progress: bool = True           # Show detailed loading progress
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
        
        # Add better logging and timing
        start_time = time.time()
        logger.info(f"Initializing VideoDataset with path: {config.path}")
        
        # Verify structure if requested
        if getattr(config, 'verify_structure', True):
            self._verify_structure()
        
        # Set up paths
        logger.info("Finding video paths...")
        self.paths = self._get_video_paths()
        
        # Limit dataset size if configured
        if getattr(config, 'max_dataset_size', None) is not None:
            orig_size = len(self.paths)
            self.paths = self.paths[:config.max_dataset_size]
            logger.info(f"Limited dataset from {orig_size} to {len(self.paths)} videos")
        
        logger.info(f"Found {len(self.paths)} videos in {config.path}")
        
        # Set up clip sampler
        self.clip_sampler = self._create_clip_sampler()
        
        # Initialize other components with better defaults
        self.dataset = None
        if PYTORCHVIDEO_AVAILABLE and not getattr(config, 'lazy_loading', True):
            try:
                # Only try to create PyTorchVideo dataset if not using lazy loading
                logger.info("Attempting to initialize PyTorchVideo dataset...")
                if PYTORCHVIDEO_AVAILABLE: # Redundant check, but keeps structure from original
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
                        except Exception as e_iter:
                            logger.warning(f"Could not fetch a test item from LabeledVideoDataset during initial test: {e_iter}. "
                                           f"PytorchVideo path might have issues for some items. Will rely on fallback if needed.")
            except Exception as e_init:
                logger.error(f"Failed to initialize PyTorchVideo dataset: {e_init}")
                self.dataset = None # Ensure self.dataset is None if initialization itself fails
        else:
            if not PYTORCHVIDEO_AVAILABLE:
                logger.info("PyTorchVideo is not available. Using fallback _load_video_directly for all items.")
            elif getattr(config, 'lazy_loading', True):
                logger.info("Lazy loading enabled. PyTorchVideo dataset will be initialized on demand if needed.")

        # Initialize for lazy loading or if PyTorchVideo init failed
        self.dataset_items = []
        self.video_cache = {}
        
        # Only preload videos or dataset items if not in fast start mode
        fast_start = getattr(config, 'fast_start', True)
        if not fast_start:
            if config.preload_to_memory:
                self._preload_videos() # This already calls _preload_pytorchvideo_items if needed
            elif self.dataset is not None: # If not preloading videos but PTV dataset exists
                self._preload_pytorchvideo_items() 
        else:
            logger.info("Fast start mode: skipping preloading")
        
        init_time = time.time() - start_time
        logger.info(f"Dataset initialization completed in {init_time:.2f} seconds")
    
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

    def _verify_structure(self):
        """Verify dataset structure and count videos."""
        logger.info(f"Verifying dataset structure at {self.config.path}")
        
        if not os.path.exists(self.config.path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.config.path}")
        
        if not os.path.isdir(self.config.path):
            # If it's a file, assume it's a list of paths, which is handled by _get_video_paths
            if os.path.isfile(self.config.path):
                logger.info(f"Dataset path {self.config.path} is a file, assuming list of video paths.")
                return # No further directory structure to verify
            raise NotADirectoryError(f"Dataset path is not a directory or a valid file: {self.config.path}")
        
        # Count action classes and videos if it's a directory
        class_dirs = [d for d in os.listdir(self.config.path) 
                      if os.path.isdir(os.path.join(self.config.path, d))]
        
        video_count = sum(len([f for f in os.listdir(os.path.join(self.config.path, cd)) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]) for cd in class_dirs)
        logger.info(f"Found {len(class_dirs)} action classes with {video_count} videos in directory structure.")

    
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
        video_path_str = self.paths[idx][0]
        
        # Clear memory before loading
        empty_cache()
        
        try:
            logger.debug(f"Loading video {idx}: {video_path_str}")
            start_time = time.time()
            
            container = av.open(video_path_str)
            stream = container.streams.video[0]
            
            video_fps = float(stream.average_rate if stream.average_rate else self.config.frame_rate)
            
            frames = []
            # Calculate frames to sample
            num_frames_to_sample = int(self.config.clip_duration * self.config.frame_rate)
            
            # Determine frame interval
            frame_interval = max(1.0, video_fps / self.config.frame_rate)
            
            output_frames_collected = 0
            current_target_frame_idx = 0.0
            
            # Use a counter to avoid infinite loops
            max_frames_to_process = min(500, int(video_fps * 10))  # Limit to 10 seconds max
            frame_count = 0
            
            for frame in container.decode(video=0):
                frame_count += 1
                if frame_count > max_frames_to_process:
                    logger.warning(f"Hit frame limit ({max_frames_to_process}) when processing {video_path_str}")
                    break
                    
                if output_frames_collected >= num_frames_to_sample:
                    break
                    
                if frame_count >= int(round(current_target_frame_idx)):
                    # Convert frame to tensor
                    img = torch.from_numpy(np.array(frame.to_image().convert('RGB'))).permute(2, 0, 1)
                    frames.append(img)
                    output_frames_collected += 1
                    current_target_frame_idx += frame_interval
            
            container.close()
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path_str}, using dummy tensor")
                # Return dummy tensor with expected dimensions
                return torch.zeros((3, num_frames_to_sample, 128, 128), 
                                   dtype=torch.float16 if self.config.use_half_precision else torch.float32)
            
            # Stack frames
            video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
            
            # Apply transforms if needed
            if self.transform:
                video_tensor = self.transform(video_tensor)
            
            # Ensure output shape is [C, T, H, W]
            if video_tensor.ndim == 4 and video_tensor.shape[1] == 3:
                video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
            
            if self.config.use_half_precision:
                video_tensor = video_tensor.to(torch.float16)
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded video in {load_time:.3f}s, shape: {video_tensor.shape}")
            
            return video_tensor
        except Exception as e:
            logger.error(f"Error loading video {video_path_str}: {e}")
            # Return dummy tensor with expected dimensions
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