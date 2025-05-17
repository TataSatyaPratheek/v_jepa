import os
import logging
import torch
from typing import List, Tuple, Dict, Optional, Callable

# Attempt to import PyTorchVideo components
try:
    from pytorchvideo.data import LabeledVideoDataset
    from pytorchvideo.data.clip_sampling import RandomClipSampler, ClipSampler # Ensure ClipSampler is imported if used as a type hint
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    print("PyTorchVideo is not available. Please install it to run this test.")
    exit()

# Configure logging
logger = logging.getLogger("standalone_pv_test")
logger.setLevel(logging.INFO)
if not logger.hasHandlers(): # Avoid adding multiple handlers if run multiple times
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_first_n_videos_from_dir(directory_path: str, n: int = 5) -> List[Tuple[str, Dict]]:
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
    labeled_video_paths: List[Tuple[str, Dict]] = []
    
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return labeled_video_paths

    label = os.path.basename(directory_path)

    count = 0
    try:
        for item in sorted(os.listdir(directory_path)):
            if count >= n:
                break
            if item.lower().endswith(video_extensions):
                full_path = os.path.join(directory_path, item)
                labeled_video_paths.append((full_path, {"label": label, "video_path": full_path})) # Add any other meta if needed
                count += 1
    except Exception as e_list:
        logger.error(f"Error listing directory {directory_path}: {e_list}")
        return [] # Return empty if listing fails
            
    if count == 0:
        logger.warning(f"No video files found in {directory_path} with extensions {video_extensions}")
        
    return labeled_video_paths

def simple_pv_transform(item_dict: Dict) -> Dict:
    """
    A basic transform for LabeledVideoDataset.
    It expects a dictionary and should return a dictionary.
    The 'video' tensor is typically (C, T, H, W).
    """
    # logger.info(f"Transform received item keys: {item_dict.keys()}")
    # if 'video' in item_dict:
    #     logger.info(f"  Video tensor shape in transform: {item_dict['video'].shape}")
    # Add any simple augmentations or just pass through
    return item_dict


def run_standalone_test():
    if not PYTORCHVIDEO_AVAILABLE:
        return

    video_directory = "/Users/vi/Documents/not work/v_jepa/data/hmdb51/processed/brush_hair"
    
    video_path_tuples: List[Tuple[str, Dict]] = get_first_n_videos_from_dir(video_directory, n=5)

    if not video_path_tuples:
        logger.error(f"No video paths obtained from {video_directory}. Exiting test.")
        return

    logger.info(f"Using {len(video_path_tuples)} video path(s) for the test:")
    for path, meta_dict in video_path_tuples:
        logger.info(f"- Path: {path}, Label: {meta_dict.get('label')}")

    clip_sampler_instance: ClipSampler = RandomClipSampler(clip_duration=2.0) # 2-second clips
    decoder_type: str = "pyav" # Common decoder

    logger.info(f"Initializing LabeledVideoDataset...")
    
    dataset_instance: Optional[LabeledVideoDataset] = None
    try:
        dataset_instance = LabeledVideoDataset(
            labeled_video_paths=video_path_tuples,
            clip_sampler=clip_sampler_instance,
            transform=simple_pv_transform, 
            decode_audio=False,
            decoder=decoder_type
        )
        logger.info("LabeledVideoDataset initialized successfully.")

        num_samples_to_try = min(2, len(video_path_tuples))
        if num_samples_to_try > 0 and dataset_instance is not None:
            logger.info(f"Attempting to fetch {num_samples_to_try} sample(s)...")
            dataset_iterator = iter(dataset_instance)
            for i in range(num_samples_to_try):
                try:
                    sample = next(dataset_iterator)
                    logger.info(f"Fetched sample {i+1}: Keys: {sample.keys()}")
                    if 'video' in sample and isinstance(sample['video'], torch.Tensor):
                        logger.info(f"  Video tensor shape: {sample['video'].shape}") # Expect (C,T,H,W)
                    else:
                        logger.warning(f"  Sample {i+1} does not contain a 'video' tensor or it's not a tensor.")
                except StopIteration:
                    logger.error("StopIteration: Dataset exhausted prematurely.")
                    break
                except Exception as e_fetch_item:
                    logger.error(f"Error fetching sample {i+1}: {e_fetch_item}", exc_info=True)
                    break 
        elif dataset_instance is None:
             logger.error("Dataset instance is None after initialization attempt.")


    except Exception as e_initialization:
        logger.error(f"Critical error initializing LabeledVideoDataset: {e_initialization}", exc_info=True)

if __name__ == "__main__":
    run_standalone_test()
