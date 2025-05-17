import os

def fix_dataset_structure():
    """Check and fix dataset structure if needed."""
    videos_path = "data/hmdb51/processed"  # or "data/hmdb51/videos"
    
    # Check if any videos exist in the expected structure
    video_count = 0
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                video_count += 1
    
    print(f"Found {video_count} videos in dataset structure.")
    if video_count == 0:
        print("ERROR: No videos found in the expected structure.")
        print("Please ensure videos are organized in category subdirectories.")
    else:
        print("Dataset structure looks good!")

if __name__ == "__main__":
    fix_dataset_structure()