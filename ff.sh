# Create directory for processed videos
mkdir -p data/hmdb51/processed

# Process videos - resize to 128x128 and convert to H.264
# Install ffmpeg if you don't have it: brew install ffmpeg
for category in data/hmdb51/videos/*; do
    category_name=$(basename "$category")
    mkdir -p data/hmdb51/processed/$category_name
    
    for video in "$category"/*.avi; do
        if [ -f "$video" ]; then
            filename=$(basename "$video")
            output_name="${filename%.*}.mp4"
            ffmpeg -i "$video" -vf "scale=128:128" -c:v h264 -crf 23 -threads 2 \
                   data/hmdb51/processed/$category_name/$output_name
        fi
    done
done