# Create a directory for the extracted videos
mkdir -p data/hmdb51/videos

# Move to the directory containing the RAR files
cd data/hmdb51/hmdb51_org

# Extract all RAR files to the videos directory
for rarfile in *.rar; do
    # Get the category name (remove .rar extension)
    category=${rarfile%.rar}
    
    # Create category directory
    mkdir -p ../videos/
    
    # Extract RAR file to category directory
    unar $rarfile -o ../videos/
    
    echo "Extracted $rarfile to ../videos/$category/"
done

# Return to original directory
cd ../../..
