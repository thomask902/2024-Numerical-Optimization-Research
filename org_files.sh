#!/bin/bash

# Define the source directory containing the files
src_dir="results/CIFAR10/GNOM"
# Define the destination directory where new directories will be created
dest_dir="tensorboard/CIFAR10/GNOM"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through each file in the source directory
for file in "$src_dir"/*; do
    # Get the base name of the file (without the directory part)
    base_name=$(basename "$file")
    # Create a new directory in the destination directory with the same name as the file (without extension)
    new_dir="$dest_dir/${base_name%.*}"
    mkdir -p "$new_dir"
    # Move the file into the new directory
    mv "$file" "$new_dir"
done