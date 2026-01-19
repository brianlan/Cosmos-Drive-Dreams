#!/bin/bash
# Helper script to prepare HD map directory for Cosmos video generation
# Usage: ./prepare_hdmap_for_generation.sh /path/to/hdmap_directory

HDMAP_DIR="${1:-.}"

cd "$HDMAP_DIR" || exit 1

# Mapping of actual directory names to expected names
declare -A VIEW_MAP=(
    ["pinhole_front_wide"]="ftheta_camera_front_wide_120fov"
    ["pinhole_left_front"]="ftheta_camera_cross_left_120fov"
    ["pinhole_right_front"]="ftheta_camera_cross_right_120fov"
    ["pinhole_back"]="ftheta_camera_rear_tele_30fov"
    ["pinhole_left_back"]="ftheta_camera_rear_left_70fov"
    ["pinhole_right_back"]="ftheta_camera_rear_right_70fov"
)

echo "Creating symlinks in $HDMAP_DIR..."

for actual in "${!VIEW_MAP[@]}"; do
    expected="${VIEW_MAP[$actual]}"
    if [ -d "$actual" ] && [ ! -e "$expected" ]; then
        ln -s "$actual" "$expected"
        echo "  Created: $expected -> $actual"
    elif [ -e "$expected" ]; then
        echo "  Skipped: $expected already exists"
    else
        echo "  Warning: $actual not found"
    fi
done

echo "Done!"
