#!/bin/bash
# Helper script to create symlinks from top-level hdmap/ to ruqi_render/hdmap/
# This ensures the generate_video_single_view.py script can find the video files
#
# Usage: ./create_hdmap_symlinks.sh /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669

SCENE_DIR="${1}"

if [ -z "$SCENE_DIR" ]; then
    echo "Usage: $0 <scene_directory>"
    echo "Example: $0 /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669"
    exit 1
fi

# Check if scene directory exists
if [ ! -d "$SCENE_DIR" ]; then
    echo "Error: Scene directory $SCENE_DIR does not exist"
    exit 1
fi

# Define paths
TOP_LEVEL_HDMAP="$SCENE_DIR/hdmap"
RENDER_HDMAP="$SCENE_DIR/ruqi_render/hdmap"

# Check if directories exist
if [ ! -d "$TOP_LEVEL_HDMAP" ]; then
    echo "Error: Top-level hdmap directory does not exist: $TOP_LEVEL_HDMAP"
    exit 1
fi

if [ ! -d "$RENDER_HDMAP" ]; then
    echo "Error: Render hdmap directory does not exist: $RENDER_HDMAP"
    exit 1
fi

# Camera views to process
CAMERA_VIEWS=(
    "pinhole_front_wide"
    "pinhole_left_front"
    "pinhole_right_front"
    "pinhole_back"
    "pinhole_left_back"
    "pinhole_right_back"
)

# Get sample name from directory name (e.g., ABC1_1735885669)
SAMPLE_NAME=$(basename "$SCENE_DIR")

echo "Creating symlinks in $TOP_LEVEL_HDMAP ..."
echo "Sample name: $SAMPLE_NAME"
echo ""

for view in "${CAMERA_VIEWS[@]}"; do
    VIEW_DIR="$TOP_LEVEL_HDMAP/$view"
    SOURCE_DIR="$RENDER_HDMAP/$view"

    # Check if source directory exists
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Warning: Source directory $SOURCE_DIR does not exist, skipping $view"
        continue
    fi

    # Create view directory if it doesn't exist
    if [ ! -d "$VIEW_DIR" ]; then
        mkdir -p "$VIEW_DIR"
        echo "Created directory: $VIEW_DIR"
    fi

    # Create the main symlink: {sample_name}_0.mp4
    TARGET_SYMLINK="$VIEW_DIR/${SAMPLE_NAME}_0.mp4"
    TARGET_FILE="$SOURCE_DIR/${SAMPLE_NAME}_0.mp4"

    if [ -f "$TARGET_FILE" ]; then
        if [ -e "$TARGET_SYMLINK" ]; then
            # Remove existing symlink
            rm -f "$TARGET_SYMLINK"
        fi
        # Create absolute symlink to avoid resolution issues through ftheta symlinks
        ln -s "$TARGET_FILE" "$TARGET_SYMLINK"
        echo "  Created: $view/${SAMPLE_NAME}_0.mp4 -> $TARGET_FILE"
    else
        echo "  Warning: Target file does not exist: $TARGET_FILE"
    fi

    # Create multi_view_spec symlink (for multi-view generation)
    MULTI_VIEW_SYMLINK="$VIEW_DIR/multi_view_spec_0.mp4"
    if [ -e "$MULTI_VIEW_SYMLINK" ]; then
        rm -f "$MULTI_VIEW_SYMLINK"
    fi
    ln -s "$TARGET_FILE" "$MULTI_VIEW_SYMLINK"
    echo "  Created: $view/multi_view_spec_0.mp4 -> $TARGET_FILE"

    # Create single_view_spec symlink (for single-view generation, typically only front_wide)
    SINGLE_VIEW_SYMLINK="$VIEW_DIR/single_view_spec_0.mp4"
    if [ -e "$SINGLE_VIEW_SYMLINK" ]; then
        rm -f "$SINGLE_VIEW_SYMLINK"
    fi
    ln -s "$TARGET_FILE" "$SINGLE_VIEW_SYMLINK"
    echo "  Created: $view/single_view_spec_0.mp4 -> $TARGET_FILE"

    echo ""
done

echo "Done! Symlinks created in $TOP_LEVEL_HDMAP"
echo ""
echo "You can verify with: tree -L 2 $TOP_LEVEL_HDMAP"
