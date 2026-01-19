# How to Convert Ruqi Data to RDS-HQ and Render HD Map Videos

This guide summarizes the commands and options we used to turn the Ruqi prefusion data into Cosmos-friendly RDS-HQ layout and HD map movies.

## 1. Convert Ruqi Prefusion Data

The converter lives in `scripts/convert_ruqi_to_rds.py`. It reads a `prefusion_ruqi*.pkl` file and emits the RDS-HQ folder structure expected by Cosmos’ rendering pipeline.

```bash
python scripts/convert_ruqi_to_rds.py \
  --src-pkl /ssd5/datasets/ruqi/scenes/ABC1_1735885669/prefusion_ruqi.pkl \
  --scene-id ABC1_1735885669 \
  --output-root /ssd5/datasets/ruqi/scenes-cosmos \
  --clip-id ABC1_1735885669 \
  --data-root /ssd5/datasets/ruqi/scenes/ABC1_1735885669 \
  --keyframe-stride 5
```

**Flag details**

* `--src-pkl`: the full 10 Hz prefusion pickle (use `_keyframe` only if you truly want 2 Hz).  
* `--scene-id`: key inside that pickle; defaults to the first key if omitted.  
* `--output-root`: destination directory; the script creates `<output-root>/<clip-id>/...`.  
* `--clip-id`: name to use for all `.tar` files (usually same as scene ID).  
* `--data-root`: base path to resolve image/LiDAR files referenced in the pickle. If you pass the scene folder itself (e.g., `/ssd5/.../scenes/ABC1_1735885669`), the converter automatically adjusts one level up so the relative paths resolve.  
* `--keyframe-stride`: use only every N‑th frame for “trusted” annotations (3D boxes, LiDAR). With Ruqi, 5 matches the 2 Hz labeled keyframes.
* `--interpolate-ego-pose / --no-interpolate-ego-pose`: interpolate ego poses to 30 FPS with SLERP (default) or fall back to simple duplication.

The converter also handles Ruqi-specific quirks:

* Sensor extrinsics are stored as **sensor → ego**; no inversion needed.  
* Intrinsics in the pickle are often `(cx, cy, fx, fy)`; the script detects and swaps them into `(fx, fy, cx, cy)` automatically.  
* Each camera’s true resolution is derived by loading the first RGB file, so pinhole intrinsics carry the correct width/height per sensor.

## 2. Render HD Map Videos

Rendering is handled by `cosmos-drive-dreams-toolkits/render_from_rds_hq.py`. It expects the RDS-HQ layout created above.

```bash
cd cosmos-drive-dreams-toolkits
USE_RAY=False PYTHONPATH=. python render_from_rds_hq.py \
  -i /ssd5/datasets/ruqi/scenes-cosmos/ABC1_1735885669 \
  -o /ssd5/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render \
  -cj ABC1_1735885669 \
  -d ruqi \
  -c pinhole \
  -s lidar \
  -s world_scenario
```

**Flag details**

* `-i / --input_root`: the clip's RDS-HQ folder (the converter's `output-root/clip-id`).
* `-o / --output_root`: where to store the rendered videos.
* `-cj`: clip list. You can pass a single ID (as above), a JSON file containing a list, or omit to render all clips.
* `-d`: dataset configuration. Use `ruqi` to pick up the Ruqi camera set and FPS.
* `-c`: camera model (`pinhole` for Ruqi).
* `-s lidar`, `-s world_scenario`: optional skips; remove them if you want those modalities too.

**Output structure**: Creates `ruqi_render/hdmap/pinhole_<camera>/` directories with rendered `.mp4` files.

The renderer automatically upsamples the 10 Hz pose stream to 30 FPS (Cosmos' assumption) and uses the per-camera intrinsics derived during conversion.

### Chunk Length / Frame Count

Dataset configs can cap each rendered clip via `TARGET_CHUNK_FRAME` and `MAX_CHUNK` (see `cosmos-drive-dreams-toolkits/config/dataset_ruqi.json`).

* `TARGET_CHUNK_FRAME` – maximum frames per chunk.
* `MAX_CHUNK` – how many chunks to render (`-1` means "no limit").

Set these high enough (e.g., `TARGET_CHUNK_FRAME: 600`, `MAX_CHUNK: -1`) if you want the full 20 s Ruqi clips instead of the default 121-frame segments.

## 2.5. Prepare HD Map Directory for Video Generation

**Important**: There's a naming mismatch between what the rendering produces and what the multi-view generation script expects.

### The Problem

| What Rendering Produces | What Multi-View Script Expects |
|-------------------------|--------------------------------|
| `pinhole_front_wide` | `ftheta_camera_front_wide_120fov` |
| `pinhole_left_front` | `ftheta_camera_cross_left_120fov` |
| `pinhole_right_front` | `ftheta_camera_cross_right_120fov` |
| `pinhole_back` | `ftheta_camera_rear_tele_30fov` |
| `pinhole_left_back` | `ftheta_camera_rear_left_70fov` |
| `pinhole_right_back` | `ftheta_camera_rear_right_70fov` |

### Solution: Create Symlinks

For each new rendered HD map directory, create symlinks to match the expected naming:

```bash
# Navigate to your hdmap directory
cd /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks

# Create symlinks for expected view names
ln -s pinhole_front_wide ftheta_camera_front_wide_120fov
ln -s pinhole_left_front ftheta_camera_cross_left_120fov
ln -s pinhole_right_front ftheta_camera_cross_right_120fov
ln -s pinhole_back ftheta_camera_rear_tele_30fov
ln -s pinhole_left_back ftheta_camera_rear_left_70fov
ln -s pinhole_right_back ftheta_camera_rear_right_70fov
```

### Helper Script

Save this as `scripts/prepare_hdmap_for_generation.sh`:

```bash
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
```

Then run:
```bash
chmod +x scripts/prepare_hdmap_for_generation.sh
./scripts/prepare_hdmap_for_generation.sh /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks
```

### Update Spec Files

Don't forget to update your `*_spec.json` files to point to the correct HD map paths.

#### Where Are Spec Files Located?

```
/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/
├── single_view_spec.json    # For single-view generation
└── multi_view_spec.json     # For multi-view generation
```

#### Step-by-Step: Creating Specs for New hdmap-chunks Data

**Step 1: Create new spec files**

```bash
cd /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/

# Copy existing specs as templates
cp single_view_spec.json single_view_spec_chunks.json
cp multi_view_spec.json multi_view_spec_chunks.json
```

**Step 2: Edit `single_view_spec_chunks.json`**

Replace `hdmap/` with `hdmap-chunks/` in the path:

```json
{
    "hdmap": {
        "control_weight": 1,
        "input_control": "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_front_wide/ABC1_1735885669_0.mp4"
    }
}
```

Quick edit with `sed`:
```bash
sed 's|/ruqi_render/hdmap/|/ruqi_render/hdmap-chunks/|g' single_view_spec.json > single_view_spec_chunks.json
```

**Step 3: Edit `multi_view_spec_chunks.json`**

Replace ALL 6 paths (all `hdmap/` → `hdmap-chunks/`):

```json
{
    "hdmap": {
        "control_weight": 1,
        "input_control": [
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_front_wide/ABC1_1735885669_0.mp4",
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_left_front/ABC1_1735885669_0.mp4",
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_right_front/ABC1_1735885669_0.mp4",
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_back/ABC1_1735885669_0.mp4",
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_left_back/ABC1_1735885669_0.mp4",
            "/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap-chunks/pinhole_right_back/ABC1_1735885669_0.mp4"
        ],
        "ckpt_path": "nvidia/Cosmos-Transfer1-7B-Sample-AV-Single2MultiView/t2w_hdmap_control.pt"
    }
}
```

Quick edit with `sed`:
```bash
sed 's|/ruqi_render/hdmap/|/ruqi_render/hdmap-chunks/|g' multi_view_spec.json > multi_view_spec_chunks.json
```

#### Create Prompt JSON File

**Location:** `/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/prompts/`

**Format:**

```json
{
    "0": "A daytime scene from the perspective of someone inside a vehicle looking out onto a modern city street..."
}
```

**Multiple Variations Example:**

```json
{
    "0": "Caption for variation 0 - sunny day",
    "1": "Caption for variation 1 - rainy day",
    "2": "Caption for variation 2 - nighttime"
}
```

**Key Points:**
- The key (e.g., `"0"`, `"1"`) becomes part of the output filename: `ABC1_1735885669_0.mp4`
- The value is your text prompt describing the scene
- Multi-view will use this prompt for all 6 camera views

**Example: Create a new prompt file**

```bash
nano /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/prompts/ABC1_1735885669_chunks.json
```

Or copy and edit existing:
```bash
cp /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/prompts/ABC1_1735885669.json \
   /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/prompts/ABC1_1735885669_chunks.json
# Then edit the file with your new caption
```

#### Update Generation Commands

When running with the new specs, update the `--controlnet_specs` argument:

```bash
# For single-view
--controlnet_specs /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/single_view_spec_chunks.json

# For multi-view
--controlnet_specs /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/multi_view_spec_chunks.json
```

The `--caption_path` for multi-view should still point to the `prompts/` directory (which contains all your prompt JSON files).

### Quick Checklist for New Data

| Step | Action | Command/File | Details |
|------|--------|--------------|---------|
| 1 | Render HD map videos | `render_from_rds_hq.py` | Output to `ruqi_render/hdmap-chunks/` |
| 2 | Create symlinks | `./prepare_hdmap_for_generation.sh /path/to/hdmap-chunks` | Creates `ftheta_camera_*` links |
| 3 | Create new spec files | `cp *.json *_chunks.json` + `sed` to replace paths | Update `hdmap/` → `hdmap-chunks/` |
| 4 | Create prompt JSON | Edit `prompts/<scene>_chunks.json` | Add your caption text |
| 5 | Run single-view | `--controlnet_specs specs/single_view_spec_chunks.json` | Output to `single_view_output/` |
| 6 | Run multi-view | `--controlnet_specs specs/multi_view_spec_chunks.json` | Output to `multi_view_output/` |

### Quick Reference: All Paths for hdmap-chunks

```bash
# Directory locations
DATA_ROOT="/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669"
HDMAP_DIR="$DATA_ROOT/ruqi_render/hdmap-chunks"
SPECS_DIR="$DATA_ROOT/specs"
PROMPTS_DIR="$DATA_ROOT/prompts"

# Spec files
SINGLE_SPEC="$SPECS_DIR/single_view_spec_chunks.json"
MULTI_SPEC="$SPECS_DIR/multi_view_spec_chunks.json"

# Prompt file
PROMPT_JSON="$PROMPTS_DIR/ABC1_1735885669_chunks.json"

# Single-view output
SINGLE_OUTPUT="$DATA_ROOT/single_view_output"

# Multi-view output
MULTI_OUTPUT="$DATA_ROOT/multi_view_output"
```

## 3. Generate Single-View Videos

Once you have the HD map videos rendered, you can generate synthetic videos using the Cosmos Transfer1 model.

### Prerequisites

1. **Environment Setup**: Follow the setup instructions in `cosmos-transfer1/INSTALL.md`
2. **Download Checkpoints**: Run the download script to get required model weights
3. **Fix Known Issues** (see Troubleshooting below if needed)

### Directory Structure

Ensure your data is organized as follows:
```
/data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/
├── prompts/
│   └── ABC1_1735885669.json          # Text prompts
├── ruqi_render/
│   └── hdmap/
│       └── pinhole_front_wide/        # HD map control videos
├── specs/
│   └── single_view_spec.json         # ControlNet configuration
└── single_view_output/               # Output directory (will be created)
```

### Single-View Generation Command

```bash
cd /home/rlan/projects/Cosmos-Drive-Dreams

# Set environment variables
export CUDA_HOME=/data/home/rlan/envs/cosmos-drive-dreams-py3.12
export LD_LIBRARY_PATH=/data/home/rlan/envs/cosmos-drive-dreams-py3.12/lib/python3.12/site-packages/nvidia/cudnn/lib:/data/home/rlan/envs/cosmos-drive-dreams-py3.12/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/cosmos-transfer1:$(pwd)/cosmos-drive-dreams-toolkits

# Run single-view generation
python scripts/generate_video_single_view.py \
  --caption_path /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669 \
  --input_path /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669 \
  --video_save_folder /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/single_view_output \
  --checkpoint_dir checkpoints \
  --is_av_sample \
  --controlnet_specs /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/single_view_spec.json \
  --num_steps 35 \
  --guidance 5.0 \
  --num_gpus 1
```

**Output**: Generates `single_view_output/ABC1_1735885669_0.mp4` (approximately 13-15 minutes per video)

## 4. Generate Multi-View Videos

After single-view generation, you can expand to 6 synchronized camera views.

### Directory Structure Requirements

The multi-view script expects specific camera view names. Create symlinks if your structure differs:

```bash
cd /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render/hdmap

# Create symlinks for expected view names
ln -s pinhole_front_wide ftheta_camera_front_wide_120fov
ln -s pinhole_left_front ftheta_camera_cross_left_120fov
ln -s pinhole_right_front ftheta_camera_cross_right_120fov
ln -s pinhole_back ftheta_camera_rear_tele_30fov
ln -s pinhole_left_back ftheta_camera_rear_left_70fov
ln -s pinhole_right_back ftheta_camera_rear_right_70fov
```

### Multi-View Generation Command

```bash
cd /home/rlan/projects/Cosmos-Drive-Dreams

# Set environment variables (same as single-view)
export CUDA_HOME=/data/home/rlan/envs/cosmos-drive-dreams-py3.12
export LD_LIBRARY_PATH=/data/home/rlan/envs/cosmos-drive-dreams-py3.12/lib/python3.12/site-packages/nvidia/cudnn/lib:/data/home/rlan/envs/cosmos-drive-dreams-py3.12/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/cosmos-transfer1:$(pwd)/cosmos-drive-dreams-toolkits

# Run multi-view generation
python scripts/generate_video_multi_view.py \
  --caption_path /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/prompts \
  --input_path /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/ruqi_render \
  --input_view_path /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/single_view_output \
  --video_save_folder /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/multi_view_output \
  --checkpoint_dir checkpoints \
  --is_av_sample \
  --controlnet_specs /data/datasets/ruqi/scenes-cosmos/ABC1_1735885669/specs/multi_view_spec.json \
  --num_steps 35 \
  --guidance 5.0 \
  --num_gpus 1
```

**Output**: Generates `multi_view_output/ABC1_1735885669_0/` containing:
- `0.mp4` - Front view
- `1.mp4` - Left front view
- `2.mp4` - Right front view
- `3.mp4` - Rear view
- `4.mp4` - Rear left view
- `5.mp4` - Rear right view
- `grid.mp4` - 2x3 grid layout

**Duration**: Approximately 40-45 minutes per generation

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_steps` | Diffusion sampling steps | 35 |
| `--guidance` | Classifier-free guidance scale | 5.0 |
| `--num_gpus` | Number of GPUs for parallel processing | 1 |
| `--fps` | Output video FPS | 24 |
| `--seed` | Random seed for reproducibility | 1 |
| `--offload_diffusion_transformer` | Offload DiT to save VRAM | false |
| `--offload_text_encoder_model` | Offload text encoder | false |

## Troubleshooting Tips

### Conversion/Rendering Issues

* If you see "Unable to determine any camera resolutions", make sure `--data-root` points at the dataset root or the scene folder (the script now auto-adjusts one level up when the latter is used).
* Black/empty HD maps usually mean the intrinsics were wrong. Re-run conversion with the updated script so the `(cx, cy, fx, fy)` swap is applied.
* Missing LiDAR warnings are expected for some Ruqi scenes; they do not stop the conversion.

### Video Generation Issues

**Issue**: `transformer_engine.pytorch has no attribute 'Linear'`

**Cause**: The `transformer_engine` package installation is corrupted or missing `__init__.py`.

**Fix**: Reinstall with correct environment variables:
```bash
CUDA_HOME=/path/to/conda/env LD_LIBRARY_PATH=/path/to/conda/env/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
pip install transformer-engine[pytorch]
```

**Issue**: `ValueError: 'aimv2' is already used by a Transformers config`

**Cause**: Version conflict between `vllm` and `transformers`.

**Fix**: Patch `/path/to/conda/env/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py` line 75:
```python
# Before:
AutoConfig.register("aimv2", AIMv2Config)
# After:
AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)
```

**Issue**: `undefined symbol: cudnnGetLibConfig`

**Cause**: System CUDA at `/usr/local/cuda-*/lib/` has incomplete cuDNN; conda's cuDNN should take priority.

**Fix**: Add conda's cuDNN path first in `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/path/to/conda/env/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

**Issue**: `hdmap files ... does not exist` (multi-view)

**Cause**: Script expects `ftheta_camera_*` view names but your data uses `pinhole_*` names.

**Fix**: Create symlinks matching expected names (see Multi-View section above).

### Complete Environment Setup

If you're starting fresh, here's the complete environment setup sequence:

```bash
# 1. Create conda environment
conda env create --file cosmos-transfer1/cosmos-transfer1.yaml
conda activate cosmos-drive-dreams-py3.12

# 2. Install dependencies
pip install -r cosmos-transfer1/requirements.txt

# 3. Install vllm (use specific version to avoid conflicts)
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0

# 4. Install decord
pip install decord==0.6.0

# 5. Patch Transformer Engine linking
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12

# 6. Install Transformer Engine with correct env
CUDA_HOME=$CONDA_PREFIX \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
pip install transformer-engine[pytorch]

# 7. Patch vllm aimv2 conflict
sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' \
    $CONDA_PREFIX/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py

# 8. Test environment
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos-transfer1/scripts/test_environment.py
```

---

Append additional notes or commands here as new datasets or workflows are added.
