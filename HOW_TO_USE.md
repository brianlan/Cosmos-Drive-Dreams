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

* `-i / --input_root`: the clip’s RDS-HQ folder (the converter’s `output-root/clip-id`).  
* `-o / --output_root`: where to store the rendered videos.  
* `-cj`: clip list. You can pass a single ID (as above), a JSON file containing a list, or omit to render all clips.  
* `-d`: dataset configuration. Use `ruqi` to pick up the Ruqi camera set and FPS.  
* `-c`: camera model (`pinhole` for Ruqi).  
* `-s lidar`, `-s world_scenario`: optional skips; remove them if you want those modalities too.

The renderer automatically upsamples the 10 Hz pose stream to 30 FPS (Cosmos’ assumption) and uses the per-camera intrinsics derived during conversion. The resulting mp4s end up under `ruqi_render/hdmap/pinhole_<camera_name>/`.

### Chunk Length / Frame Count

Dataset configs can cap each rendered clip via `TARGET_CHUNK_FRAME` and `MAX_CHUNK` (see `cosmos-drive-dreams-toolkits/config/dataset_ruqi.json`).  

* `TARGET_CHUNK_FRAME` – maximum frames per chunk.  
* `MAX_CHUNK` – how many chunks to render (`-1` means “no limit”).  

Set these high enough (e.g., `TARGET_CHUNK_FRAME: 600`, `MAX_CHUNK: -1`) if you want the full 20 s Ruqi clips instead of the default 121-frame segments.

## Troubleshooting Tips

* If you see “Unable to determine any camera resolutions”, make sure `--data-root` points at the dataset root or the scene folder (the script now auto-adjusts one level up when the latter is used).  
* Black/empty HD maps usually mean the intrinsics were wrong. Re-run conversion with the updated script so the `(cx, cy, fx, fy)` swap is applied.  
* Missing LiDAR warnings are expected for some Ruqi scenes; they do not stop the conversion.

Append additional notes or commands here as new datasets or workflows are added.
