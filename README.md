# Algae Counter

A Python pipeline for counting algae dots in imperfect microscope-style images of 4x4 grid samples.

This project was built for datasets where a single physical sample may be photographed in multiple overlapping, partially cut, or imperfect images such as:

- `ABCD 1 (1)`
- `ABCD 1 (2)`

Both of those belong to the same sample, `ABCD 1`.

The pipeline:

- groups images belonging to the same sample
- reconstructs the sample geometry from partial views
- handles overlap between images of the same sample
- counts only the small algae-like dots
- filters out large splotches / blobs
- outputs both total algae counts per sample and per-box counts for each cell in the 4x4 grid

## Features

- Robust sample grouping from filenames like `NAME N (k)`
- Lightweight geometric alignment for multi-image samples
- Fallback grid inference when some grid lines are missing or only partially visible
- Optional Cellpose-based final algae detection
- Large-blob filtering to reduce false counts from splotches
- Duplicate-merging across overlapping photos
- CSV outputs for totals, box counts, raw detections, clustered detections, and mappings
- Debug visualizations for inspection
- Local CPU workflow and Modal GPU workflow

## How it works

### 1. Group images by sample
Files are grouped by their shared stem before the final parenthesized image index.

Example:

- `CG CL 1 (1).bmp`
- `CG CL 1 (2).bmp`

Both map to the sample key:

- `CG CL 1`

### 2. Detect / infer the 4x4 grid
The script looks for bright grid lines and infers the 5 vertical and 5 horizontal boundaries needed for a 4x4 grid.

If some lines are missing or only partially visible, it does not drop the sample immediately. Instead, it tries to infer the full regular grid from the partial detections.

### 3. Align multiple imperfect photos of the same sample
If a sample has multiple images, the pipeline estimates how they overlap in a shared sample coordinate system.

Important design choice:

- alignment uses lightweight grid / geometry logic
- Cellpose is used only for final algae detection

This was intentional because alignment is primarily a geometric problem, while Cellpose is better used for object detection.

### 4. Detect algae candidates
There are two detector modes:

#### Threshold detector
A fast classical detector based on:

- grayscale conversion
- Gaussian background subtraction
- thresholding
- morphological cleanup
- connected components

#### Cellpose detector
A heavier learned detector that can improve detection of faint or irregular algae-like dots.

The current implementation supports:

- pretrained Cellpose models such as `cpsam`
- optional black-hat preprocessing before Cellpose
- filtering by size and aspect ratio after segmentation

### 5. Filter non-algae blobs
Large blobs and highly elongated detections can be rejected using:

- minimum area fraction
- maximum area fraction
- maximum aspect ratio

This helps remove large splotches and non-dot artifacts.

### 6. Merge duplicates in overlap regions
When two images of the same sample overlap, the same algae may appear twice.

The pipeline merges detections in the shared coordinate system so the same algae is counted once.

### 7. Write outputs
The script writes:

- total algae count per sample
- algae count per box in the 4x4 grid
- raw detections
- merged / clustered detections
- alignment mappings
- summary JSON
- debug images

## Project structure

A minimal structure looks like:

```text
.
├── main.py
├── README.md
├── data/
│   ├── CG CL 1 (1).bmp
│   ├── CG CL 1 (2).bmp
│   └── ...
└── output/
```

## Requirements

### Python libraries

- `numpy`
- `opencv-python` or `opencv-python-headless`
- `cellpose`
- `modal` (only if using Modal)

The script also uses standard-library modules such as:

- `argparse`
- `csv`
- `json`
- `math`
- `re`
- `pathlib`
- `dataclasses`
- `itertools`

## Installation

### Local environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy opencv-python cellpose modal
```

If you only want local CPU execution and do not need Modal yet, you can still install the same set.

## Running locally

### Threshold detector

```bash
python main.py \
  --input_dir data \
  --output_dir output \
  --detector threshold
```

### Cellpose detector

```bash
python main.py \
  --input_dir data \
  --output_dir output \
  --detector cellpose \
  --cellpose_model_type cpsam \
  --cellpose_cellprob_threshold -1.5 \
  --cellpose_min_size_px 4 \
  --cellpose_blackhat_px 21 \
  --cellpose_batch_size 1 \
  --cellpose_tile_size 256 \
  --min_blob_area_frac 1e-6 \
  --max_blob_area_frac_abs 0.002 \
  --merge_radius_frac 0.01
```

## Running on Modal GPU

This project supports running the pipeline remotely on Modal with GPU acceleration.

### 1. Install and set up Modal

```bash
pip install modal
modal setup
```

### 2. Create Modal volumes

```bash
modal volume create algae-input
modal volume create algae-output
modal volume create algae-model-cache
```

### 3. Upload your local data folder

If your local data is in a folder called `data`:

```bash
modal volume put algae-input data /
```

### 4. Run the pipeline remotely

```bash
modal run main.py::modal_main \
  --input-subdir data \
  --output-subdir outputc2 \
  --detector cellpose \
  --cellpose-model-type cpsam \
  --cellpose-cellprob-threshold -1.5 \
  --cellpose-min-size-px 4 \
  --cellpose-blackhat-px 21 \
  --cellpose-batch-size 1 \
  --cellpose-tile-size 256 \
  --min-blob-area-frac 1e-6 \
  --max-blob-area-frac-abs 0.002 \
  --merge-radius-frac 0.01
```

### 5. Download the results back locally

```bash
modal volume get algae-output /outputc2 ./outputc2
```

## Output files

Typical outputs include:

- `sample_counts.csv` — total algae count per sample
- `box_counts.csv` — algae count per box in the 4x4 grid
- `raw_detections.csv` — raw per-image detections before overlap clustering
- `clustered_detections.csv` — merged detections after duplicate reconciliation
- `mappings.csv` — image-to-sample alignment information
- `summary.json` — configuration and totals summary
- debug images in a `debug/` subfolder

## Important CLI arguments

### General

- `--input_dir` — local input image directory
- `--output_dir` — local output directory
- `--detector` — `threshold` or `cellpose`
- `--quiet` — reduce logging

### Threshold detector tuning

- `--dark_threshold`
- `--min_blob_area_frac`
- `--max_blob_area_frac_abs`
- `--grid_line_margin_frac`
- `--merge_radius_frac`

### Cellpose tuning

- `--cellpose_model_type`
- `--cellpose_diameter`
- `--cellpose_flow_threshold`
- `--cellpose_cellprob_threshold`
- `--cellpose_min_size_px`
- `--cellpose_blackhat_px`
- `--cellpose_batch_size`
- `--cellpose_tile_size`
- `--use_cellpose_gpu`

## Recommended detector choice

### Alignment
Use the built-in lightweight geometric alignment.

This project intentionally does not use Cellpose for alignment because:

- alignment is a grid / geometry problem
- Cellpose is slower
- Cellpose helps more for final algae detection than for reconstructing the grid layout

### Final algae detection
Use Cellpose when:

- algae are faint
- the background is uneven
- thresholding undercounts too much

Use the threshold detector when:

- you want faster iteration
- images are relatively clean
- you want a simpler baseline

## Typical workflow

1. Put all microscope images in `data/`
2. Run the threshold detector first as a quick baseline
3. Switch to Cellpose if thresholding undercounts or misses faint dots
4. Inspect debug images and CSVs
5. Tune the blob-size and merge parameters if needed
6. Optionally run on Modal GPU for faster Cellpose inference

## Known design decisions

- Samples are identified from filenames, not folder structure
- A single sample can be represented by multiple overlapping image files
- Missing grid lines are handled by inference when possible
- Large splotches are treated as non-algae and filtered by postprocessing
- Overlap merging is done in shared sample coordinates

## Git / repo hygiene

Common files to ignore in `.gitignore`:

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.DS_Store
```

## Future improvements

Potential next steps for this project:

- fine-tuning Cellpose on labeled algae masks
- adding a dedicated failure report CSV for problematic samples
- adding a small calibration / benchmarking script against manual counts
- improving debug overlays for inferred grid lines and overlap regions
- adding batch evaluation metrics for comparing threshold vs Cellpose detectors

## License

Add your preferred license here.

## Acknowledgments

This project uses:

- OpenCV for image processing
- NumPy for array operations
- Cellpose for learned segmentation-based detection
- Modal for remote GPU execution
