# Training and evaluation on LaMAR HGE

SceneAligner trained on the phone sequences of the HGE building of [LaMAR](https://lamar.ethz.ch/), with the
train/test split of [UnLoc](https://github.com/matthias-wueest/UnLoc) and the evaluation protocol of Z-FLoc, so
that the three methods compare on the same windows of frames.

Differences from the C3 pipeline:
- The reconstruction is [MapAnything](https://github.com/facebookresearch/map-anything), given the photos, their
  intrinsics and their ground-truth relative poses (`reconstruction.py`). The gravity direction comes from the pose
  of the first frame (the LaMAR world frame is z-up): GeoCalib is not used.
- Every training sample is a window of consecutive frames (15 to 100, like the evaluation windows). The
  correspondences are not annotated: the reconstructed points are placed in the world with the ground-truth pose
  of the first camera and on the floorplan with its affine transform; those on walls of the floorplan are the
  correspondences (`prepare_lamar.py`). The LiDAR depth maps of the phone give a second density map per window.
- The floorplan is stored at 1024 px instead of 512 (see Resolution).
- `train.py` logs to Weights & Biases (`--wandb_project`) and saves a visualization every `--vis_every` steps.

## Data

```
<LAMAR_ROOT>/
├── HGE/sessions/map/                # benchmark release (https://lamar.ethz.ch/lamar/): trajectories.txt, sensors.txt,
│   └── raw_data/ios_<date>_<time>_<NNN>/images/<timestamp>.jpg
├── ios_<date>_<time>/               # raw phone sessions (shells/lamar/download_lamar.sh): depths.txt, sensors.txt, rigs.txt,
│   └── raw_data/depth/<timestamp>.png
└── hge.png                          # the floorplan (3278 x 2334 px), not redistributable
```

- `split.yaml`: the sessions of UnLoc (12 train, 1 val, 3 test). They are sub-sequences `ios_<date>_<time>_<NNN>`
  of the map session; their floorplan-aligned poses are in `HGE/sessions/map/trajectories.txt` (camera-to-world)
  and their photos in `HGE/sessions/map/raw_data/<session>/images/`.
- The LiDAR depth maps (16-bit PNG, millimetres) are in the raw release of the phone session `ios_<date>_<time>`
  (the name without `_<NNN>`), matched to the photos by timestamp. `shells/lamar/download_lamar.sh` fetches the 15 raw sessions
  of the split (about 15 GB) from `https://cvg-data.inf.ethz.ch/lamar/raw/HGE/sessions/`:
  ```bash
  qsub ~/oss/SceneAligner/shells/lamar/download_lamar.sh   # LAMAR_ROOT defaults to /groups/gai51740/RBT3/umemura/zfloc/lamar
  ```
  The benchmark release `HGE.zip` (7 GB) is downloaded from `BENCH_URL` when `HGE/sessions/map` is missing. A raw session outside the split (e.g. `ios_2022-01-20_17.30.07`)
  is not used.
- World to floorplan: the affine transform fitted on four building entrances by UnLoc (Appendix A.1.2), as in
  Z-FLoc: about 18.3 px/m (0.0546 m/px), with a reflection. Errors are converted with 0.055 m/px, the value Z-FLoc
  uses. The fit residual at the entrances is 4 to 5 px (about 0.25 m).
- Training frames: the indoor intervals of UnLoc, inside the floorplan, within 1.5 m of the median camera height.
  Test frames: the index shift of Z-FLoc (`HGE_INDEX_SHIFT` in `lamar.py`). For `ios_2022-01-18_17.10.39_000` the
  shift keeps 516 frames while Z-FLoc evaluates 483: pass `--zfloc_npz <z-floc>/assets/metadata/hge.npz` to
  evaluate exactly the frames of Z-FLoc (they are matched to the bundled ground-truth positions).

Check what the scripts will find (sessions, frames after the filters, depth coverage, sensors, the z-up world):

```bash
python train/lamar/lamar.py --check --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> --floorplan <LAMAR_ROOT>/hge.png
```

## Installation

In the environment of SceneAligner (`requirements.txt`, which the evaluation code imports), add MapAnything and
wandb:

```bash
pip install git+https://github.com/facebookresearch/map-anything.git@3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
pip install wandb
```

The weights `facebook/map-anything` (CC BY-NC 4.0) are downloaded from the Hugging Face Hub;
`--mapanything_model facebook/map-anything-apache` selects the Apache-2.0 weights.

## Training data

```bash
python train/lamar/prepare_lamar.py --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --output <data> --recon_cache <data>/recon_cache
```

For every train session, windows of 15, 20, 35, 50 and 100 consecutive frames (stride: half the length, no window
across a gap of the kept frames) are reconstructed with MapAnything. Each window gives:
- `mapanything/`: density maps of the reconstruction for two confidence thresholds (`--conf 45 75`), three x/z
  slices and four gammas, with correspondences from the reconstructed points on walls. Where the LiDAR is valid,
  points farther than 10 % (or 0.3 m) from it are left out (`--no_lidar_gate` keeps them). `--plan_source lidar`
  takes the LiDAR points instead, as the Structured3D training of the paper does with the ground-truth depth.
- `lidar/`: density maps of the LiDAR points, seen from above in the gravity-aligned frame of the first camera.

Samples with fewer than `--min_correspondences 200` points, or whose correspondences do not agree with a
similarity transform, are dropped. The CSV columns `rgb`, `camera_plan` and `camera_density` (the photo and the
position of the last camera) are only used by the visualizations of `train.py`. All windows share the floorplan,
so `scene_id` is `hge` for all of them (only the points of a sample are its negatives in the feature matching
loss); `--scene_id cell` names 20 m cells instead, so that windows of different cells are negatives of each other.

## Training

```bash
python train.py --data <data> --output <run> --epochs 2 --resolutions 1024 1280 --min_points 1000 \
    --wandb_project scenealigner-lamar --wandb_name <run>
```

Every 100 steps the losses go to `log.csv` and wandb; every `--vis_every` 500 steps, `<run>/vis/<step>.png` and
the wandb panel `sample` show the floorplan of the first sample of the batch with the ground-truth (green) and
predicted (orange) correspondences, the ground-truth (blue) and estimated (red) position of the last camera, next
to its photo. `--min_points 1000` keeps the random crops (a sample has 200 to 20000 correspondences).

## Evaluation

```bash
python train/lamar/evaluate_lamar.py --map_dir <LAMAR_ROOT>/HGE/sessions/map --floorplan <LAMAR_ROOT>/hge.png \
    --checkpoint_dir <run> --recon_cache <cache> [--zfloc_npz <z-floc>/assets/metadata/hge.npz] [--save_vis]
```

The protocol of Z-FLoc: for each length L in 15, 20, 35, 50, 100, the kept frames of the three test sessions are
cut into non-overlapping windows of L frames (the remainder is dropped); a window succeeds when the positions of
its last 10 cameras are all within 1 m of the ground truth; SR@1m is the mean over the windows of all sessions,
averaged over the seeds 42 to 46 (which seed the query sampling and RANSAC of the alignment; the
reconstructions are deterministic). A window without a similarity transform counts as a failure. The JSON also
holds the median position and angular errors. Baselines: `--checkpoint_dir pre-trained` (DINOv3 only) and
`--checkpoint_dir jhcho99/SceneAligner` (the LoRA layers trained on C3). `--sessions val --seq_length 20 --seeds 42`
evaluates the val session (indoor frames) for model selection.

## ABCI jobs

The jobs in `shells/` (PBS, conda environment `scenealigner`) run in `REPO`, by default `~/oss/SceneAligner`
(PBS runs a copy of the script, so neither its location nor the submission directory is used); every GPU job has a rt_HF and a rt_HG version, thin wrappers that pass `DATA` and `RTYPE` to the common script:

```bash
qsub shells/lamar/download_lamar.sh                                        # raw sessions, HGE/sessions/map, hge.png
qsub shells/lamar/prepare_hg.sh                                            # lamar.py --check, prepare_lamar.py -> data/lamar_hge
qsub shells/train/lamar_hg.sh                                              # train.py -> outputs/lamar_lamar_hge_rt_HG, wandb
qsub -v CHECKPOINT=outputs/lamar_lamar_hge_rt_HG shells/lamar/evaluate_hg.sh  # val, then test
```

`REPO`, `LAMAR_ROOT`, `DATA`, `RUN`, `EPOCHS`, `PLAN_SIZE`, `CHECKPOINT`, `ZFLOC_NPZ` and `WANDB_PROJECT` are passed with
`qsub -v`, e.g. `qsub -v DATA=data/lamar_hge,EPOCHS=4 shells/train/lamar_hf.sh`.

## Resolution

The floorplan and the density maps are given to DINOv3 as square images of `--plan_size` pixels (default 1024;
`train.py --resolutions 1024 1280`). The original plan (3278 px) cannot be used: ViT-B/16 would see about 42k
tokens, and the LoRA layers are trained at 512 to 1280 px. SceneAligner resizes the C3 floorplans to 512 px, but
HGE is a whole building (179 m x 127 m): at 512 px, 1 px = 0.35 m and 1 m = 2.9 px, too coarse for SR@1m; at
1024 px, 1 px = 0.175 m and 1 m = 5.7 px (a 16 px patch covers 2.8 m). The resolution only changes the precision of
the correspondences and the compute; the ground truth and the metric (full-resolution pixels, 0.055 m/px) do not
depend on it. `--plan_size 512` reproduces the setting of the paper.

Known limits: the LiDAR reaches about 5 m and has holes in the halls; windows without walls (open halls) and
windows across the index shift of the test sessions (two separate parts of the building in one density map) are
expected to fail; the ground-truth affine transform itself is accurate to about 0.25 m.

## Unit tests

```bash
python -m pytest train/lamar/tests
```
