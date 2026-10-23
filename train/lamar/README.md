# Training and evaluation on LaMAR HGE

SceneAligner trained on the phone sequences of the HGE building of [LaMAR](https://lamar.ethz.ch/), with the
train/test split of [UnLoc](https://github.com/matthias-wueest/UnLoc) and the evaluation protocol of Z-FLoc, so
that the three methods compare on the same windows of frames.

The training data follow those of SceneAligner: `prepare_c3.py` (in the wild) as the base, and `prepare_s3d.py`
(Structured3D, in the upstream repository [Cornell-VAILab/SceneAligner](https://github.com/Cornell-VAILab/SceneAligner),
removed from this fork) where correspondences come from ground-truth geometry instead of annotations:

| | C3 (`prepare_c3.py`) | Structured3D (`prepare_s3d.py`) | LaMAR HGE (`prepare_lamar.py`) |
|---|---|---|---|
| reconstruction | pi3 / VGGT, 50 to 150 photos, or one | pi3, one view | MapAnything with ground-truth intrinsics and relative poses, windows of 15 to 100 frames, every 1st, 2nd or 4th frame |
| gravity | GeoCalib | GeoCalib | ground-truth pose (z-up world) |
| reconstruction samples | conf 30/45/60/75/90 x x/z 0/2.5/5 x gamma 0.25/0.5/0.75/1 | conf 45, x/z 2.5, gamma 0.5 | as C3 |
| ground-truth geometry samples | - | density map of the ground-truth depth | density map of the LiDAR depth maps |
| correspondences | annotated | ground-truth points on walls (dark pixels of the 512 px plan) | points within 0.5 m of a wall of hge.png |
| minimum correspondences | 8 per photo | 10 per sample | 8 per frame, 10 per LiDAR sample |
| consistency test | RANSAC similarity, 10 px of 512 | - | as C3 (reconstruction samples) |
| floorplan | 512 px | 512 px | 1024 px, walls thickened to 0.5 m (1-3 px lines in hge.png) |
| `train.py` | defaults | `--epochs 20 --resolutions 512 --min_points 1000` | `--epochs 20 --resolutions 1024 --min_points 1000 --negatives batch --plan_grid 8` |

The 0.5 m wall tolerance covers the error of the affine transform (about 0.25 m); a first version with 0.2 m lost
most far walls. `--negatives batch` and `--plan_grid 8` are the only additions to the training of SceneAligner (see
Training). `train.py` also logs to Weights & Biases (`--wandb_project`), saves a visualization every
`--vis_every` steps and resumes after the last completed epoch.

## Data

```
<LAMAR_ROOT>/
├── HGE/sessions/map/                # benchmark release (https://lamar.ethz.ch/lamar/): trajectories.txt, sensors.txt,
│   └── raw_data/ios_<date>_<time>_<NNN>/images/<timestamp>.jpg
├── ios_<date>_<time>/               # raw phone sessions (shells/lamar/download_lamar.sh): depths.txt, sensors.txt,
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
- Training frames: the indoor intervals of UnLoc, inside the floorplan.
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

`import mapanything` also succeeds from a directory that contains a `mapanything/` folder (e.g. next to a clone,
as a namespace package) without the package being installed: check with
`cd / && python -c "import mapanything.models"`. The jobs of `shells/lamar/` do this check first.

The weights `facebook/map-anything` (CC BY-NC 4.0) are downloaded from the Hugging Face Hub;
`--mapanything_model facebook/map-anything-apache` selects the Apache-2.0 weights.

## Training data

```bash
python train/lamar/prepare_lamar.py --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --output <data> --recon_cache <data>/recon_cache
```

For every train session, windows of 15, 20, 35, 50 and 100 frames, taken every 1st, 2nd or 4th kept frame
(`--steps 1 2 4`: wider views, as the photo collections of C3; starts every half span; no window across a gap of
more than 5 s of the kept frames), are reconstructed with MapAnything and cached as
`<session>_t<first timestamp>_L<length>_k<step>.npz`. Each window gives:
- `mapanything/`: density maps of the reconstruction for the confidence thresholds, x/z slices and gammas of
  `prepare_c3.py`, with the correspondences of the frames that have at least 8 of them, kept if they agree with a
  similarity transform. `--plan_source lidar` places the correspondences with the LiDAR points instead of the
  reconstructed ones (as `prepare_s3d.py` does with the ground-truth depth).
- `lidar/`: one density map of the LiDAR points (gamma 0.5), seen from above in the gravity-aligned frame of the
  first camera, as `prepare_s3d.depth_sample`, if it has at least 10 correspondences.

The CSV columns `rgb`, `camera_plan` and `camera_density` (the photo and the position of the last camera) are only
used by the visualizations of `train.py`. All windows share the floorplan, so `scene_id` is `hge` for all of them;
`--scene_id cell` names 20 m cells instead.

To look at the correspondences, `visualize_correspondences.py` draws, for random windows of the cache (or
`--windows <name>`), the pixels of a frame whose points lie on walls, these points on a density map (`--conf`,
`--xz`) and on hge.png (cropped around them, the camera in blue), in seven colours from the left (red) to the right
(violet) of the photo. It reads
`recon_cache` and needs no GPU:

```bash
python train/lamar/visualize_correspondences.py --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --recon_cache <data>/recon_cache --output <data>/vis_correspondences
```

## Training

```bash
python train.py --data <data> --output <run> --epochs 20 --resolutions 1024 --min_points 1000 \
    --negatives batch --plan_grid 8 --wandb_project scenealigner-lamar --wandb_name <run>
```

Every 100 steps the losses go to `log.csv` and wandb; every `--vis_every` 500 steps, `<run>/vis/<step>.png` and
the wandb panel `sample` show the floorplan of the first sample of the batch with the ground-truth (green) and
predicted (orange) correspondences, the ground-truth (blue) and estimated (red) position of the last camera, next
to its photo. If `<run>/state.pt` exists, the training resumes after the last completed epoch (resubmit the job
when it reaches its walltime).

The points of a window lie in a few patches of the building plan (1 patch = 2.8 m at 1024 px), so their floorplan
descriptors are almost equal and, with only the points of the sample as negatives, the feature matching loss stays
at its chance level log(1024) whatever the model (first run). `shells/train/train.sh` therefore passes
`--negatives batch` (the points of the other windows of the batch are negatives as well) and `--plan_grid 8` (one
correspondence per 8 px cell of the floorplan, the free slots being random negatives over the whole plan). `acc`
(valid points whose most similar plan patch is within one patch of the ground truth) and `chance` (the loss of
uniform descriptors) are logged with the losses.

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

`prepare.sh` runs one `prepare_lamar.py` per GPU of the node (`--shard i --num_shards n`, round robin over the
windows, logs in `$DATA/shards/prepare_<i>.log`), then merges their CSV files (`--merge`): use `prepare_hf.sh` (a
full node). `train.py` and `evaluate_lamar.py` use one GPU: `lamar_hg.sh` / `evaluate_hg.sh`.

`REPO`, `LAMAR_ROOT`, `DATA`, `RUN`, `EPOCHS`, `NUM_GPUS`, `NEGATIVES`, `PLAN_GRID`, `PLAN_SIZE`, `CHECKPOINT`, `ZFLOC_NPZ` and `WANDB_PROJECT` are passed with
`qsub -v`, e.g. `qsub -v DATA=data/lamar_hge,EPOCHS=4 shells/train/lamar_hf.sh`. The cached reconstructions of an
older version (`<session>_s<start>_L<length>.npz`, indexed by the frames of other filters) are not used and can be deleted.

## Resolution

The floorplan is given to DINOv3 as a square image of `--plan_size` pixels (default 1024; `train.py --resolutions
1024`, as Structured3D trains at the resolution of its floorplans); the density maps are rendered at 512 px, as for
C3, for training and evaluation. The original plan (3278 px) cannot be used: ViT-B/16 would see about 42k
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
