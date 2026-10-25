# Training and evaluation on LaMAR HGE

SceneAligner trained on the phone sequences of the HGE building of [LaMAR](https://lamar.ethz.ch/), with the
train/test split of [UnLoc](https://github.com/matthias-wueest/UnLoc) and the evaluation protocol of Z-FLoc, so
that the three methods compare on the same windows of frames.

The training data follow those of SceneAligner: `prepare_c3.py` (in the wild) as the base, and `prepare_s3d.py`
(Structured3D, in the upstream repository [Cornell-VAILab/SceneAligner](https://github.com/Cornell-VAILab/SceneAligner),
removed from this fork) where correspondences come from ground-truth geometry instead of annotations:

| | C3 (`prepare_c3.py`) | Structured3D (`prepare_s3d.py`) | LaMAR HGE (`prepare_lamar.py`) |
|---|---|---|---|
| reconstruction | pi3 / VGGT, 50 to 150 photos, or one | pi3, one view | MapAnything with ground-truth intrinsics and relative poses on chunks of up to 500 consecutive frames of a session (cached); per epoch, 8864 random windows assembled from the cached points: 50 to 500 frames drawn among n to 8 n consecutive frames, in half of them a quarter to a half from other sessions within 2 m, a random 25 to 100 % of the pixels of every frame, every window a different combination of frames |
| gravity | GeoCalib | GeoCalib | ground-truth pose (z-up world) |
| reconstruction samples | conf 30/45/60/75/90 x x/z 0/2.5/5 x gamma 0.25/0.5/0.75/1 | conf 45, x/z 2.5, gamma 0.5 | one of those of C3 per window, at random |
| ground-truth geometry samples | - | density map of the ground-truth depth | density map of the LiDAR depth maps |
| correspondences | annotated | ground-truth points on walls (dark pixels of the 512 px plan) | points within 0.2 m of a wall of hge.png |
| minimum correspondences | 8 per photo | 10 per sample | 8 per frame, 10 per LiDAR sample |
| consistency test | RANSAC similarity, 10 px of 512 | - | as C3 (reconstruction samples) |
| floorplan | 512 px | 512 px | 1024 px, walls thickened to 0.5 m (1-3 px lines in hge.png) |
| `train.py` | defaults | `--epochs 20 --resolutions 512 --min_points 1000` | `--epochs 20 --resolutions 1024 --min_points 1000 --negatives batch --plan_grid 8` |

The 0.2 m wall tolerance is strict: with the error of the affine transform (about 0.25 m) and of the reconstruction,
far walls lose many points (0.5 m kept them but was found too loose; `--max_wall_dist` changes it). `--negatives batch` and `--plan_grid 8` are the only additions to the training of SceneAligner (see
Training). `train.py` also logs to Weights & Biases (`--wandb_project`), saves a visualization every
`--vis_every` steps and, with `--resume`, resumes after the last completed epoch.

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
python train/lamar/prepare_lamar.py --stage recon --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --session_cache <data>/session_cache                        # GPU
python train/lamar/prepare_lamar.py --stage samples --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --session_cache <data>/session_cache --output <data> --epochs 20  # CPU
```

`--stage recon` reconstructs the kept frames of every session once with MapAnything, in chunks of at most
`--chunk_size 500` consecutive frames (`lamar.session_chunks`, inside runs without a gap of more than 5 s), and
caches them in `<data>/session_cache/<session>/t<first timestamp>_n<frames>/` (`reconstruction.save_chunk`): the
points of every frame in its own camera frame (float16, memory-mapped), its confidence, mask, image and intrinsics.
Delete the cache after changing `--chunk_size` (`lidar_cam.npy` of an older version is not used).

`--stage samples` draws, for each of `--epochs` epochs, `--windows_per_epoch 8864` random windows
(`lamar.sample_epoch_windows`; with a LiDAR sample each, 17728 samples, 554 steps of 32: half the 1108 of an epoch of the 600
windows x 60 density maps before):
- a session drawn proportionally to its kept frames, a run of kept frames drawn proportionally to its length, n = 50
  to 500 frames (`--num_frames`, at most the run), drawn at random in a span of n to 8 n consecutive kept frames
  (`--max_spread`; consecutive frames are about 0.5 m apart) at a random start (`lamar.sample_window`);
- in half of the windows (`--mix_prob 0.5`), a quarter to a half of the frames (all but the last) replaced with frames
  of one or two other sessions of the split whose cameras are within 2 m (`--mix_radius`) of a camera of the window
  (`lamar.mix_sessions`; the ground-truth poses of all the sessions share the world frame);
- every frame keeps a random fraction in [0.25, 1] (`--min_keep`) of its pixels and of its LiDAR points; the
  percentiles of the density map are those of the kept points;
- every window a combination of frames not drawn before, in any epoch.

A window takes the cached points of its frames, placed with the ground-truth poses relative to its first frame
(`reconstruction.window_from_cache`), so a window can span chunks and sessions. The windows of epoch e go to
`<data>/epoch_<e>` (`meta/windows.csv` lists their frames as `<session>:<timestamp>` and the kept fractions); the test
windows stay those of Z-FLoc (15 to 100 consecutive frames). The val set has 600 windows (`VAL_WINDOWS`). Each
window gives:
- `mapanything/`: one density map of the reconstruction, for a random confidence threshold, x/z slice and gamma of
  `prepare_c3.py` (the threshold and slice pairs are tried in random order until one keeps the correspondences of
  the frames that have at least 8 of them and they agree with a similarity transform). `--plan_source lidar` places
  the correspondences with the LiDAR points instead of the reconstructed ones (as `prepare_s3d.py` does with the
  ground-truth depth).
- `lidar/`: one density map of the LiDAR points (gamma 0.5) at the resolution of the depth maps
  (`lamar.lidar_points`), seen from above in the gravity-aligned frame of the first camera, as
  `prepare_s3d.depth_sample`, if it has at least 10 correspondences; at most `--max_points` of them are kept, at random.

The CSV columns `rgb`, `camera_plan` and `camera_density` (the photo and the position of the last camera) are only
used by the visualizations of `train.py`. All windows share the floorplan, so `scene_id` is `hge` for all of them;
`--scene_id cell` names 20 m cells instead.

To look at the correspondences, `visualize_correspondences.py` draws, for random windows of one epoch (or
`--windows <name>`), the pixels of a frame whose points lie on walls, these points on a density map (`--conf`,
`--xz`) and on hge.png (cropped around them, the camera in blue), in seven colours from the left (red) to the right
(violet) of the photo. It reads
the session cache and needs no GPU:

```bash
python train/lamar/visualize_correspondences.py --map_dir <LAMAR_ROOT>/HGE/sessions/map --raw_dir <LAMAR_ROOT> \
    --floorplan <LAMAR_ROOT>/hge.png --session_cache <data>/session_cache --data <data>/epoch_000 \
    --output <data>/vis_correspondences    # val windows: --data <data>/val/epoch_000
```

## Training

```bash
python train.py --data <data> --output <run> --epochs 20 --resolutions 1024 --min_points 1000 \
    --negatives batch --plan_grid 8 --wandb_project scenealigner-lamar --wandb_name <run>
```

Every 100 steps the losses go to `log.csv` and wandb; every `--vis_every` 500 steps, `<run>/vis/<step>.png` and
the wandb panel `sample` show the floorplan of the first sample of the batch with the ground-truth (green) and
predicted (orange) correspondences, the ground-truth (blue) and estimated (red) position of the last camera, next
to its photo. Every job of `train.sh` writes a new run `outputs/lamar_<data>_<RTYPE>_<YYYYmmdd_HHMMSS>`; to
continue a run after its last completed epoch (e.g. at the walltime), `qsub -v RTYPE=rt_HF,RESUME=<run> shells/train/lamar_hf.sh`
(`train.py --resume`: LoRA layers, optimizer, step and wandb run of `<run>/state.pt`). Without `--resume`,
`train.py` refuses an `--output` that already has a `state.pt`.

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
qsub shells/lamar/prepare_hg.sh                                            # lamar.py --check, prepare_lamar.py -> /groups/gai51740/RBT3/umemura/zfloc/dataset/lamar
qsub shells/train/lamar_hg.sh                                              # train.py -> outputs/lamar_lamar_rt_HG_<date>_<time>, wandb
qsub -v RTYPE=rt_HG,CHECKPOINT=outputs/lamar_lamar_rt_HG_<date>_<time> shells/lamar/evaluate_hg.sh  # val, then test
```

`prepare.sh` runs `--stage recon` with one process per GPU of the node (`--workers`), then `--stage samples` with
`NUM_WORKERS` CPU processes (default: the CPUs of the node, at most one per 8 GB of free memory; `EPOCHS` sets of
`WINDOWS` windows, default 20 and 8864; `VAL_WINDOWS` 600). Each stage is one python process that shares the chunks or
windows between its workers (`multiprocessing.Pool`) and shows one progress bar of all of them in the job output
(`PYTHONUNBUFFERED`); a failing worker stops the job with its traceback. The samples do not depend on the number of
workers (the random generator of a window is seeded by its index). The epochs whose CSV files are written are
skipped on a rerun: delete `$DATA/epoch_*` and `$DATA/<split>/epoch_*` after changing `EPOCHS`, `WINDOWS` or `VAL_WINDOWS`.
The data go to `DATA`, by default `/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar`. `evaluate.sh` runs one
process per GPU (`--shard i --num_shards n`, round robin; logs in `$OUT/<val|test>_results.shard<i>.log`) and
merges their outputs (`--merge`; the seeds of the alignment depend on the window, not on the shard): `_hf` uses all the GPUs of
a node, `_hg` one. `train.sh` runs `train.py` with `torchrun` on all the GPUs of the node (DistributedDataParallel,
a batch of 4 windows per GPU: 32 on rt_HF, lr unchanged); the points of the batches of all GPUs are the candidates
of the feature matching loss (all_gather), and the logs are the mean over the GPUs.

`prepare.sh` also makes one set of validation windows: for now of the test sessions (their Z-FLoc frames,
`lamar.test_frames`), in `$DATA/test/epoch_000` (`SPLITS`, default `train test`; only these windows:
`qsub -v RTYPE=rt_HF,SPLITS=test shells/lamar/prepare_hf.sh`; the UnLoc val session: `SPLITS="train val"` and
`VAL_DATA=$DATA/val`). `train.py` takes the windows of
`<data>/epoch_<e>` at epoch e (in turn if there are fewer sets than epochs). When they exist, `train.sh` passes them to
`train.py --val_data`, which computes the losses, `acc` and `chance` on them after every epoch (the same
augmentations at every epoch, seeded by `--seed`) and logs them to wandb (`val/...`) and `<run>/val_log.csv`.

`REPO`, `LAMAR_ROOT`, `DATA`, `RUN`, `EPOCHS`, `WINDOWS`, `VAL_WINDOWS`, `NUM_GPUS`, `NUM_WORKERS`, `CACHE`, `SPLITS`, `VAL_DATA`, `RESUME`, `NEGATIVES`, `PLAN_GRID`, `PLAN_SIZE`, `CHECKPOINT`, `ZFLOC_NPZ` and `WANDB_PROJECT` are passed with
`qsub -v`, e.g. `qsub -v RTYPE=rt_HF,DATA=<data>,EPOCHS=4 shells/train/lamar_hf.sh`; `qsub -v` replaces the `#PBS -v RTYPE=...` of the script, so `RTYPE` (`rt_HF` for `_hf`, `rt_HG` for `_hg`) is always given with it (RESERVE service). The cached reconstructions of an
older version (`<session>_s<start>_L<length>.npz`, indexed by the frames of other filters), the per-window cache
`$DATA/recon_cache` and the samples directly in `$DATA` (`mapanything/`, `lidar/`, `lamar_*.csv`) of the previous
version are not used and can be deleted.

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
