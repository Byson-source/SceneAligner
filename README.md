# [NeurIPS'26] SceneAligner: 3D-Grounded Floorplan Localization in the Wild

### [Paper](https://arxiv.org/abs/2605.22581) | [Project Page](https://Cornell-VAILab.github.io/SceneAligner)

![teaser](./assets/teaser.png)
> Given a collection of in-the-wild images and a rasterized floorplan, SceneAligner reconstructs a gravity-aligned 3D point cloud from the images and globally aligns this reconstruction to the 2D floorplan map, thereby localizing the images within the floorplan. As illustrated above, our approach successfully aligns images capturing large-scale 3D environments, including exterior scenes (Doddabasappa Temple, left) and interior spaces (Église Saint-Martin d’Agonac, right).

## Installation

```bash
conda create -n scenealigner python=3.11
conda activate scenealigner
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Optional, to render images of the 3D viewer without a browser (`--render`):

```bash
pip install playwright && playwright install chromium
```

## Demo

The example is the Friedrichswerder Church in Berlin from the [C3 dataset](https://c3po-correspondence.github.io/):
20 interior and 40 exterior photos, which share one floorplan.

```bash
python demo.py --images examples/S221/interior examples/S221/exterior --floorplan examples/S221/floorplan.jpg --output outputs/S221
```

Each directory is a photo collection, reconstructed and aligned to the floorplan independently; collections without visual overlap thus end up in a common frame. Collections of more than 150 photos are split into several reconstructions. For each reconstruction, the demo saves the floorplan-aligned 3D scene (`.npz`), the density map and its alignment with the reliable correspondences, and it draws the cameras on the floorplan (`cameras.png`). It then starts the 3D viewer at http://localhost:8080. With `--share`, the viewer is also served at a public viser share URL, valid for 24 hours.

```bash
python -m scenealigner.viewer outputs/S221                                # interactive viewer
python -m scenealigner.viewer outputs/S221 --render first_frame.png       # image of the first-frame view
python -m scenealigner.viewer outputs/S221 --render demo.png --angle 270  # image after a 270° turn of the turntable
```

![demo](./assets/demo.png)
> View of the example after a 270° turn of the viewer's turntable.

## Dataset

We use a clean subset of [C3](https://c3po-correspondence.github.io/). Its test split is listed in `data/c3/test.csv` and its train split in `train.csv` of our [dataset repository](https://huggingface.co/datasets/jhcho99/SceneAligner):

| Column | Description |
|---|---|
| `uid` | Identifier of the photo-floorplan pair in C3 |
| `scene_id` | `S<scene>-<I or E>-F<floorplan>`: interior (I) or exterior (E) photos of a scene, paired with a floorplan |
| `chunk_idx` | Test only: reconstruction the photo belongs to (scenes of more than 150 photos are split) |
| `scene_name`, `plan_path`, `photo_path` | As in C3; paths are relative to `<C3>/visual/<scene_name>/` |
| `type` | `interior` or `exterior` |

Download [C3](https://huggingface.co/datasets/kwhuang/C3) (427 GB) and extract its archives (the evaluation only needs `geometric/geometric_test.tar.gz` and the `visual/` archives of the test scenes, 58 GB):

```bash
hf download kwhuang/C3 --repo-type dataset --local-dir <C3>
(cd <C3>/geometric && for f in *.tar.gz; do tar -xzf "$f"; done)
(cd <C3>/visual && for f in *.tar.gz; do tar -xzf "$f"; done)
```

The dataset repository (52 GB) also holds our cached test data: the π³ reconstructions of the test photos (with the scene gravity from GeoCalib), their density maps, and the GeoCalib gravity of each photo:

```bash
hf download jhcho99/SceneAligner --repo-type dataset --local-dir <cache>
```

## Evaluation

On our cached reconstructions, which reproduces Tables 1 and 2 of the paper up to small numerical differences between GPUs:

```bash
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions                               # Ours
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions --checkpoint_dir pre-trained  # pretrained DINOv3
```

End to end from the photos, with the public π³ weights of MegaDepth-X; the numbers may differ slightly:

```bash
python evaluate_c3.py --c3_root <C3>
```

## Training

The training data are density maps of π³ and VGGT reconstructions of the C3 train split, of photo collections and of single photos, with correspondences derived from the C3 annotations. VGGT is only needed for the second command:

```bash
pip install einops && pip install --no-deps git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce
```

```bash
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data>                 # π³, photo collections
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --model vggt    # VGGT, photo collections
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --single_view   # π³, single photos
python train.py --data <data> --output <run>
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions --checkpoint_dir <run>
```

Each `prepare_c3.py` command goes through five confidence thresholds, which `--conf` can split over several runs.

## Structured3D

Single-view localization on [Structured3D](https://structured3d-dataset.org/) (Table 3) uses its perspective images (`full`) and annotations of the test scenes (`scene_03250` to `scene_03499`), extracted to `data/Structured3D/scene_*`. Structured3D is a synthetic indoor dataset, so its views are reconstructed with the original π³ weights rather than those fine-tuned on Internet photos by MegaDepth-X. The floorplans are drawn from `annotation_3d.json` at 1 cm per pixel: walls in black, rooms in white, doors and windows closed.

```bash
python evaluate_s3d.py
python evaluate_s3d.py --scenes 3200-3220 --output outputs/s3d_results_3200-3220.json   # other scenes
```

`--scenes FIRST-LAST` evaluates an inclusive range of scenes instead of the test scenes; scenes missing from `data/Structured3D` are skipped. The floorplans and the π³ reconstructions of the views are cached in `outputs/s3d_reconstructions` (`--recon_cache`): later runs skip the floorplan drawing, π³ and GeoCalib.
Per view, `outputs/s3d_views/<scene>/<room>_<position>.png` (`--image_dir`) shows the RGB image next to the floorplan with the aligned density map and the estimated (green) and ground-truth (blue) cameras, each a dot with a 1 m viewing-direction arrow.

The released weights were trained for 20 epochs instead of 10, as we observed that training longer leads to a higher accuracy than reported in Table 3:

| Recall (%) | Pos@0.1m | Pos@0.5m | Pos@1.0m | Ang&Pos@(30,1m) |
|---|---|---|---|---|
| Table 3 | 3.5 | 37.5 | 53.8 | 51.6 |
| Released weights | 4.7 | 44.3 | 59.7 | 58.0 |


## LaMAR HGE

Training and evaluation on the phone sequences of the HGE building of LaMAR, with MapAnything reconstructions of
windows of consecutive frames (ground-truth intrinsics and relative poses as input), the split of UnLoc and the
evaluation protocol of Z-FLoc: see [train/lamar/README.md](train/lamar/README.md).

## Release
- [x] Demo, inference and evaluation on C3, with the model checkpoint ([jhcho99/SceneAligner](https://huggingface.co/jhcho99/SceneAligner)), the dataset and the cached test data
- [x] Training on C3 and training data generation
- [x] Structured3D

## License
The code is released under the [MIT License](LICENSE). The models and data it uses keep their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md); the π³ weights are for non-commercial use only.

## Citation
If you find our work useful for your research, please cite our paper:

````BibTeX
@inproceedings{cho2026scenealigner,
  title={SceneAligner: 3D-Grounded Floorplan Localization in the Wild},
  author={Junhyeong Cho and Ruojin Cai and Hadar Averbuch-Elor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
````