# CLAUDE.md

## Important rule
1. Please reply in Japanese as concise as possible
2. Always commit to branch called "develop"
3. Be willing to create appropriate folder structure and helper functions to foster readability. Code as concise as possible
4. always activate 'scenealigner' conda environment by doing `export PATH=${HOME}/miniconda3/bin:$PATH && source activate scenealigner`
5. for every shell script, please refer to abci doc and you need to write for example:
   ```bash
   #!/bin/bash
   #PBS -P gai51740
   #PBS -q R9920261300
   #PBS -l select=1
   #PBS -l walltime=04:00:00
   #PBS -v RTYPE=rt_HG
   #PBS -m abe
   #PBS -k oed
   ```
6. GPU を使う shell（`shells/train/`）は必ず rt_HF 版 `<name>_hf.sh` と rt_HG 版 `<name>_hg.sh` の 2 つを作る。
   中身は `shells/train/train.sh` に `DATA` と `RTYPE` を渡すだけの wrapper にする。

## Current task: SceneAligner を LaMAR HGE で学習・評価
目的: unloc / Z-FLoc と同じ LaMAR HGE の split・座標系・評価窓で SceneAligner を学習し、比較する。詳細は `train/lamar/README.md`。

確定事項
- 再構成は MapAnything に GT intrinsics と GT 相対姿勢 (先頭フレーム基準, metric) を入力。gravity は GT pose (world z-up) から。GeoCalib は使わない。
- pose・画像は benchmark 版 `HGE/sessions/map` (unloc と同じ)、LiDAR depth は raw セッション `ios_<date>_<time>` から timestamp で対応付け。
- split は unloc (`train/lamar/split.yaml`: train 12 / val 1 / test 3)。評価は Z-FLoc (非重複 L 窓 15/20/35/50/100、最終 10 フレーム全て <1 m で成功、seed 42〜46、0.055 m/px)。
- 学習窓は連続フレーム、長さ 15〜100。sample は MapAnything density + LiDAR density。対応点は再構成点を GT pose + 4 点 affine で floorplan に写し、壁上の点だけ使う。
- floorplan (`hge.png` 3278×2334, 再配布不可) は `--plan_size 1024` に縮小して入力 (原寸は ViT の token 数で不可、512 は 1 m = 2.9 px で粗い)。
- 学習は wandb (`--wandb_project`)。`--vis_every` step ごとに floorplan 上の対応点 (GT 緑 / 予測 橙)・現在位置 (GT 青 / 推定 赤) と現在フレーム RGB を hstack して保存・ログ。

構成
- `train/lamar/`: `lamar.py` (CAPTURE 読込, affine, 壁判定, フィルタ, 窓, LiDAR), `reconstruction.py` (MapAnything), `prepare_lamar.py`, `evaluate_lamar.py`, `split.yaml`, `tests/`
- `shells/lamar/`: `download_lamar.sh`, `prepare.sh` (+`_hf`/`_hg`), `evaluate.sh` (+`_hf`/`_hg`)。`shells/train/`: `train.sh`, `lamar_hf.sh`, `lamar_hg.sh`
- 既存の変更: `train.py` (wandb, 可視化), `scenealigner/training.py` (任意列 `rgb`/`camera_plan`/`camera_density`), `scenealigner/visualization.py`, `scenealigner/inference.py` (解像度を floorplan から取得。512 では不変)

状態
- 実装済み。クラウド環境では py_compile・`bash -n`・`pytest train/lamar/tests` のみ検証。GPU / LaMAR データ / MapAnything での実行は未確認。

未確認事項
- raw セッションの depth sensor id と `rigs.txt` の変換、depth と画像の timestamp 一致率。
- test `ios_2022-01-18_17.10.39_000` は index shift で 516 フレーム、Z-FLoc は 483 (`--zfloc_npz` で照合)。
- ABCI doc はクラウドから取得できず、PBS ヘッダは develop の `shells/test.sh` を雛形にした。
- benchmark 版 `HGE.zip` は `BENCH_URL` で取得できた (job 2741583)。`hge.png` は `FLOORPLAN_SRC` に無く未コピー。

次の手順 (ABCI。shell は REPO 既定 `~/oss/SceneAligner` なのでどこからでも qsub 可)
1. `qsub shells/lamar/download_lamar.sh`
2. `python train/lamar/lamar.py --check ...` で session 解決・z-up・kept 数 (410/397/483)・depth 一致率を確認
3. 1 窓の smoke (MapAnything の pose が GT と一致、対応点の相似フィット det>0)
4. ベースライン: `qsub -v CHECKPOINT=pre-trained shells/lamar/evaluate_hg.sh`、`CHECKPOINT=jhcho99/SceneAligner`
5. `qsub shells/lamar/prepare_hg.sh` → `qsub shells/train/lamar_hg.sh` → `qsub -v CHECKPOINT=<run> shells/lamar/evaluate_hg.sh`
