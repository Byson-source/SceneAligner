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
- 学習データは upstream (Cornell-VAILab/SceneAligner) の `prepare_c3.py` を基本、GT 幾何から対応点を作る部分は `prepare_s3d.py` (fork では削除済み、`git show 65f91be:prepare_s3d.py`) に合わせる。独自設定は入れない (対応表は `train/lamar/README.md`)。
  - 窓: 長さ 15/20/35/50/100 × フレーム間隔 1/2/4 (C3 の写真集合に倣い広く)。名前・cache は `<session>_t<先頭 ts>_L<len>_k<step>`。学習フレームは UnLoc の室内区間 + plan 内のみ。
  - MapAnything sample: C3 と同じ conf 30/45/60/75/90 × xz 0/2.5/5 × gamma 4 種、フレームごと 8 点以上、RANSAC 一貫性 (10 px/512)。
  - LiDAR sample: S3D の GT depth sample と同じ (高さ 20〜95 %, xz 2.5 %, gamma 0.5, 10 点以上, 一貫性なし)。
  - 対応点: S3D と同じく壁上の点のみ。許容幅は hge.png の壁から 0.5 m (affine 誤差 0.25 m。0.2 m は遠い壁が消えた)。
  - 学習: S3D の `--epochs 20 --min_points 1000`、解像度は plan と同じ単一 1024。density map は学習・評価とも 512。
- floorplan (`hge.png` 3278×2334, 再配布不可) は `--plan_size 1024` に縮小して入力 (原寸は ViT の token 数で不可、512 は 1 m = 2.9 px で粗い)。
- upstream に無い追加は `--negatives batch --plan_grid 8` と plan の壁を 0.5 m に太らせる処理のみ (ユーザー承認済み)。negatives/plan_grid は、初回 run が feat が chance = ln 1024 に張り付いた: 窓の点が plan 上の数 patch に固まり、負例が同 sample 内だけだったため。`acc`・`chance` も記録。`<run>/state.pt` から epoch 単位で再開。
- 学習は wandb (`--wandb_project`)。`--vis_every` step ごとに floorplan 上の対応点 (GT 緑 / 予測 橙)・現在位置 (GT 青 / 推定 赤) と現在フレーム RGB を hstack して保存・ログ。

構成
- `train/lamar/`: `lamar.py` (CAPTURE 読込, affine, 壁判定, フィルタ, 窓, LiDAR), `reconstruction.py` (MapAnything), `prepare_lamar.py`, `evaluate_lamar.py`, `visualize_correspondences.py` (RGB・density・floorplan の対応点を 7 色で描画), `split.yaml`, `tests/`
- `shells/lamar/`: `download_lamar.sh`, `prepare.sh` (+`_hf`/`_hg`。GPU ごとに 1 プロセス `--shard`、最後に `--merge`。HF は全 GPU を使う), `evaluate.sh` (+`_hf`/`_hg`)。train/evaluate は 1 GPU なので HG で投げる。`shells/train/`: `train.sh`, `lamar_hf.sh`, `lamar_hg.sh`
- 既存の変更: `train.py` (wandb, 可視化), `scenealigner/training.py` (任意列 `rgb`/`camera_plan`/`camera_density`), `scenealigner/visualization.py`, `scenealigner/inference.py` (解像度を floorplan から取得。512 では不変)

状態
- 実装済み。クラウド環境では py_compile・`bash -n`・`pytest train/lamar/tests` のみ検証。GPU / LaMAR データ / MapAnything での実行は未確認。

未確認事項
- intrinsics はフレームごと (`cam_phone_<ts>` / `depth_phone_<ts>`)。depth は画像に registered (rig なし)。depth と画像の timestamp 一致率、K_depth / K の比 (`--check` に表示) は未確認。
- test `ios_2022-01-18_17.10.39_000` は index shift で 516 フレーム、Z-FLoc は 483 (`--zfloc_npz` で照合)。
- ABCI doc はクラウドから取得できず、PBS ヘッダは develop の `shells/test.sh` を雛形にした。
- benchmark 版 `HGE.zip` は `BENCH_URL` で取得できた (job 2741583)。`hge.png` は `FLOORPLAN_SRC` に無く未コピー。

次の手順 (ABCI。shell は REPO 既定 `~/oss/SceneAligner` なのでどこからでも qsub 可)
1. `qsub shells/lamar/download_lamar.sh`
2. `python train/lamar/lamar.py --check ...` で session 解決・z-up・kept 数 (410/397/483)・depth 一致率を確認
3. 1 窓の smoke (MapAnything の pose が GT と一致、対応点の相似フィット det>0)
4. ベースライン: `qsub -v CHECKPOINT=pre-trained shells/lamar/evaluate_hg.sh`、`CHECKPOINT=jhcho99/SceneAligner`
5. `qsub shells/lamar/prepare_hg.sh` → `qsub shells/train/lamar_hg.sh` → `qsub -v CHECKPOINT=<run> shells/lamar/evaluate_hg.sh`
