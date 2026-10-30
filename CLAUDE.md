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
  - 再構成: session の kept frame を連続 ≤500 枚の chunk ごとに MapAnything で 1 回だけ推論し、各フレームの点を自カメラ座標 (float16) で `$DATA/session_cache/<session>/` に cache。窓は cache の点を GT pose で窓先頭基準に並べて作る (`reconstruction.window_from_cache`、chunk 跨ぎ可)。
  - 学習窓 (ユーザー指定): epoch ごとに別の窓 (`$DATA/epoch_<e>`、既定 20 epoch)、1 epoch 8864 窓 (LiDAR と合わせ 17728 sample = 554 step/epoch、ユーザー指定で旧 1108 step の半分)、val は 600 窓。全 epoch を通して同じフレームの組み合わせは使わない。session は kept frame 数に比例して選び、n=5〜300 枚 (ユーザー指定) を n〜8n 枚の連続区間から飛び飛びにランダム抽出 (5 s 超の欠落を跨がない)。半分の窓は 1/4〜1/2 のフレームを 2 m 以内を通る別 session (1〜2 本) のフレームに置換。各フレームの画素 (と LiDAR 点) を 25〜100 % のランダムな割合で残す (density の percentile は残した点で)。名前は `<session>_t<先頭 ts>_n<枚数>_<crc32>`、フレームは `<epoch>/meta/windows.csv` (`<session>:<ts>`)。評価窓は Z-FLoc のまま。学習フレームは UnLoc の室内区間 + plan 内のみ。
  - MapAnything sample: 1 窓 1 枚。C3 の conf 30/45/60/75/90 × xz 0/2.5/5 × gamma 4 種からランダム (一貫性を通るまで conf×xz を乱順に試す)、フレームごと 8 点以上、RANSAC 一貫性 (10 px/512)。
  - LiDAR sample: S3D の GT depth sample と同じ (高さ 20〜95 %, xz 2.5 %, gamma 0.5, 10 点以上, 一貫性なし)。depth の解像度 (256×192) のまま unproject (`lamar.lidar_points`、再構成 grid に拡大しない)、対応点は `--max_points` 20000 に間引く (間引き前は 1 窓 24 MB でディスクが溢れた)。
  - 対応点: S3D と同じく plan 座標・壁判定は同じ画素の LiDAR 点 (`--plan_source lidar` 既定、prepare.sh `PLAN_SOURCE`。recon は MapAnything 点で density と plan が自明に一致し誤った壁点が入るため変更)。conf の percentile は mask 内の画素のみ。affine の 4 点残差は 0.24〜0.30 m (壁許容 0.2 m より大、`--check` に表示)。壁上の点のみ。許容幅は hge.png の壁から 0.2 m (0.5 m は緩すぎるとしてユーザー判断で戻した。遠い壁の点は減る)。
  - 学習: S3D の `--epochs 20 --min_points 1000`、解像度は plan と同じ単一 1024。density map は学習・評価とも 512。
- floorplan (`hge.png` 3278×2334, 再配布不可) は `--plan_size 1024` に縮小して入力 (原寸は ViT の token 数で不可、512 は 1 m = 2.9 px で粗い)。
- upstream に無い追加は `--negatives batch --plan_grid 8 --false_negative_px 8 --false_negatives soft` と plan の壁を 0.5 m に太らせる処理のみ (ユーザー承認済み)。false_negative_px は偽負例対策: plan 上 (augmentation 前の `plan_ref`) で 8 px (1.4 m) 以内の batch 内の点を負例から外す (同じ窓の MapAnything/LiDAR、同じ場所の別窓、隣接 cell。`train.near_points`、train.sh `FALSE_NEGATIVE_PX`)。ユーザー判断で無視でなく soft positive (重み exp(-d²/2σ²)、σ = 4 px、自分は 1、正規化した重みへの InfoNCE) を既定に (`FALSE_NEGATIVES=soft`、`ignore` も可)。negatives/plan_grid は、初回 run が feat が chance = ln 1024 に張り付いた: 窓の点が plan 上の数 patch に固まり、負例が同 sample 内だけだったため。`acc`・`chance` も記録。run は job ごとに `outputs/lamar_<data>_<RTYPE>_<日付_時刻>` に新規作成 (上書き防止)。再開は `qsub -v RTYPE=rt_HG,RESUME=<run>` (`train.py --resume`、`<run>/state.pt` から epoch 単位、wandb run も継続)。`--resume` 無しで state.pt がある output はエラー。
- fine-tune: `--init <LoRA>` (例 `jhcho99/SceneAligner`) で学習済み LoRA から開始、`--lr` 可変。train.sh は `INIT`/`LR` (既定 空/1e-4)。初回 run (DINOv3+新規 LoRA) は train acc 高・val acc ≈ chance (0.2〜2 %) で 1 building への暗記と判断し、ユーザー承認で追加: `qsub -v RTYPE=rt_HF,INIT=jhcho99/SceneAligner,LR=1e-5,EPOCHS=5 shells/train/lamar_hf.sh`。
- 学習は wandb (`--wandb_project`)。`--vis_every` step ごとに floorplan 上の対応点 (GT 緑 / 予測 橙)・現在位置 (GT 青 / 推定 赤) と現在フレーム RGB を hstack して保存・ログ。

構成
- `train/lamar/`: `lamar.py` (CAPTURE 読込, affine, 壁判定, フィルタ, 窓, LiDAR), `reconstruction.py` (MapAnything), `prepare_lamar.py`, `evaluate_lamar.py`, `evaluate_zfloc.py` + `zfloc_solver.py` / `ransac.py`, `visualize_correspondences.py` (RGB・density・floorplan の対応点を 7 色で描画), `split.yaml`, `tests/`
- `shells/lamar/`: `download_lamar.sh`, `prepare.sh` (+`_hf`/`_hg`。HF は全 GPU を使う), `evaluate.sh` (+`_hf`/`_hg`。GPU ごとに `--shard` で分割し `--merge`。seed は窓 index で決まり分割に依らない)。train は torchrun + DDP でノードの全 GPU (rt_HF 8 枚、GPU あたり batch 4 = 計 32、lr 1e-4 据え置き、feat の負例は全 GPU の点を all_gather)。prepare→train の一括投入は login node で `bash shells/lamar/prepare_train_hf.sh` (recon 1 job → samples 4 job (`STAGE`/`PART`/`NUM_PARTS`、`--part`/`--num_parts` で epoch e を part e % N が作る、データは 1 job と同一) → train 4 node を `-W depend=afterok` で連結、`TRAIN_EPOCHS` 可。`TRAIN_JOBS` 本の train job を `afterany` で RESUME 連結し walltime を跨ぐ、`PREPARE=0 RESUME=<run>` で既存 run の延長)。再開時の loss ramp (`w_regr`/`w_struct`、total_steps 比例) は state.pt の `total_steps` (無ければ最後の step = 完走 run の総 step) で固定し、EPOCHS を増やして延長しても ramp が戻らない。state.pt は一時ファイル経由で保存。再開時の lr は `--lr` (LR / `TRAIN_LR`) で上書き (optimizer の state は継続、ユーザー指定で既存 run を 3e-4 で延長)。複数 node は `qsub -l select=4 -v RTYPE=rt_HF shells/train/lamar_hf.sh` (ユーザー指定で batch 128、lr 据え置き。node ごとに mpirun (hpcx) で torchrun、run 名に `x4`。ABCI での動作は未確認)。prepare は `--stage recon` (`multiprocessing.Pool`、GPU ごとに 1 worker)、次に `--stage samples` (Pool、CPU `NUM_WORKERS` (既定 min(CPU 数, 空きメモリ/8 GB)。`nproc` は `OMP_NUM_THREADS=1` で 1 を返すので外して数える) worker、`EPOCHS`/`WINDOWS`)。各 stage は全体で進捗バー 1 本、log ファイルでなく job 出力に即表示。窓の乱数は窓 index で決まり worker 数に依らない。窓の組み立ては float32・平坦 index・memmap はフレーム単位コピーで高速化 (合成 150 フレーム窓で 37 s → 15 s、`projection_geometry` は upstream `reconstruct_projection_geometry` と一致をテスト)。epoch の窓が揃うと CSV を書き、再実行時は済んだ epoch を飛ばす (EPOCHS/WINDOWS 変更時は `epoch_*` を手で消す)。データ (`DATA`) の既定は `/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar`。`SPLITS` (既定 `train test`。ユーザー指定でひとまず test split を validation に使う (val が短すぎて確認できないため)。窓の作り方・augmentation は train と同じ (室内区間 + plan 内、UnLoc も test に同じフィルタ) だが、ユーザー指定で test だけフレーム間隔 1 固定 (`--max_spread 1`、連続 n 枚)) で validation の窓も `$DATA/test/epoch_000` (1 set、固定) に作り (UnLoc val は `SPLITS="train val"`・`VAL_DATA=$DATA/val`)、train.sh はあれば `--val_data` を渡して epoch ごとに val の loss/acc/chance を wandb `val/`・`<run>/val_log.csv` に記録 (augmentation は `--seed` で固定)。`shells/train/`: `train.sh`, `lamar_hf.sh`, `lamar_hg.sh`
- 既存の変更: `train.py` (wandb, 可視化), `scenealigner/training.py` (任意列 `rgb`/`camera_plan`/`camera_density`), `scenealigner/visualization.py`, `scenealigner/inference.py` (解像度を floorplan から取得。512 では不変、`points_conf=None` で conf filter なし)
- Z-FLoc 入力での評価 (ユーザー指定): `train/lamar/evaluate_zfloc.py` は Z-FLoc の `generate_pcd.py` 点群 (`all_pcds`、窓の最後のフレームのカメラ座標、conf 無し → conf filter なし)・窓・`poses_map.npy` (create_lamar_hge.py) を使い Z-FLoc と同じメトリクス。重力は GT (ユーザー判断)。RANSAC はユーザー承認で Z-FLoc の scale band (期待スケール × (1 ∓ 0.35)) 外の 2 点仮説を棄却、inlier refit のスケールもクリップ (`train/lamar/ransac.py`。`--scale_tol -1` で従来の cv2、`evaluate_lamar.py` は既定で従来通り)。`shells/lamar/zfloc_eval.sh` (+`_hg` のみ、ユーザー判断で GPU 1 枚で十分、逐次) が create_lamar_hge → generate_pcd (Z-FLoc env `ZFLOC_ENV`、`ZFLOC_REPO`) → 評価まで実行。checkpoint の既定は `outputs/scratch` (ユーザー指定、`evaluate.sh` も同じ)。進捗は job 出力 (段階ごとの時刻、L ごとの点群数、窓ごとに SR・中央値誤差・その L の途中 SR・経過/ETA) と `$PCD_ROOT/hge_<L>.log`・`$OUT/test_results.log`。
  - 後段 Z-FLoc (ユーザー指定、`--solver zfloc` / `SOLVER=zfloc` が既定、`--solver ransac` で上記 RANSAC): 対応点だけ SceneAligner、N 対応の全 nC2 ペアを similarity 仮説にし Z-FLoc 2C solver の後段 (scale band → BEV 壁線の粗スコアで上位 1000 → alignment/penetration + scale penalty → 最良を LM) で選ぶ (`train/lamar/zfloc_solver.py`、Z-FLoc は `--zfloc_repo` から import のみで無変更)。壁 BEV は Z-FLoc `extract_primitives.py` (`zfloc_eval.sh` の 3/4 段)。gravity はユーザー判断で Z-FLoc 推定 (`gravity/<stem>.npy`) に揃え、density map もそれで作る (density px → BEV px が厳密)。誤差は `zfloc/offline.py` と同じ `last_frame_pose_estimation`。scenealigner env に numba・scipy・pandas が必要。

状態
- 実装済み。クラウド環境では py_compile・`bash -n`・`pytest train/lamar/tests` のみ検証。GPU / LaMAR データ / MapAnything での実行は未確認 (DDP 学習も GPU 上では未確認。loss の分散版は CPU gloo のテストで単一 GPU 版との一致のみ確認)。

未確認事項
- intrinsics はフレームごと (`cam_phone_<ts>` / `depth_phone_<ts>`)。depth は画像に registered (rig なし)。depth と画像の timestamp 一致率、K_depth / K の比 (`--check` に表示) は未確認。
- test `ios_2022-01-18_17.10.39_000` は index shift で 516 フレーム、Z-FLoc は 483 (`--zfloc_npz` で照合)。`zfloc_eval.sh` は再生成した点群・poses_map で一貫させる。`gt_camera_poses` と trajectories.txt の窓内相対姿勢一致は assert するが実データで未確認。
- `--solver zfloc` は合成データのテスト (density px → BEV px が Z-FLoc の投影と一致、外れ値 75 % で similarity 回復) のみ。実 BEV での対応点の位置・nC2 の時間は未確認。primitive の無い窓は失敗として分母に入れる (Z-FLoc `offline.py` は BEV の無い窓を数えない)。
- ABCI doc はクラウドから取得できず、PBS ヘッダは develop の `shells/test.sh` を雛形にした。
- benchmark 版 `HGE.zip` は `BENCH_URL` で取得できた (job 2741583)。`hge.png` は `FLOORPLAN_SRC` に無く未コピー。

次の手順 (ABCI。shell は REPO 既定 `~/oss/SceneAligner` なのでどこからでも qsub 可)
1. `qsub shells/lamar/download_lamar.sh`
2. `python train/lamar/lamar.py --check ...` で session 解決・z-up・kept 数 (410/397/483)・depth 一致率を確認
3. 1 窓の smoke (MapAnything の pose が GT と一致、対応点の相似フィット det>0)
4. ベースライン: `qsub -v RTYPE=rt_HG,CHECKPOINT=pre-trained shells/lamar/evaluate_hg.sh`、`CHECKPOINT=jhcho99/SceneAligner`
5. `qsub shells/lamar/prepare_hg.sh` → `qsub shells/train/lamar_hg.sh` → `qsub -v RTYPE=rt_HG,CHECKPOINT=<run> shells/lamar/evaluate_hg.sh`
