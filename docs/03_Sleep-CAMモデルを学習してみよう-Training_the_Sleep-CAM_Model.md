**[🇯🇵 日本語](#03-sleep-camモデルを学習してみよう)** | **[🇺🇸 English](#03-training-the-sleep-cam-model)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**　**[次へ ▶](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**

# 03: Sleep-CAMモデルを学習してみよう

本テンプレートの `src/models/sleep_cam.py` は，生体信号を入力する1次元CNNです．
Loaderの `(窓数, チャンネル, サンプル数)` を授業用アダプターがモデルの `(B,1,L,C)` に変換します．

## 学習を実行する

```bash
python -m src.train -c configs/SleepCAM.yaml
```

CPUで小さく動作確認する場合は，設定の `training.max_epochs: 1`，`dataset.limit` を小さめにし，次を実行します．
選択した被験者数が `dataset.cv.n_folds` 以上になるようにしてください．

```bash
python -m src.train -c configs/SleepCAM.yaml -a cpu --num-workers 0
```

チャンネル・窓・正規化・被験者CVは [第2章](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)の設定に従います．
5-foldでは各回にtestを1fold，validationを次の1fold，trainを残り3foldとします．
validationでチェックポイント選択と早期終了を行い，そのfoldのtestで最終評価します．

## 保存されるもの

`results/experiments/<実験名>@<実行日時>/` に以下を保存します．

- `config.yaml`：実行した設定．
- `cv_setting.json`：使用記録と被験者のfold割り当て．
- `weights/foldN_best.pt`，`foldN_last.pt`：モデル・オプティマイザーの状態．
- `weights/foldN_normalization.pt`：train-wise正規化の統計量（その方式のとき）．
- `histories/`：学習・validationの損失とAccuracyのCSV・グラフ．
- `training.log`：実行ログ．

モデルや学習率などを変更するときも，実験ごとの設定と保存分割を残して比較してください．
CSV分割方式からの更新は [移行手順](./07_Loader_APIへの移行.md)を参照してください．

**[■ 目次](../README.md)**　**[◀ 前へ](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**　**[次へ ▶](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**　**[Next ▶](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**

# 03: Training the Sleep-CAM Model

Use the commands in the Japanese section above. Training and evaluation share the Sleep-Data-Hub cohort/CV APIs and the saved `cv_setting.json`.

**[■ Table of Contents](../README.md)**　**[◀ Previous](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**　**[Next ▶](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**
