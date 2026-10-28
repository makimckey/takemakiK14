**[🇯🇵 日本語](#04-sleep-camで睡眠ステージを推定しよう)** | **[🇺🇸 English](#04-inferring-sleep-stages-with-sleep-cam)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**　**[次へ ▶](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

# 04: Sleep-CAMで睡眠ステージを推定しよう

学習時に保存した分割とモデルを使って評価します．
実際に作成された実験ディレクトリのパスを指定してください．

```bash
python -m src.test -i 実際の実験ディレクトリ
```

`config.yaml` と `cv_setting.json` を読み，学習時と同じ記録・被験者・窓設定を復元します．
Loaderの `load_training_testing_data` からそのfoldのtestを取得します．
train-wise正規化を選んだ実験では，学習時に保存した統計量も読みます．
データが追加されても評価対象を増やさず，使用記録が欠けていた場合はエラーで停止します．

## 評価結果

各foldのtest loss・Accuracyと，その平均を `results/metrics/` のJSONに保存します．
出力先は `-o` で変更できます．

```bash
python -m src.test -i 実際の実験ディレクトリ -o results/metrics/recheck.json
```

## 1記録の予測と可視化

```bash
python tools/visualize_hypnogram.py -f NPZのパス -c 実験ディレクトリ/config.yaml -m 実験ディレクトリ/weights/fold0_best.pt
```

設定と同じ窓・チャンネル・正規化で推論し，元の窓位置に予測を戻して描画します．
このツールの予測比較にはAccuracy・Macro-F1・Cohen’s κも表示されます．
性能比較には学習時のtest記録を使ってください．
train-wiseの場合は対応する `fold0_normalization.pt` をモデルと一緒に保持します．
モデルファイル名を変更した場合は `--normalization-stats` で統計量ファイルを明示できます．

旧CSV分割方式で作った実験結果を評価する場合は，その実験の旧コード・設定を使用します．
新しいCV割り当てと正規化で行う実験は，別の実験として学習し直して比較してください．

**[■ 目次](../README.md)**　**[◀ 前へ](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**　**[次へ ▶](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**　**[Next ▶](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

# 04: Inferring Sleep Stages with Sleep-CAM

Use the commands in the Japanese section above. Training and evaluation share the Sleep-Data-Hub cohort/CV APIs and the saved `cv_setting.json`.

**[■ Table of Contents](../README.md)**　**[◀ Previous](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**　**[Next ▶](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**
