# 導入済みリポジトリをLoaderのコホート・CV APIへ移行する

2026-10-16

この手順は，テンプレートから既に作成した各自のリポジトリへ変更を反映するためのものです．
テンプレートの更新は，そこから作成済みのリポジトリへ自動では反映されません．
更新対象のファイルを比較し，自分の実装と統合してください．

## 1. 更新するファイル

10月14日の標準形式対応版を導入済みなら，今回のコード更新は以下です．

| ファイル | 反映する内容 |
| --- | --- |
| `src/datasets/loader/cohort_data.py` | **追加必須**．コホート読み込み，LoaderのCV，validation追加，正規化，分割保存・復元，単一記録予測． |
| `src/train.py` | **更新必須**．CSV読込をコホート・CV APIに置き換え，`cv_setting.json` を保存する． |
| `src/test.py` | **更新必須**．学習時の保存分割と正規化統計量を読み戻す． |
| 使用中の `configs/*.yaml` | **設定移行必須**．下記の新しい `dataset` に置き換える．実験名・モデル・学習率などの独自設定は各自の内容を統合する． |
| `tools/visualize_hypnogram.py` | **予測可視化を使う場合は更新必須**．新しい窓・チャンネル・正規化設定に対応する． |
| `tools/split_dataset.py` | 旧コマンドを実行したときに，新方式への移行案内を表示する版へ更新する． |

以前の版から更新する場合は，`src/datasets/loader/` に
`sleep_loader.py`，`cohort_mix_rules.toml`，`standard_sleep_data.py` と各依存ファイルも必要です．
Loader本体とルールはSleep-Data-Hubのcommit `15c29aca09d5d6dd387e4d115b249409ef9e253f` の配布ファイルです．
10月14日の対応版からはこの2ファイルの内容に変更はありません．
Loaderフォルダを反映する際は，各自で追加したモデルやtransformを残して差分を統合してください．

説明・動作確認用には，次も反映してください．

- `README.md`，`docs/02_*`，`docs/03_*`，`docs/04_*`，この移行手順．
- `src/datasets/loader/README.md`．
- `tests/test_cohort_data.py` と更新版の `tests/test_standard_sleep_data.py`．

今回のCV移行では，第9・10回の実習コードや `run.sh`・`setup.sh` の上書きは不要です．
コンテナ内から `/data/ise_data/physionet2018.sqlite` が見えることを確認してください．

## 2. YAMLのdatasetを置き換える

以前の `dataset.paths`，`total_folds`，`config`，`transforms` は削除し，以下を設定します．
独自に使っているすべての学習用YAMLへ反映してください．
テンプレートでは `configs/SleepCAM.yaml` と `configs/example_change_activations/*.yaml` を更新済みです．

```yaml
dataset:
  cohort_dir: /data/ise_data
  cohort_id: physionet2018
  limit: 20
  cv:
    n_folds: 5
    seed: 42
    shuffle_subjects: true
  channel_set: [EEG_C3-R, EEG_C4-R, EOG_LOC-R, EMG_L-EMG_R]
  window_sec: 30.0
  stride_sec: 30.0
  y_spec: stages(latest)
  normalization: record-wise
  exclude_labels: [-1]
```

| 旧設定・操作 | 新設定・操作 |
| --- | --- |
| CSVを作り，`dataset.paths` から読む | `cohort_dir` と `cohort_id` を指定し，SQLite検索後に `load_cohort` で読む． |
| `dataset.total_folds` | `dataset.cv.n_folds`．validationも確保するため3以上． |
| `dataset.config.channels` / チャンネル選択transform | `dataset.channel_set`．モデルの `input_channels` と本数を揃える． |
| `seq_length: 1`, `target_idx: -1` | `window_sec: 30`, `stride_sec: 30`, `y_spec: stages(latest)`． |
| 例：`seq_length: 2`, `target_idx: -1` | 60秒窓・30秒ストライド・`stages(latest)`．Unknownの除外は窓の正解ラベルに基づくため，旧連続エポック処理とは条件が異なる． |
| `ExcludeLabels([-1])` | `exclude_labels: [-1]`． |
| `ZScoreNormalize` | `normalization` を指定．record-wiseでは記録・チャンネルごとに正規化する． |
| `--max-subjects` | `dataset.limit` で**記録数**を指定．被験者数の上限ではない． |

Loaderはコホートの信号と窓テンソルをメモリに展開します．
まず20記録で動かし，メモリ容量に合わせて `limit` を増やしてください．全件なら `null` とします．
同一被験者に複数記録がある場合も，同じfoldへ割り当てます．

活性化関数の比較用設定を独自に作成した場合は，`training.loss` も確認してください．

```yaml
training:
  # 各自のlr・batch_size・max_epochs等を維持する
  loss:
    name: cross_entropy
    args: null
```

## 3. 学習と評価

テンプレートと同じ依存ライブラリの環境で，リポジトリのルートから実行します．

```bash
python -m src.train -c configs/SleepCAM.yaml
```

Loaderの `build_cv_setting()` が被験者にfold番号を割り当て，
`load_training_testing_data()` が指定foldのtestと残りのtrain候補を返します．
授業用アダプターはtrain候補から次のfoldをvalidationに取り分けます．
testが `k`，validationが `(k + 1) % n_folds`，trainが残りです．

新しく作成した実験ディレクトリを指定して評価します．

```bash
python -m src.test -i 実際の実験ディレクトリ
```

実験の `config.yaml` と `cv_setting.json` を組にして保存してください．
後者は実際に使った記録と被験者のfold割り当てを含みます．
評価時はこの記録集合を復元し，データ追加による分割の変化を防ぎます．
使用記録が消えた場合や，被験者との対応が変わった場合は停止します．
実験結果と同様に `results/` 配下で扱い，Gitへ追加する必要はありません．

train-wise正規化ではvalidation/testを除いたtrainだけから統計量を計算します．
`weights/foldN_normalization.pt` も保存し，評価・単一記録の予測で再利用します．

## 4. 既存CSV・旧実験・独自コード

新しい学習・評価は `configs/npz_lists/*.csv` を参照しません．
既存CSVを残していても新しい実験には影響しません．生成し直す操作も不要です．

旧CSV方式で作成した実験は，新しい `cv_setting.json` を持ちません．
新しい分割を後付けして旧モデルを評価すると，学習・評価の被験者が重なるおそれがあります．
旧実験を再現するときは，その実験に対応する旧コード・CSV・設定・重みを使ってください．
新方式は新しい実験として学習し直します．fold割り当てと正規化の単位も変わるため，
同じfold番号が旧実験と同じ被験者を意味するわけではありません．

独自コードで `dataset.paths`，`dataset.total_folds`，`dataset.config`，`dataset.transforms`，
`create_transform_from` を参照していた場合は，新設定と `cohort_data.py` の関数に合わせて変更してください．
モデル側の `(B,N,T,C)` インターフェースは授業用アダプターが維持します．

## 5. 更新後の確認

```bash
python -m pytest tests -q
```

合成データで，信号名・ラベルの対応，教師あり記録のSQL選択，複数夜の被験者分割，
保存分割の再現，正規化へのvalidation/test混入防止，単一記録予測との前処理一致を確認します．
実データを一括で学習する前に，少ない記録数・`training.max_epochs: 1`・CPUでも動作確認できます．
