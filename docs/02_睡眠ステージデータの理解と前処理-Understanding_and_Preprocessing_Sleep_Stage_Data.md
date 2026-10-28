**[🇯🇵 日本語](#02-睡眠ステージデータの理解と前処理)** | **[🇺🇸 English](#02-understanding-and-preprocessing-sleep-stage-data)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./01_環境を整えよう-Getting_the_Environment_Ready.md)**　**[次へ ▶](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**

# 02: 睡眠ステージデータの理解と前処理

## データの基本

- 配布場所：`/data/ise_data`（コンテナ内も同じパス，読み取り専用）
- コホート：PhysioNet 2018，睡眠ステージ判定ルール：AASM（30秒エポック）
- ファイル：各記録の `r-*.npz` とコホートの `physionet2018.sqlite`
- 授業用チャンネル：`EEG_C3-R`, `EEG_C4-R`, `EOG_LOC-R`, `EMG_L-EMG_R`
- サンプリング周波数：200 Hz；1エポックは6,000点
- ラベル：`0=Wake, 1=REM, 2=N1, 3=N2, 4=N3, -1=Unknown`

## 読み込み

標準形式のNPZには `signal_concat`，`signal_offsets`，`signal_info`，
`signal_sampling_interval_ms` などが入っています．信号はエポック化して保存されていません．
テンプレートに同梱したSleep-Data-Hub Loaderを授業用の関数から呼び出します．

```python
from src.datasets.loader.standard_sleep_data import find_example_file, load_sleep_record

path = find_example_file("/data/ise_data")  # 正解ステージのある記録を選ぶ
record = load_sleep_record(path)
X = record.signals                      # (エポック数, 6000, 4)
y = record.stages                       # (エポック数,)
print(record.channel_names)
print(record.sampling_rate)
print(record.metadata.signal_info)     # 信号名とフィルタ情報
```

`channels=["EEG_C3-R", "EOG_LOC-R"]` のように信号名で選択できます．
選択した信号間のサンプリング間隔が異なる場合はエラーになります．
EDF変換・リサンプリング・30秒窓の切り出しを学生が読み込み部分に実装する必要はありません．
`load_sleep_record` はリサンプリングを行わず，記録の実際の周波数を返します．

標準Loader単体では，次のように読み込めます．

```python
from src.datasets.loader.sleep_loader import load_npz
loaded = load_npz(path, purpose="supervised_stage", allow_pickle=False)
try:
    print(loaded.dfs_by_dt_ms.keys())
    print(loaded.sampling_interval_ms)
    print(loaded.stages.shape)
finally:
    loaded.release()
```

Loaderはサンプリング間隔別にDataFrameを返します．
授業用関数はその結果をLoaderの窓分割APIに渡し，既存モデルに合う `(N,T,C)` に変換します．

## 前処理と分割

配布信号は標準化時にフィルタ処理されています．`signal_info` を確認し，
実習では追加処理前後の波形・スペクトラムを比較します．
200 Hzでフィルタをかける場合，上限を100 Hz未満にしてください．

学習用データは，Sleep-Data-Hub Loaderのコホート・CV APIで作ります．
`cohort_dir` が `ise_data` という名前なので，`cohort_id="physionet2018"` も明示します．
SQLiteの `usage` で教師あり睡眠ステージ判定に使える記録を選びます．

```python
from src.datasets.loader.sleep_loader import (
    load_cohort, CVConfig, build_cv_setting, load_training_testing_data,
)

cohort = load_cohort(
    "/data/ise_data",
    cohort_id="physionet2018",
    purpose="supervised_stage",
    where="r.usage LIKE ?",
    params=['%"supervised_stage"%'],
    limit=20,
    allow_pickle=False,
)
try:
    cv = build_cv_setting(
        cohort, CVConfig(n_folds=5, seed=42, shuffle_subjects=True),
    )
    bundle = load_training_testing_data(
        cohort, cv, fold_idx=0,
        channel_set=["EEG_C3-R", "EEG_C4-R", "EOG_LOC-R", "EMG_L-EMG_R"],
        window_sec=30.0, stride_sec=30.0,
        y_spec="stages(latest)", normalization="record-wise",
    )
    print(bundle.train.x.shape, bundle.train.y.shape)
    print(bundle.test.x.shape, bundle.test.y.shape)
finally:
    cohort.release()
```

`build_cv_setting` が被験者ごとにfoldを割り当てます．
上のAPI単体では，指定したfoldがtest，それ以外がtrainです．
`bundle.train` と `bundle.test` には信号・ラベルのほか，
被験者・記録・元の窓位置をたどる索引も入っています．
信号の軸は `(窓数, チャンネル, サンプル数)` です．

このLoaderはコホートの信号と窓テンソルをメモリに展開します．
授業用の配布設定は正解付き20記録（`limit: 20`）です．
`limit` は被験者数ではなく記録数なので，複数夜ある被験者を含む場合は被験者数が少なくなります．
全件を使う場合は `limit: null` にしますが，使用メモリを確認して段階的に増やしてください．

## 授業のtrain/validation/test

`src/datasets/loader/cohort_data.py` が上記APIを呼び出し，学習用にvalidationを追加します．
fold番号を `k` とすると，次の分割です．

| 用途 | 被験者のfold |
| --- | --- |
| test | `k` |
| validation | `(k + 1) % n_folds` |
| train | 残りのfold |

validationはLoaderのtrain側から被験者索引を使って分けます．
同じ被験者の記録が異なる用途へ入ることはありません．3分割に必要なため `n_folds >= 3` とします．
Unknown（-1）は各窓の正解ラベルに基づいて除外し，窓の索引は保持します．
30秒より長い窓の場合，正解は `y_spec`（latest/center/vote）に従います．

正規化は `dataset.normalization` で指定します．

- `none`：正規化しない．
- `record-wise`：各記録・各チャンネルで正規化する（配布設定）．
- `epoch-wise`：各窓・各チャンネルで正規化する．
- `train-wise`：validation/testを除いたtrainの各チャンネルの統計量を使う．統計量を保存し，評価・予測でも再利用する．

record-wise/epoch-wiseはLoaderの正規化APIを使います．train-wiseはvalidationを分けてUnknownを除いた後に計算します．
従来の `ZScoreNormalize` の記録全体をまとめる正規化とは，統計量を計算する単位が異なります．

## 実行

`configs/SleepCAM.yaml` の `dataset` を確認して，テンプレートのルートで実行します．

```bash
python -m src.train -c configs/SleepCAM.yaml
```

学習時に `cv_setting.json` へ実際に使った記録と被験者のfold割り当てを保存します．
評価時はその設定を読み戻します．
旧CSV分割方式から導入済みリポジトリを更新する場合は，[移行手順](./07_Loader_APIへの移行.md)を参照してください．

**[■ 目次](../README.md)**　**[◀ 前へ](./01_環境を整えよう-Getting_the_Environment_Ready.md)**　**[次へ ▶](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./01_環境を整えよう-Getting_the_Environment_Ready.md)**　**[Next ▶](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**

# 02: Understanding and Preprocessing Sleep Stage Data

Use the standardized PhysioNet 2018 files under `/data/ise_data` through
`src.datasets.loader.standard_sleep_data.load_sleep_record`.
The bundled Sleep-Data-Hub Loader selects channels by name and creates 30-second windows.
The teaching wrapper returns signals as `(epochs, samples, channels)` and aligned stages.
The default four channels run at 200 Hz (6,000 samples per epoch).

Use `load_cohort`, `build_cv_setting`, and `load_training_testing_data` for cohort loading
and participant-wise cross-validation. The course adapter reserves the next fold for validation.
Start training directly with `python -m src.train -c configs/SleepCAM.yaml`.
The default is 20 labeled records; loading is eager, so increase the limit according to memory capacity.
The exact record selection and fold assignment are saved in `cv_setting.json` and reused for evaluation.
Unknown (-1) is excluded according to the label of each window.
The supplied signals have already been filtered during standardization.
Inspect `record.metadata.signal_info` before additional filtering.

**[■ Table of Contents](../README.md)**　**[◀ Previous](./01_環境を整えよう-Getting_the_Environment_Ready.md)**　**[Next ▶](./03_Sleep-CAMモデルを学習してみよう-Training_the_Sleep-CAM_Model.md)**
