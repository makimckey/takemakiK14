# 主専攻実験を受講する学生さんへ
2025/Nov./12  
文責：堀江和正（kdesleep-code）

> [!CAUTION]
> **重要: このREADMEを削除しないでください。**  
> このファイルには、KDE睡眠チームのリポジトリ利用に関する必須ルール、ライセンス、利用方針が記載されています。
> 
> **このリポジトリは 実験用レポジトリのテンプレートです。直接 clone / push は禁止です（学生の権限は Read のみ）。**  
> 利用方法：右上の Use this template → Create a new repository を選び、以下を設定してください。  
> Repository name：各自の名前や課題名を付ける  
> Visibility：**Private** に  
>（必要な場合のみ）Include all branches を有効化  
> **作成後は自分のリポジトリに対してのみ commit / push を行ってください。**  

> [!CAUTION]
> **Important: Do not delete this README.**  
> This file contains the mandatory rules, license, and usage policies of the KDE Sleep Team repository.
> 
> **This is an experiment repository template. Cloning/pushing here is prohibited (students have Read access only).**  
> How to use: Use this template → Create a new repository, then:  
> Repository name: choose your own (e.g., your name/topic)  
> Visibility: must be **Private**   
> Enable Include all branches only if needed  
> **After creation, commit/push to your own repo (not this template).**  

---

## 📂 リポジトリ構成
- `src/` : 実験用のコードのテンプレート  
- `final_report/` : 最終レポートの提出場所  
- `exercise_report/` : 各週の課題の提出場所  
- `results/` : 実験結果ファイル置き場（**git管理対象外**）
- `.github/` : GithubのAutomation機能用．編集しないことをお勧めします．

> [!CAUTION]
> **実験用データや統計処理前の生データのPushは禁止です。**  
> **実験結果や大きなファイルは必ず `results/` フォルダに保存してください。**  
> **禁止拡張子や50MBを超えるファイルは GitHub 側のルールセットによりPushが拒否されます。**  

---

## ✅ ファイルポリシー
このリポジトリでは **GitHub Organization の Push Ruleset** によりファイル制限が適用されています。  

- 許可されるファイルタイプ:  
  `.py`, `.ipynb`, `.tex`, `.bib`, `.md`, `.yml`, `.yaml`,  
  `.csv`, `.txt`, `.json`, `.xlsx`,  `.jpg`, `.pdf`, `.eps`  

- 禁止される拡張子（Push自体が拒否されます）:  
  `.edf`, `.bdf`, `.bin`, `.rml`, `.wav`, `.mp3`, `.m4a`, `.flac`, `.aac`, `.npz`, `.npy`, `.png`, `.svg`, `.tiff` , `.tif`

- **50MB以上のファイルはPushできません。エラーが出た場合は `results/` に移してください。**

- `README.md` は **必須**であり、削除禁止です。  

---

## 🎓 学生への注意
- コードは整理し、必ずコメントを記載してください。  
- **ファイルポリシーで許可されたファイルのみをPushしてください。**  
- **生体信号データや被験者情報は絶対にアップロードしないでください。**  
- 実験結果や大きな出力ファイルは、必ず `results/` フォルダに保存してください。  

---

## 実験データの読み込み（Sleep-Data-Hub標準形式）

授業用のPhysioNet 2018データは `/data/ise_data` にあり，コンテナ内でも同じパスで読み取り専用です．
旧形式の `np.load(...)["signal"]` では読めません．
`src/datasets/loader/standard_sleep_data.py` を使います．
実験用リポジトリにLoaderを同梱しているため，Sleep-Data-Hubを別途cloneする必要はありません．

```python
from src.datasets.loader.standard_sleep_data import find_example_file, load_sleep_record
record = load_sleep_record(find_example_file())
print(record.signals.shape)  # (エポック数, 6000, 4)
print(record.channel_names)
```

学習はLoaderの `load_cohort` → `build_cv_setting` → `load_training_testing_data` を使います．
SQLiteから教師あり記録を選び，被験者単位でtrain/validation/testに分けます．

```bash
python -m src.train -c configs/SleepCAM.yaml
```

設定は `dataset.cohort_dir`，`cohort_id`，`cv`，`channel_set`，`window_sec`，`stride_sec`，
`y_spec`，`normalization`，`exclude_labels` です．`channel_set` の数と `model.config.input_channels` を揃えてください．
Loaderがデータをメモリに展開するため，配布設定は `dataset.limit: 20`（正解付き20記録）です．
記録数はメモリを確認して増やせます．全件を使う場合は `limit: null` とします．
5-foldではtestが指定fold，validationが次のfold，trainが残り3foldです．

実験ディレクトリに `config.yaml` と `cv_setting.json`（使用記録・被験者のfold割り当て）を保存します．
再評価は保存した分割を使います．

```bash
python -m src.test -i 実際の実験ディレクトリ
```

詳細は [データの理解と前処理](docs/02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)，
導入済みリポジトリの更新は [Loader APIへの移行](docs/07_Loader_APIへの移行.md)を参照してください．

## 🚀 初回セットアップと実行方法
リポジトリを Clone したら、まず以下を実行してください:
```bash
./setup.sh
```

> [!NOTE]  

- Docker コンテナの作成と環境確認を行い、自動的に run.sh が生成されます。  
- setup.sh は一度実行すると削除されます。  
- 2回目以降の実行時は以下を使用してください:  
```bash
./run.sh
```

- Docker コンテナが起動し、開発環境に入ります。  
- ここで `python src/main.py` などを実行してください。  

## ⚙️ Automation (CI)
- **Guard: 必須ファイル/フォルダのチェック（main への push 時）**  
  - 次を監視します：  
    - 必須ファイル：`README.md`, `LICENSE`, `requirements.txt`, `.gitignore` , `.editconfig`
      - **無ければエラー**（Actions は失敗、**Issue が自動作成**され push した人にアサイン）  
      - **空ファイルは警告**（CI は通過）  
    - 必須フォルダ：`src/`, `lab-notebook/`, `paper`  

---

# Message for Information-System-Experiment Students

---

## 📂 Repository Structure
- `src/` : Source code for experiments or analysis  
- `final_report/` : Submission folder for the final report.
- `exercise_report/` : Submission folder for weekly assignments  
- `results/` : Storage for experimental results (**ignored by git**)
- `.github/` : For GitHub Automation workflows. Editing is not recommended.
  
> [!CAUTION]  
> **Pushing experimental data or raw experimental results (not yet statistically processed) is strictly prohibited.**  
> **Always save experimental results and large files in the `results/` directory.**  
> **Pushes containing forbidden extensions or files over 50MB will be rejected by the GitHub Push Ruleset.**  

---

## ✅ File Policy
File restrictions are enforced by the **GitHub Organization Push Ruleset**:  

- Allowed file types:  
  `.py`, `.ipynb`, `.tex`, `.bib`, `.md`, `.yml`, `.yaml`,  
  `.csv`, `.txt`, `.json`, `.xlsx`,  `.jpg`, `.pdf`, `.eps`  

- Forbidden extensions (Push will be rejected):  
  `.edf`, `.bdf`, `.bin`, `.rml`, `.wav`, `.mp3`, `.m4a`, `.flac`, `.aac`, `.npz`, `.npy`, `.png`, `.svg`, `.tiff` , `.tif`

- **Files larger than 50MB cannot be pushed. If you encounter this error, move the file to the `results/` folder.**

- `README.md` is **mandatory** and must not be removed.  

---

## 🎓 Notes for Students
- Always keep your code organized and documented.  
- **Only push files permitted by the File policy.**  
- **Never upload raw biosignal data or subject information.**  
- Always save experimental results and large output files under the `results/` directory.  

---

## 🚀 Initial Setup and Running
After cloning the repository, first run:
```bash
./setup.sh
```

- This will make your own Docker container, verify the environment, and automatically generate run.sh.  
- setup.sh will be deleted after it is executed once.  
- For all subsequent runs, use:  
```bash
./run.sh
```

- This will start the Docker container and open the development environment.  
- Inside, you can run commands such as `python src/main.py`.  

## ⚙️ Automation (CI)
- **Guard: Required files/folders check (on push to `main`)**  
  - Monitors:  
    - Required files: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `.editconfig`
      - **Missing ⇒ error** (workflow fails and an **Issue is auto-created** and assigned to the pusher)  
      - **Empty ⇒ warning** (workflow still passes)  
    - Required directories: `src/`, `lab-notebook/`, `paper/`
