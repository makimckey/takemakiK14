**[🇯🇵 日本語](#00-はじめに)** | **[🇺🇸 English](#00-introduction-goals-and-learning-outcomes)** 

**[■ 目次](../README.md)**　**[次へ ▶](./01_環境を整えよう-Getting_the_Environment_Ready.md)**

# 00: はじめに

これは、脳波や生体信号（例：EEG/EOG/EMG）から **睡眠ステージ（Wake/N1/N2/N3/REM）** を推定し、Sleep-CAM の訓練・評価・可視化までを行うチュートリアルです．

## 学習目標

- 睡眠ステージ分類の基本（データ構造、アノテーション、30秒エポックの扱い）を理解する
- シンプルな SleepCAM モデルで睡眠ステージ分類を学習・評価できる
- 推定結果から **ヒプノグラム（hypnogram）** を描ける

## 想定する前提

- Python の基本文法を理解している

## 📂 リポジトリ構成
- `src/` : 実験や解析用のコード  
- `paper/` : 論文執筆用のLaTeXファイル (`main.tex`, `refs.bib`)  
- `lab-notebook/` : 実験ノート (`yyyymmdd.md`)  
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
- 論文を書く際は `paper/main.tex` と `paper/refs.bib` を更新してください。  
- 図などのファイルは `paper/figs/` 以下に配置してください。  
- **生体信号データや被験者情報は絶対にアップロードしないでください。**  
- 実験結果や大きな出力ファイルは、必ず `results/` フォルダに保存してください。  

**[■ 目次](../README.md)**　**[次へ ▶](./01_環境を整えよう-Getting_the_Environment_Ready.md)**

---

**[■ Table of Contents](../README.md)**　**[Next ▶](./01_環境を整えよう-Getting_the_Environment_Ready.md)**

# 00: Introduction: Goals and Learning Outcomes

This is a tutorial for estimating **sleep stages (Wake/N1/N2/N3/REM)** from brain waves and biological signals (e.g., EEG/EOG/EMG), and for training, evaluating, and visualizing Sleep-CAM.

## Learning Objectives

- Understand the basics of sleep stage classification (data structure, annotation, handling of 30-second epochs)
- Be able to train and evaluate sleep stage classification with a simple SleepCAM model
- Be able to draw a **hypnogram** from the estimation results

## Prerequisites

- Understand the basic syntax of Python

## 📂 Repository Structure
- `src/` : Source code for experiments or analysis  
- `paper/` : LaTeX files for writing papers (`main.tex`, `refs.bib`)  
- `lab-notebook/` : Experiment notebooks (`yyyymmdd.md`)  
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
- Update `paper/main.tex` and `paper/refs.bib` when writing papers.  
- Place figures and related files under the `paper/figs/` directory.  
- **Never upload raw biosignal data or subject information.**  
- Always save experimental results and large output files under the `results/` directory.  

**[■ Table of Contents](../README.md)**　**[Next ▶](./01_環境を整えよう-Getting_the_Environment_Ready.md)**
