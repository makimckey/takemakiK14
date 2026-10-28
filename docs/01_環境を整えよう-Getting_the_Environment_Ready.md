**[🇯🇵 日本語](#01-環境を整えよう)** |  **[🇺🇸 English](#01-getting-the-environment-ready)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./00_はじめに-Introduction.md)**　**[次へ ▶](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**

# 01: 環境を整えよう

ここでは、本チュートリアルで利用する Python 環境を準備します。

## 🚀 初回セットアップと実行方法

リポジトリを Clone したら、まず以下を実行してください:
```bash
./setup.sh
```

> [!NOTE]  
> - Docker コンテナの作成と環境確認を行い、自動的に run.sh が生成されます。  
> - setup.sh は一度実行すると削除されます。  
> - 2回目以降の実行時は以下を使用してください:  

```bash
./run.sh
```

## 開発ライブラリのセットアップ

まずは以下のコマンドを実行し，Docker コンテナ内へ入ります．

```shell
./run.sh
```

そうしたらまずは uv という Python のライブラリマネージャーをインストールします．
以下のコマンドを実行して，uv をインストールしてください．

```shell
pip install uv
```

uv がインストールされましたら，以下のコマンドで環境を再現します．

```shell
uv sync
```

先程のコマンドで Python の仮想環境が作成されます．以下のコマンドで仮想環境を有効化します．

```shell
source .venv/bin/activate
```

実験の際はこの仮想環境を有効にしてから Python を実行します．
これで環境構築は完了です！

**[■ 目次](../README.md)**　**[◀ 前へ](./00_はじめに-Introduction.md)**　**[次へ ▶](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./00_はじめに-Introduction.md)**　**[Next ▶](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**

# 01: Getting the Environment Ready

Here, we will prepare the Python environment used in this tutorial.

## 🚀 Initial Setup and How to Run

After cloning the repository, first execute the following:

```bash
./setup.sh
```

> [!NOTE]
> - This will create the Docker container, check the environment, and automatically generate run.sh.
> - setup.sh will be deleted once it is executed.
> - For subsequent runs, use the following:

```bash
./run.sh
```

## Setting Up Development Libraries

First, run the following command to enter the Docker container.

```shell
./run.sh
```

Then, install a Python library manager called uv.
Run the following command to install uv:

```shell
pip install uv
```

Once uv is installed, reproduce the environment with the following command:

```shell
uv sync
```

The previous command will create a Python virtual environment. Enable it with the following command:

```shell
source .venv/bin/activate
```

When running experiments, make sure to activate this virtual environment before running Python.
That completes the environment setup!


**[■ Table of Contents](../README.md)**　**[◀ Previous](./00_はじめに-Introduction.md)**　**[Next ▶](./02_睡眠ステージデータの理解と前処理-Understanding_and_Preprocessing_Sleep_Stage_Data.md)**
