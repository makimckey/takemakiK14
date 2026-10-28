"""旧CSV分割コマンドの移行案内。分割はLoaderのCV APIが行います。"""

if __name__ == "__main__":
    raise SystemExit(
        "CSV分割はLoaderのCV APIに移行しました。\n"
        "configs/SleepCAM.yaml の dataset を設定し、次を実行してください。\n"
        "  python -m src.train -c configs/SleepCAM.yaml\n"
        "導入済みリポジトリの移行手順: docs/07_Loader_APIへの移行.md"
    )
