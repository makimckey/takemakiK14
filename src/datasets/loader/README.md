# Sleep-Data-Hub Loader

`sleep_loader.py` と `cohort_mix_rules.toml` は `Sleep-Data-Hub/loader/` からそのまま取り込んだ配布用ファイルです．
取り込み元：Sleep-Data-Hub，commit `15c29aca09d5d6dd387e4d115b249409ef9e253f`．
作者の情報と元のLoader実装は維持しています．更新時は元リポジトリの同名ファイルと照合してください．

`cohort_data.py` はコホート読込・被験者CV・validation追加・分割保存と再評価を提供します．
`standard_sleep_data.py` は第9・10回の単一記録の窓分割・配列軸変換，
`sleep_stage_dataset.py` はPyTorch Datasetと記録単位のシャッフルを提供します．
元データは `/data/ise_data` を読み取り専用で使用します．

学習・評価は `cohort_data.py` 経由で標準Loaderの高水準APIを使います．
旧 `SleepStageDataset` は既存の単一記録処理向けに残しています．新しい学習用設定は `docs/07_Loader_APIへの移行.md` を参照してください．
