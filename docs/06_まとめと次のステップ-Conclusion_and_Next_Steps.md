**[🇯🇵 日本語](#06-まとめと次のステップ)** | **[🇺🇸 English](#06-conclusion-and-next-steps)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

# 06: まとめと次のステップ

## この教材でできるようになったこと

- PSG/EDF を読み込み、前処理 → 特徴量化 → 学習 → 推論 → 可視化の一連を再現
- CAM/Grad-CAM により、モデルの根拠を可視化・考察
- ヒプノグラムと基本指標（TST/SE/各ステージ比）を自動算出

## 次のステップ（拡張アイデア）

1.	多チャンネル活用：EEG/EOG/EMG を別チャネルとして入力しアテンションで統合
2.	シーケンスモデル：HMM/CRF、Temporal-CNN、Transformer で文脈を取り込む
3.	ドメイン適応：被験者間ばらつきに対して正規化や適応学習を検討
4.	イベント検出：スピンドルや K-complex を検出し、特徴量として条件付け
5.	解釈の強化：リージョンマスキングや周波数アブレーションで因果感のある説明へ
6.	評価の厳密化：LOSO（Leave-One-Subject-Out）で汎化性能を検証、統計比較を実施

## 品質・倫理

- 医療判断には専門家の監督と追加検証が不可欠
- データの匿名化、同意取得、ライセンス順守を徹底

お疲れさまでした！必要に応じてコードを整理し、レポジトリやノートブックとして共有できる形に仕上げてみましょう。

**[■ 目次](../README.md)**　**[◀ 前へ](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**

# 06: Conclusion and Next Steps

(Write the content here)

**[■ Table of Contents](../README.md)**　**[◀ Previous](./05_推定結果を可視化してhypnogramを描こう-Visualizing_Results_Drawing_the_Hypnogram.md)**
