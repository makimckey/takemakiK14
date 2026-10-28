**[🇯🇵 日本語](#05-推定結果を可視化してhypnogramを描こう)** | **[🇺🇸 English](#05-visualizing-results-drawing-the-hypnogram)** 

**[■ 目次](../README.md)**　**[◀ 前へ](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**　**[次へ ▶](./06_まとめと次のステップ-Conclusion_and_Next_Steps.md)**

# 05: 推定結果を可視化してhypnogramを描こう

## ヒプノグラムの作成

ステージの時間推移を 30 秒単位で段差プロットします。

```python
# visualize.py（概念コード）
import numpy as np, matplotlib.pyplot as plt

STAGES = ["W","N1","N2","N3","REM"]
map2int = {s:i for i,s in enumerate(STAGES)}

def plot_hypnogram(pred_stages, epoch_sec=30):
    y = np.array([map2int[s] if isinstance(s,str) else s for s in pred_stages])
    t = np.arange(len(y)+1) * epoch_sec / 60.0  # 分
    y_step = np.repeat(y, 2); y_step = np.r_[y_step, y[-1]]
    t_step = np.repeat(t[:-1], 2); t_step = np.r_[t_step, t[-1]]
    plt.figure(figsize=(10,3))
    plt.step(t_step, y_step, where="post")
    plt.yticks(range(len(STAGES)), STAGES)
    plt.xlabel("Time [min]"); plt.ylabel("Stage")
    plt.title("Hypnogram")
    plt.tight_layout(); plt.show()
```

## 指標の算出（例）

- TST（総睡眠時間）：(全エポック − W エポック) × 30 秒
- 睡眠効率（SE）：TST / 就床時間
- 各ステージ割合：N1/N2/N3/REM の比率
- REM 潜時：入眠から最初の REM までの時間

```python
def sleep_metrics(pred, epoch_sec=30, time_in_bed_min=None):
    pred = np.asarray(pred)
    is_sleep = pred != map2int["W"]
    tst_min = is_sleep.sum()*epoch_sec/60
    stage_ratio = {s:(pred==i).mean() for s,i in map2int.items()}
    se = None
    if time_in_bed_min:
        se = tst_min / time_in_bed_min
    return {"TST_min":tst_min, "StageRatio":stage_ratio, "SE":se}
```

**[■ 目次](../README.md)**　**[◀ 前へ](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**　**[次へ ▶](./06_まとめと次のステップ-Conclusion_and_Next_Steps.md)**

---

**[■ Table of Contents](../README.md)**　**[◀ Previous](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**　**[Next ▶](./06_まとめと次のステップ-Conclusion_and_Next_Steps.md)**

# 05: Visualizing Results: Drawing the Hypnogram

(Write the content here)

**[■ Table of Contents](../README.md)**　**[◀ Previous](./04_Sleep-CAMで睡眠ステージを推定しよう-Inferring_Sleep_Stages_with_Sleep-CAM.md)**　**[Next ▶](./06_まとめと次のステップ-Conclusion_and_Next_Steps.md)**
