import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("data/wine.csv")

X = df["alcohol"]
Y = df["quality"]

# 線形回帰
a, b = np.polyfit(X, Y, 1)

# 回帰直線用データ
x_line = np.linspace(X.min(), X.max(), 100)
y_line = a * x_line + b

# 図の作成
fig, ax = plt.subplots(figsize=(6, 6))

# 散布図
ax.scatter(X, Y, color="#1f5b7a", s=20)

# 回帰直線
ax.plot(x_line, y_line, color="red", linewidth=2)

# 軸を消す
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])
ax.set_yticks([])

# 軸の矢印
ax.arrow(
    X.min()-0.5, Y.min()-0.5,
    (X.max()-X.min())+1.5, 0,
    head_width=0.08,
    head_length=0.15,
    fc="black", ec="black",
    length_includes_head=True
)

ax.arrow(
    X.min()-0.5, Y.min()-0.5,
    0, (Y.max()-Y.min())+1.0,
    head_width=0.08,
    head_length=0.15,
    fc="black", ec="black",
    length_includes_head=True
)

# x, y ラベル
ax.text(X.mean(), Y.min()-0.9, "alcohol", fontsize=16, style="italic")
ax.text(X.min()-0.9, Y.mean(), "quality", fontsize=16, style="italic")

# 回帰式
ax.text(
    X.max()-2,
    Y.max()+0.2,
    rf"$y={a:.2f}x+{b:.2f}$",
    fontsize=16
)

plt.tight_layout()
plt.show()