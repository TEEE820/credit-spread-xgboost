"""论文图表：中文字体设置 + 全部图/表一键产出。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib import import_module

model = import_module("03_model")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")

RATING_NAMES = {
    "cp_3y": "中票3Y", "cp_5y": "中票5Y",
    "bank_3y": "商行债3Y", "bank_5y": "商行债5Y",
}
MAIN = "cp_3y"


def generate_all():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    files = []
    data = pd.read_csv(os.path.join(PROC, "model_data.csv"), parse_dates=[0], index_col=0)
    targets = model.TARGETS

    # 图1：各利差时序
    fig, ax = plt.subplots(figsize=(10, 4))
    for t in targets:
        ax.plot(data.index, data[t], label=RATING_NAMES[t.replace("spread_", "")], lw=1)
    ax.set_title("图1 各品种信用利差时序（2015-2026）")
    ax.set_ylabel("利差（%）")
    ax.legend()
    fp = os.path.join(FIG, "fig1_spread_series.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图2：描述统计箱线图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([data[t].dropna() for t in targets],
               tick_labels=[RATING_NAMES[t.replace("spread_", "")] for t in targets])
    ax.set_title("图2 各品种利差分布")
    fp = os.path.join(FIG, "fig2_boxplot.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图3-6：各利差 SHAP 重要性条形图
    for i, t in enumerate(targets):
        rating = t.replace("spread_", "")
        imp = pd.read_csv(os.path.join(PROC, f"shap_importance_{rating}.csv"))
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.head(12).iloc[::-1].plot.barh(x="feature", y="mean_abs_shap", ax=ax, legend=False)
        ax.set_title(f"图{3+i} {RATING_NAMES[rating]} SHAP 特征重要性")
        ax.set_xlabel("平均 |SHAP|")
        fp = os.path.join(FIG, f"fig{3+i}_shap_{rating}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图7：品种间 SHAP 对比（前 8 特征）
    merged = None
    for t in targets:
        rating = t.replace("spread_", "")
        imp = pd.read_csv(os.path.join(PROC, f"shap_importance_{rating}.csv")).set_index("feature")
        imp = imp.rename(columns={"mean_abs_shap": RATING_NAMES[rating]})
        merged = imp if merged is None else merged.join(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    merged.head(8).plot.bar(ax=ax)
    ax.set_title("图7 品种间特征重要性对比")
    ax.set_ylabel("平均 |SHAP|")
    fp = os.path.join(FIG, "fig7_cross_spread_shap.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图8：滚动预测（主利差 cp_3y）
    roll = pd.read_csv(os.path.join(PROC, f"rolling_results_{MAIN}.csv"), parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll["date"], roll["y_true"], label="实际", lw=1)
    ax.plot(roll["date"], roll["y_pred"], label="XGBoost 预测", lw=1, alpha=0.8)
    ax.set_title("图8 中票3Y利差滚动样本外预测（5年训练窗）")
    ax.set_ylabel("利差（%）")
    ax.legend()
    fp = os.path.join(FIG, "fig8_rolling_pred.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图9：模型性能对比（R²）
    perf = pd.read_csv(os.path.join(PROC, "model_performance.csv"))
    perf_plot = perf.copy()
    perf_plot["品种"] = perf_plot["rating"].map(RATING_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    perf_plot[["品种", "ols", "random_forest", "xgboost"]].set_index("品种").plot.bar(ax=ax)
    ax.set_title("图9 模型性能对比（测试集 $R^2$）")
    ax.set_ylabel("$R^2$")
    fp = os.path.join(FIG, "fig9_model_performance.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 表1：描述统计表
    t1 = pd.DataFrame({
        RATING_NAMES[t.replace("spread_", "")]: [
            data[t].mean(), data[t].std(), data[t].min(), data[t].max(),
            data[t].skew(), data[t].kurtosis(),
        ] for t in targets
    }, index=["均值", "标准差", "最小值", "最大值", "偏度", "峰度"]).round(4)
    fp = os.path.join(TAB, "table1_summary_stats.csv"); t1.to_csv(fp, encoding="utf-8-sig"); files.append(fp)

    # 表2：模型性能表
    perf2 = perf[["rating", "ols", "random_forest", "xgboost"]].copy()
    perf2["品种"] = perf2["rating"].map(RATING_NAMES)
    perf2 = perf2[["品种", "ols", "random_forest", "xgboost"]].round(4)
    fp = os.path.join(TAB, "table2_model_performance.csv")
    perf2.to_csv(fp, index=False, encoding="utf-8-sig"); files.append(fp)

    # 表3：滚动评估指标表
    rm = pd.read_csv(os.path.join(PROC, "rolling_metrics.csv"))
    rm["品种"] = rm["rating"].map(RATING_NAMES)
    rm["RMSE改进"] = (1 - rm["rmse"] / rm["rw_rmse"]) * 100
    t3 = rm[["品种", "rmse", "mae", "direction_accuracy", "rw_rmse", "RMSE改进"]].round(4)
    t3.columns = ["品种", "RMSE", "MAE", "方向准确率", "随机游走RMSE", "RMSE改进%"]
    fp = os.path.join(TAB, "table3_rolling_metrics.csv")
    t3.to_csv(fp, index=False, encoding="utf-8-sig"); files.append(fp)

    print("generated", len(files), "files")
    return files


if __name__ == "__main__":
    generate_all()
