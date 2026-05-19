import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb

# =========================
# 0. Basic settings (journal style)
# =========================
mpl.rcParams.update({
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})
plt.rcParams["figure.dpi"] = 300


# =========================
# 1. General function for training and prediction
# =========================
def train_xgb_and_predict(
    csv_path,
    label_col="CID",
    model_name="model",
    test_size=0.3,
    random_state=10,
):
    """
    Read the data, split it into training and test sets, train XGBoost, and return test-set predictions and metrics.
    """
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[label_col])
    y = df[label_col].values
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "lambda": 10,
        "subsample": 0.98,
        "colsample_bytree": 0.92,
        "min_child_weight": 3,
        "eta": 0.03,
        "seed": 0,
        "nthread": 8,
        "gamma": 0.9,
        "learning_rate": 0.06,
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Probability prediction and default threshold of 0.5
    y_prob = bst.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    auc = metrics.roc_auc_score(y_test, y_prob)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]

    print(f"\n=== {model_name} ===")
    print(f"AUC  = {auc:.4f}")
    print(f"ACC  = {acc:.4f}")
    print(f"F1   = {f1:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return {
        "name": model_name,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "cm": cm,
    }


# =========================
# 2. Load the two datasets and train the models
# =========================
erosion_csv = "erosion_label.csv"  # Mapped erosion labels
geop_csv = "geop_label.csv"        # GEOP output labels
label_col = "CID"

erosion_res = train_xgb_and_predict(
    erosion_csv, label_col=label_col, model_name="Erosion"
)
geop_res = train_xgb_and_predict(
    geop_csv, label_col=label_col, model_name="GEOP"
)


# =========================
# 3. Helper functions
# =========================
def compute_positive_fraction(y_prob, n_thresholds=101):
    """
    Compute the percentage of samples predicted as positive under different thresholds.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    frac = []
    for th in thresholds:
        frac.append((y_prob >= th).mean() * 100.0)
    return thresholds, np.array(frac)


def plot_confusion_matrix(ax, cm, title=""):
    """
    Plot a compact confusion matrix. Color indicates percentage; text indicates count and percentage.
    cm: 2 x 2, [[TN, FP], [FN, TP]]
    """
    cm = np.asarray(cm)
    total = cm.sum()
    cm_pct = cm / total * 100.0 if total > 0 else np.zeros_like(cm, dtype=float)

    im = ax.imshow(cm_pct, vmin=0, vmax=100, cmap="Blues")

    # Add count and percentage labels inside each cell
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=6,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=8)

    # Add a compact colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)


# =========================
# 4. Generate the c1-c3 composite figure
# =========================
fig = plt.figure(figsize=(7.5, 6.0))
outer_gs = gridspec.GridSpec(
    3, 1,
    height_ratios=[1.0, 1.2, 1.0],
    hspace=0.5
)

# ---------- (c1) Overall performance ----------
ax_c1 = fig.add_subplot(outer_gs[0])

metric_names = ["Accuracy", "F1", "AUC"]
erosion_vals = [erosion_res["acc"], erosion_res["f1"], erosion_res["auc"]]
geop_vals    = [geop_res["acc"], geop_res["f1"], geop_res["auc"]]

x = np.arange(len(metric_names))
width = 0.35

ax_c1.bar(x - width/2, erosion_vals, width, label="Erosion", color="#4C72B0")
ax_c1.bar(x + width/2, geop_vals,    width, label="GEOP",    color="#DD8452")

# Add value labels above the bars
for i, v in enumerate(erosion_vals):
    ax_c1.text(x[i] - width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
for i, v in enumerate(geop_vals):
    ax_c1.text(x[i] + width/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

ax_c1.set_xticks(x)
ax_c1.set_xticklabels(metric_names)
ax_c1.set_ylabel("Score")
ax_c1.set_ylim(0.0, 1.05)
ax_c1.set_title("(c1) Overall performance", loc="left")
ax_c1.legend(loc="lower right", frameon=False)

# ---------- (c2) Confusion matrices ----------
inner_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[1], wspace=0.25
)

ax_cm1 = fig.add_subplot(inner_gs[0, 0])
ax_cm2 = fig.add_subplot(inner_gs[0, 1])

plot_confusion_matrix(ax_cm1, erosion_res["cm"], title="Erosion (threshold = 0.5)")
plot_confusion_matrix(ax_cm2, geop_res["cm"],   title="GEOP (threshold = 0.5)")

fig.text(
    0.01, 0.55,
    "(c2) Confusion matrices",
    fontsize=8,
    rotation=90,
    va="center"
)

# ---------- (c3) Fraction of area flagged ----------
ax_c3 = fig.add_subplot(outer_gs[2])

thr_e, frac_e = compute_positive_fraction(erosion_res["y_prob"])
thr_g, frac_g = compute_positive_fraction(geop_res["y_prob"])

ax_c3.plot(thr_e, frac_e, color="#4C72B0", lw=1.2, label="Erosion")
ax_c3.plot(thr_g, frac_g, color="#DD8452", lw=1.2, label="GEOP")

ax_c3.axvline(0.5, color="k", linestyle="--", linewidth=0.7)
ax_c3.text(0.5, ax_c3.get_ylim()[1]*0.95, "0.5", ha="center", va="top", fontsize=6)

ax_c3.set_xlim(0, 1)
ax_c3.set_ylim(0, 100)
ax_c3.set_xlabel("Decision threshold")
ax_c3.set_ylabel("Positive predictions (%)")
ax_c3.set_title("(c3) Fraction of area flagged", loc="left")
ax_c3.legend(loc="upper right", frameon=False)
ax_c3.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.show()
