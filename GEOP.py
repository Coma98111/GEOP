import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import shap
from scipy.ndimage import gaussian_filter1d

# =========================
# 0. Basic settings (journal style)
# =========================
mpl.rcParams.update({
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    "pdf.fonttype": 42,  # Use TrueType (Type 42) for PDF so that Illustrator and similar software can read fonts correctly
    "ps.fonttype": 42,  # Use the same font embedding setting for EPS
    "svg.fonttype": "none",  # Keep text as editable text in SVG instead of converting it to path subsets
})
plt.rcParams['figure.dpi'] = 300




# =========================
# 1. General functions
# =========================
def train_xgb_and_analyze(csv_path, label_col, scenario_name,
                          test_size=0.5, random_state=10, n_repeats=5):
    """
    Run the full analysis for one dataset and one label system:
    - train/test split
    - XGBoost training
    - ROC/AUC calculation
    - permutation importance calculation (ΔAUC and standard deviation)
    - SHAP value calculation
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    feature_names = X.columns.tolist()

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'lambda': 10,
        'subsample': 0.98,
        'colsample_bytree': 0.92,
        'min_child_weight': 3,
        'eta': 0.03,
        'seed': 0,
        'nthread': 8,
        'gamma': 0.9,
        'learning_rate': 0.06
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Probability prediction
    y_prob = bst.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    # Basic metrics
    auc = metrics.roc_auc_score(test_y, y_prob)
    acc = metrics.accuracy_score(test_y, y_pred)
    f1 = metrics.f1_score(test_y, y_pred)
    print(f"\n=== {scenario_name} ===")
    print(f"AUC = {auc:.4f}, ACC = {acc:.4f}, F1 = {f1:.4f}")

    # ROC curve
    fpr, tpr, _ = metrics.roc_curve(test_y, y_prob)
    original_auc = auc

    # ---------- Permutation importance based on ΔAUC ----------
    def permutation_importance_auc(model, test_x, y_true, feature_names, n_repeats=5):
        all_perm_scores = []
        for _ in range(n_repeats):
            perm_auc = []
            for feat in feature_names:
                X_perm = test_x.copy()
                X_perm[feat] = np.random.permutation(X_perm[feat])
                y_prob_perm = model.predict(xgb.DMatrix(X_perm))
                perm_auc.append(metrics.roc_auc_score(y_true, y_prob_perm))
            all_perm_scores.append(perm_auc)

        all_perm_scores = np.array(all_perm_scores)  # [n_repeats, n_features]
        mean_perm_auc = all_perm_scores.mean(axis=0)
        delta_auc = original_auc - mean_perm_auc
        std_delta = np.std(original_auc - all_perm_scores, axis=0)

        out = pd.DataFrame({
            "Feature": feature_names,
            "ΔAUC": delta_auc,
            "ΔAUC Std": std_delta,
            "Mean Permuted AUC": mean_perm_auc
        }).sort_values("ΔAUC", ascending=False)

        return out

    perm_df = permutation_importance_auc(
        bst, test_x, test_y, feature_names, n_repeats=n_repeats
    )
    perm_df["Scenario"] = scenario_name

    # SHAP analysis
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(test_x)

    return {
        "name": scenario_name,
        "feature_names": feature_names,
        "perm": perm_df,
        "shap_values": shap_values,
        "X_test": test_x,
        "y_true": test_y.to_numpy(),
        "y_prob": y_prob,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "roc_auc": auc
    }


def make_calibration_points(y_true, y_prob, n_bins=10, min_samples=20):
    """Calculate calibration-curve points: mean predicted probability and observed frequency in each probability bin."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    prob_mean = []
    freq_pos = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])

        n = mask.sum()
        if n < min_samples:
            prob_mean.append(np.nan)
            freq_pos.append(np.nan)
        else:
            prob_mean.append(y_prob[mask].mean())
            freq_pos.append(y_true[mask].mean())

    return np.array(bin_centers), np.array(prob_mean), np.array(freq_pos)


def compute_f1_curve(y_true, y_prob, n_thresholds=50):
    """Calculate F1 scores under different decision thresholds."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    f1_scores = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    return thresholds, np.array(f1_scores)


# =========================
# 2. Process the two input tables separately
# =========================
erosion_csv = "erosion_label.csv"   # Observed erosion-label dataset
geop_csv    = "geop_label.csv"      # GEOP-derived label dataset
label_col   = "CID"                 # Name of the label column in both tables

erosion_res = train_xgb_and_analyze(
    erosion_csv, label_col, scenario_name="Erosion label",
    test_size=0.3, random_state=10, n_repeats=5
)

geop_res = train_xgb_and_analyze(
    geop_csv, label_col, scenario_name="GEOP label",
    test_size=0.3, random_state=10, n_repeats=5
)

# =========================
# 3. Combine ΔAUC results for panel (a)
# =========================
perm_all = pd.concat([erosion_res["perm"], geop_res["perm"]], ignore_index=True)

pivot_delta = perm_all.pivot(index="Feature", columns="Scenario", values="ΔAUC")
pivot_std   = perm_all.pivot(index="Feature", columns="Scenario", values="ΔAUC Std")

order = pivot_delta.sort_values(by="Erosion label", ascending=True).index
pivot_delta = pivot_delta.loc[order]
pivot_std   = pivot_std.loc[order]

features = pivot_delta.index.tolist()
y_pos = np.arange(len(features))

# =========================
# 4. Mean absolute SHAP values for panel (e)
# =========================
shap_vals_erosion = erosion_res["shap_values"]
feat_names = erosion_res["feature_names"]
mean_abs_shap = np.mean(np.abs(shap_vals_erosion), axis=0)

shap_importance_erosion = pd.DataFrame({
    "Feature": feat_names,
    "MeanAbsSHAP": mean_abs_shap
})

imp_merge = (
    perm_all[perm_all["Scenario"] == "Erosion label"][["Feature", "ΔAUC"]]
    .merge(shap_importance_erosion, on="Feature", how="inner")
)

# ---- Select the top five factors for SHAP dependence plots in panel (b) ----
top5_features = (
    perm_all[perm_all["Scenario"] == "Erosion label"]
    .sort_values("ΔAUC", ascending=False)["Feature"]
    .head(5)
    .tolist()
)
print("\nTop 5 features (by ΔAUC of Erosion label):", top5_features)

# =========================
# 5. Calibration-curve data for panel (d)
# =========================
bins = 10
centers_e, prob_e, freq_e = make_calibration_points(
    erosion_res["y_true"], erosion_res["y_prob"], n_bins=bins, min_samples=20
)
centers_g, prob_g, freq_g = make_calibration_points(
    geop_res["y_true"], geop_res["y_prob"], n_bins=bins, min_samples=20
)

mask_e = ~np.isnan(prob_e) & ~np.isnan(freq_e)
mask_g = ~np.isnan(prob_g) & ~np.isnan(freq_g)

# =========================
# 6. Threshold–F1 curve data for panel (f)
# =========================
thr_e, f1_e = compute_f1_curve(erosion_res["y_true"], erosion_res["y_prob"])
thr_g, f1_g = compute_f1_curve(geop_res["y_true"], geop_res["y_prob"])

# =========================
# 7. Generate the three-row multi-panel figure (a–f)
# =========================
fig = plt.figure(figsize=(7.2, 5.0))
gs = gridspec.GridSpec(
    3, 3,
    height_ratios=[1.1, 1.1, 1.0],
    width_ratios=[1.7, 1.7, 1.0],
    hspace=0.5,
    wspace=0.35
)

# ---------- (a) Grouped horizontal bar chart of ΔAUC ----------
ax_a = fig.add_subplot(gs[0, :])

bar_height = 0.32
ax_a.barh(
    y_pos - bar_height/2,
    pivot_delta["Erosion label"],
    height=bar_height,
    xerr=pivot_std["Erosion label"],
    label="Erosion label",
    color="tab:blue",
    ecolor="k",
    capsize=2,
    linewidth=0.6
)

ax_a.barh(
    y_pos + bar_height/2,
    pivot_delta["GEOP label"],
    height=bar_height,
    xerr=pivot_std["GEOP label"],
    label="GEOP label",
    color="tab:orange",
    ecolor="k",
    capsize=2,
    linewidth=0.6
)

ax_a.axvline(0, color='k', linestyle='--', linewidth=0.6)
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(features)
ax_a.set_xlabel("ΔAUC (AUC decrease)")
ax_a.set_ylabel("Feature")
ax_a.set_title("(a) Permutation importance (ΔAUC ± 1σ)", loc="left")
ax_a.legend(loc="lower right", frameon=False)
ax_a.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# ---------- (b) Top-five SHAP dependence plots: two models with smoothed trends ----------
shap_vals_e = erosion_res["shap_values"]
X_test_e    = erosion_res["X_test"]
feat_names_e = erosion_res["feature_names"]

shap_vals_g = geop_res["shap_values"]
X_test_g    = geop_res["X_test"]
feat_names_g = geop_res["feature_names"]

sub_gs = gs[1, 0:2].subgridspec(2, 3, wspace=0.4, hspace=0.45)

# First calculate a unified y-axis range for the two models
ymins, ymaxs = [], []
for feat in top5_features:
    j_e = feat_names_e.index(feat)
    j_g = feat_names_g.index(feat)
    y_e = shap_vals_e[:, j_e]
    y_g = shap_vals_g[:, j_g]
    y_all = np.concatenate([y_e, y_g])
    ymins.append(np.nanpercentile(y_all, 1))
    ymaxs.append(np.nanpercentile(y_all, 99))

ymin = min(ymins)
ymax = max(ymaxs)
pad = 0.1 * (ymax - ymin)
ymin -= pad
ymax += pad

for idx, feat in enumerate(top5_features):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(sub_gs[row, col])

    # ---- Erosion label ----
    j_e = feat_names_e.index(feat)
    x_e = X_test_e[feat].values
    y_e = shap_vals_e[:, j_e]
    order_e = np.argsort(x_e)
    xs_e = x_e[order_e]
    ys_e = y_e[order_e]
    sigma_e = max(2, len(ys_e) / 80.0)
    ys_e_smooth = gaussian_filter1d(ys_e, sigma=sigma_e)

    ax.scatter(xs_e, ys_e, s=6, alpha=0.25,
               color="tab:blue", edgecolor="none", label="Erosion label" if idx == 0 else None)
    ax.plot(xs_e, ys_e_smooth, color="tab:blue", linewidth=1.0)

    # ---- GEOP label ----
    j_g = feat_names_g.index(feat)
    x_g = X_test_g[feat].values
    y_g = shap_vals_g[:, j_g]
    order_g = np.argsort(x_g)
    xs_g = x_g[order_g]
    ys_g = y_g[order_g]
    sigma_g = max(2, len(ys_g) / 80.0)
    ys_g_smooth = gaussian_filter1d(ys_g, sigma=sigma_g)

    ax.scatter(xs_g, ys_g, s=6, alpha=0.25,
               color="tab:orange", edgecolor="none", label="GEOP label" if idx == 0 else None)
    ax.plot(xs_g, ys_g_smooth, color="tab:orange", linewidth=1.0)

    # y = 0 reference line
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)

    # Apply the unified y-axis range
    ax.set_ylim(ymin, ymax)

    # Use the feature name as the x-axis label for each subplot; no title is used
    ax.set_xlabel(feat, fontsize=7)

    # Show y-axis tick labels only in the left column to avoid crowding
    if col == 0:
        ax.tick_params(axis='y', labelleft=True)
    else:
        ax.tick_params(axis='y', labelleft=False)

    ax.tick_params(axis='both', labelsize=6)

# Overall left-side label for panel (b)
fig.text(
    0.015, 0.36,
    "(b) SHAP dependence of leading factors\nSHAP value",
    fontsize=8, rotation=90, va="center"
)

# Place a small legend in the lower-right position of panel (b) to avoid occupying the main plot area
ax_b_legend = fig.add_subplot(sub_gs[1, 2])
ax_b_legend.axis("off")
ax_b_legend.scatter([], [], color="tab:blue", label="Erosion label")
ax_b_legend.scatter([], [], color="tab:orange", label="GEOP label")
ax_b_legend.legend(loc="center", frameon=False)


# ---------- (c) ROC curves ----------
ax_c = fig.add_subplot(gs[1, 2])
ax_c.plot(
    erosion_res["roc_fpr"], erosion_res["roc_tpr"],
    color="tab:blue", lw=1.2,
    label=f"Erosion (AUC = {erosion_res['roc_auc']:.2f})"
)
ax_c.plot(
    geop_res["roc_fpr"], geop_res["roc_tpr"],
    color="tab:orange", lw=1.2,
    label=f"GEOP (AUC = {geop_res['roc_auc']:.2f})"
)
ax_c.plot([0, 1], [0, 1], 'k--', lw=0.6)

ax_c.set_xlim([0.0, 1.0])
ax_c.set_ylim([0.0, 1.05])
ax_c.set_xlabel("False positive rate")
ax_c.set_ylabel("True positive rate")
ax_c.set_title("(c) ROC curves", loc="left")
ax_c.legend(fontsize=6, loc="lower right", frameon=False)
ax_c.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ax_c.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# ---------- (d) Calibration curves ----------
bottom_gs = gs[2, 0:2].subgridspec(1, 2, wspace=0.45)

ax_d = fig.add_subplot(bottom_gs[0, 0])
ax_d.plot(prob_e[mask_e], freq_e[mask_e],
          marker='o', linestyle='-', lw=1.0,
          color='tab:blue', label="Erosion label")
ax_d.plot(prob_g[mask_g], freq_g[mask_g],
          marker='s', linestyle='-', lw=1.0,
          color='tab:orange', label="GEOP label")
ax_d.plot([0, 1], [0, 1], 'k--', lw=0.6)

ax_d.set_xlim(0, 1)
ax_d.set_ylim(0, 1)
ax_d.set_xlabel("Predicted probability")
ax_d.set_ylabel("Observed frequency")
ax_d.set_title("(d) Calibration curves", loc="left")
ax_d.legend(loc="upper left", frameon=False)

# ---------- (e) ΔAUC versus mean |SHAP| ----------
ax_e = fig.add_subplot(bottom_gs[0, 1])
ax_e.scatter(
    imp_merge["ΔAUC"],
    imp_merge["MeanAbsSHAP"],
    s=18,
    edgecolor='k',
    linewidth=0.3
)
for _, row in imp_merge.iterrows():
    ax_e.text(
        row["ΔAUC"], row["MeanAbsSHAP"],
        row["Feature"],
        fontsize=5,
        ha='left', va='center'
    )

ax_e.set_xlabel("ΔAUC (Erosion label)")
ax_e.set_ylabel("Mean |SHAP value|")
ax_e.set_title("(e) ΔAUC vs mean |SHAP|", loc="left")
ax_e.axvline(0, color='k', linestyle='--', linewidth=0.6)
ax_e.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# ---------- (f) F1 score versus decision threshold ----------
ax_f = fig.add_subplot(gs[2, 2])
ax_f.plot(thr_e, f1_e, color="tab:blue", lw=1.0, label="Erosion label")
ax_f.plot(thr_g, f1_g, color="tab:orange", lw=1.0, label="GEOP label")
ax_f.axvline(0.5, color='k', linestyle='--', linewidth=0.6)

ax_f.set_xlim(0, 1)
ax_f.set_ylim(0, 1.05)
ax_f.set_xlabel("Decision threshold")
ax_f.set_ylabel("F1 score")
ax_f.set_title("(f) F1 score vs threshold", loc="left")
ax_f.legend(loc="lower left", frameon=False)
ax_f.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.show()
