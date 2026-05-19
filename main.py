import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import shap
import matplotlib.colors as mcolors
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sns

# --------------------------Basic settings--------------------------
# Font settings. English labels are used to avoid Chinese-font rendering issues.
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs are displayed correctly.

# Custom color ramp (green-yellow-red).
colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
green_red_cmap = mcolors.LinearSegmentedColormap.from_list("green_red_cmap", colors)

# --------------------------Data preparation--------------------------
# Load the dataset.
df = pd.read_csv('geop_label.csv', encoding='GBK')
data = df.iloc[:, 1:-1]  # Features.
target = df.iloc[:, -1]  # Labels.

# Split the dataset into training and testing subsets while preserving class balance.
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.3, random_state=10, stratify=target
)
feature_names = data.columns.tolist()  # List of feature names.

# --------------------------Model training--------------------------
# Build DMatrix objects for XGBoost.
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# XGBoost parameter settings.
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 7,
    'lambda': 10,
    'subsample': 0.98,
    'colsample_bytree': 0.92,
    'min_child_weight': 3,
    'eta': 0.029781123443663814,
    'seed': 0,
    'nthread': 8,
    'gamma': 0.91,
    'learning_rate': 0.06
}

# Train the model with early stopping.
bst = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# --------------------------Model evaluation--------------------------
# Prediction and default thresholding.
ypred = bst.predict(dtest)  # Predicted probabilities.
y_pred = (ypred >= 0.5) * 1  # Predicted classes.

# Print evaluation metrics.
print("\n=== Model Evaluation ===")
print(f'Precision: {metrics.precision_score(test_y, y_pred):.4f}')
print(f'Recall: {metrics.recall_score(test_y, y_pred):.4f}')
print(f'F1-score: {metrics.f1_score(test_y, y_pred):.4f}')
print(f'Accuracy: {metrics.accuracy_score(test_y, y_pred):.4f}')
print(f'AUC: {metrics.roc_auc_score(test_y, ypred):.4f}')

# --------------------------Recall curve and threshold analysis--------------------------
# Calculate recall, precision and F1 score under different decision thresholds.
thresholds = np.linspace(0, 1, 100)
recall_scores = []
precision_scores = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (ypred >= threshold) * 1
    recall_scores.append(metrics.recall_score(test_y, y_pred_threshold))
    precision_scores.append(metrics.precision_score(test_y, y_pred_threshold))
    f1_scores.append(metrics.f1_score(test_y, y_pred_threshold))

# Figure 1: recall curve.
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_scores, color='green', lw=2, label='Recall')
plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.title('Recall Curve Across Different Thresholds')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 2: combined recall-precision-F1 threshold curves.
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recall_scores, color='green', lw=2, label='Recall')
plt.plot(thresholds, precision_scores, color='blue', lw=2, label='Precision')
plt.plot(thresholds, f1_scores, color='purple', lw=2, label='F1-score')
plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Recall, Precision and F1-score Across Thresholds')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: precision-recall curve.
precision, recall, _ = metrics.precision_recall_curve(test_y, ypred)
pr_auc = metrics.auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------Permutation importance analysis (Delta AUC)--------------------------
original_auc = metrics.roc_auc_score(test_y, ypred)
print(f'\nOriginal test set AUC: {original_auc:.4f}')


# Calculate permutation importance.
def permutation_importance_auc(model, test_x, y_true, feature_names, n_repeats=5):
    auc_scores = []
    for _ in range(n_repeats):
        perm_auc = []
        for feature in feature_names:
            X_perm = test_x.copy()
            X_perm[feature] = np.random.permutation(X_perm[feature])
            ypred_perm = model.predict(xgb.DMatrix(X_perm))
            perm_auc.append(metrics.roc_auc_score(y_true, ypred_perm))
        auc_scores.append(perm_auc)
    mean_perm_auc = np.mean(auc_scores, axis=0)
    delta_auc = original_auc - mean_perm_auc  # Negative values may occur.
    return pd.DataFrame({
        'Feature': feature_names,
        'ΔAUC': delta_auc,
        'Mean Permuted AUC': mean_perm_auc
    }).sort_values(by='ΔAUC', ascending=False)


# Calculate and print permutation-importance results.
perm_importance = permutation_importance_auc(bst, test_x, test_y, feature_names)
print("\n=== Permutation Importance (ΔAUC) ===")
print(perm_importance)


# Calculate the standard deviation of Delta AUC as a stability metric.
def permutation_auc_std(model, test_x, y_true, feature_names, n_repeats=5):
    perm_std = []
    for feature in feature_names:
        perm_scores = []
        for _ in range(n_repeats):
            X_perm = test_x.copy()
            X_perm[feature] = np.random.permutation(X_perm[feature])
            perm_scores.append(metrics.roc_auc_score(y_true, model.predict(xgb.DMatrix(X_perm))))
        perm_std.append(np.std(original_auc - np.array(perm_scores)))
    return perm_std


perm_importance['ΔAUC Std'] = permutation_auc_std(bst, test_x, test_y, feature_names)

# --------------------------Delta AUC visualization--------------------------
# Figure 1: Delta AUC bar chart with error bars.
plt.figure(figsize=(10, 6))
bar_colors = ['skyblue' if x >= 0 else 'salmon' for x in perm_importance['ΔAUC']]
plt.barh(perm_importance['Feature'], perm_importance['ΔAUC'], color=bar_colors)
plt.errorbar(
    x=perm_importance['ΔAUC'],
    y=range(len(perm_importance)),
    xerr=perm_importance['ΔAUC Std'],
    fmt='none', ecolor='black', capsize=5
)
plt.axvline(x=0, color='black', linestyle='--')  # Zero reference line.
plt.xlabel('ΔAUC (AUC Decrease)')
plt.ylabel('Features')
plt.title('Permutation Importance with Stability (ΔAUC ± Std)')
plt.gca().invert_yaxis()  # Sort from high to low.
plt.tight_layout()
plt.show()

# Figure 2: comparison between original AUC and permuted AUC.
plt.figure(figsize=(10, 6))
x = np.arange(len(perm_importance))
width = 0.35
plt.bar(x - width / 2, [original_auc] * len(perm_importance), width, label='Original AUC')
plt.bar(x + width / 2, perm_importance['Mean Permuted AUC'], width, label='Permuted AUC')
plt.xlabel('Features')
plt.ylabel('AUC Value')
plt.title('Original vs Permuted AUC by Feature')
plt.xticks(x, perm_importance['Feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 3: relative feature contribution.
total_abs_auc = perm_importance['ΔAUC'].abs().sum()
perm_importance['Relative Contribution (%)'] = (perm_importance['ΔAUC'].abs() / total_abs_auc) * 100

plt.figure(figsize=(10, 6))
plt.barh(perm_importance['Feature'], perm_importance['Relative Contribution (%)'], color=bar_colors)
for i, v in enumerate(perm_importance['Relative Contribution (%)']):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
plt.xlabel('Relative Contribution (%)')
plt.ylabel('Features')
plt.title('Relative Importance by Absolute ΔAUC')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --------------------------SHAP analysis--------------------------
# Calculate SHAP values.
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(test_x)
expected_value = explainer.expected_value

# Calculate SHAP-based importance.
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)

print("\n=== SHAP-based Feature Importance ===")
print(shap_importance)

# --------------------------SHAP visualization--------------------------
# Figure 1: SHAP summary plot.
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_x, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show()

# Figure 2: individual dependence plots for the top four features.
top4_features = shap_importance['Feature'].iloc[:4].tolist()
print(f'\nTop 4 features for detailed analysis: {top4_features}')

for feature in top4_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values, test_x,
        show=False, dot_size=30
    )
    plt.title(f'SHAP Dependence Plot: {feature}')
    plt.tight_layout()
    plt.show()

# Figure 3: 4 x 4 feature-interaction matrix for all pairwise combinations among the top four features.
if len(top4_features) == 4:
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, main_feature in enumerate(top4_features):
        for j, interact_feature in enumerate(top4_features):
            # Plot the SHAP interaction dependence plot.
            shap.dependence_plot(
                main_feature, shap_values, test_x,
                interaction_index=interact_feature,
                ax=axs[i, j], show=False,
                dot_size=15,  x_jitter=0.1
            )

            # Refine subplot appearance.
            axs[i, j].set_title(f'{main_feature} vs {interact_feature}', fontsize=10)
            axs[i, j].set_xlabel(main_feature, fontsize=8)
            axs[i, j].set_ylabel(f'SHAP value of {main_feature}', fontsize=8)
            axs[i, j].tick_params(axis='both', labelsize=6)

            # Add a diagonal annotation.
            if i == j:
                axs[i, j].text(0.5, 0.5, 'Same feature',
                               ha='center', va='center',
                               fontsize=12, color='gray', fontweight='bold')

    fig.suptitle('SHAP Interaction Matrix: Pairwise Comparison of Top 4 Features',
                 fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Figure 4: waterfall plot for one sample.
sample_idx = 0  # Select the first sample.
shap_waterfall = shap.Explanation(
    values=shap_values[sample_idx],
    base_values=expected_value,
    data=test_x.iloc[sample_idx],
    feature_names=feature_names
)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_waterfall)
plt.title(f'SHAP Waterfall Plot (Sample {sample_idx})')
plt.tight_layout()
plt.show()

# --------------------------Correlation analysis between Delta AUC and SHAP--------------------------
# Merge the two importance metrics.
corr_df = perm_importance[['Feature', 'ΔAUC']].merge(
    shap_importance, on='Feature'
)

# Correlation heatmap.
plt.figure(figsize=(8, 6))
corr = corr_df[['ΔAUC', 'Mean |SHAP|']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between ΔAUC and Mean |SHAP|')
plt.tight_layout()
plt.show()

# Scatter plot.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='ΔAUC', y='Mean |SHAP|', hue='Feature', data=corr_df, s=100)
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('ΔAUC (Permutation Importance)')
plt.ylabel('Mean |SHAP Value|')
plt.title('Feature Importance: ΔAUC vs SHAP')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------------ROC curve--------------------------
fpr, tpr, _ = metrics.roc_curve(test_y, ypred)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# --------------------------Save results--------------------------
# Save prediction results and threshold-analysis results.
threshold_analysis = pd.DataFrame({
    'Threshold': thresholds,
    'Recall': recall_scores,
    'Precision': precision_scores,
    'F1-score': f1_scores
})

test_result = pd.DataFrame({
    'Actual': test_y,
    'Predicted Probability': ypred,
    'Predicted Class': y_pred
}).reset_index(drop=True)

# Save all results.
test_result.to_csv('prediction_results.csv', index=False)
threshold_analysis.to_csv('threshold_analysis.csv', index=False)
perm_importance.to_csv('permutation_importance.csv', index=False)
shap_importance.to_csv('shap_importance.csv', index=False)

print("\nAll results saved successfully!")
