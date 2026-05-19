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

# --------------------------基础设置--------------------------
# 字体设置（全英文避免中文问题）
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 自定义色带（绿-黄-红）
colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]
green_red_cmap = mcolors.LinearSegmentedColormap.from_list("green_red_cmap", colors)

# --------------------------数据准备--------------------------
# 导入数据集
df = pd.read_csv('taget - qinshi.csv', encoding='GBK')
data = df.iloc[:, 1:-1]  # 特征
target = df.iloc[:, -1]  # 标签

# 划分训练集和测试集（保持类别平衡）
train_x, test_x, train_y, test_y = train_test_split(
    data, target, test_size=0.3, random_state=10, stratify=target
)
feature_names = data.columns.tolist()  # 特征名称列表

# --------------------------模型训练--------------------------
# 构建DMatrix
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# XGBoost参数设置
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

# 训练模型（带早停机制）
bst = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# --------------------------模型评估--------------------------
# 预测与阈值设置
ypred = bst.predict(dtest)  # 概率预测
y_pred = (ypred >= 0.5) * 1  # 类别预测

# 输出评估指标
print("\n=== Model Evaluation ===")
print(f'Precision: {metrics.precision_score(test_y, y_pred):.4f}')
print(f'Recall: {metrics.recall_score(test_y, y_pred):.4f}')
print(f'F1-score: {metrics.f1_score(test_y, y_pred):.4f}')
print(f'Accuracy: {metrics.accuracy_score(test_y, y_pred):.4f}')
print(f'AUC: {metrics.roc_auc_score(test_y, ypred):.4f}')

# --------------------------新增：召回率曲线及阈值分析--------------------------
# 计算不同阈值下的召回率、精确率和F1分数
thresholds = np.linspace(0, 1, 100)
recall_scores = []
precision_scores = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (ypred >= threshold) * 1
    recall_scores.append(metrics.recall_score(test_y, y_pred_threshold))
    precision_scores.append(metrics.precision_score(test_y, y_pred_threshold))
    f1_scores.append(metrics.f1_score(test_y, y_pred_threshold))

# 图1：召回率曲线（Recall Curve）
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

# 图2：召回率-精确率-阈值综合曲线
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

# 图3：召回率-精确率曲线（PR Curve）
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

# --------------------------置换重要性分析（ΔAUC）--------------------------
original_auc = metrics.roc_auc_score(test_y, ypred)
print(f'\nOriginal test set AUC: {original_auc:.4f}')


# 计算置换重要性
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
    delta_auc = original_auc - mean_perm_auc  # 可能包含负值
    return pd.DataFrame({
        'Feature': feature_names,
        'ΔAUC': delta_auc,
        'Mean Permuted AUC': mean_perm_auc
    }).sort_values(by='ΔAUC', ascending=False)


# 计算并输出置换重要性结果
perm_importance = permutation_importance_auc(bst, test_x, test_y, feature_names)
print("\n=== Permutation Importance (ΔAUC) ===")
print(perm_importance)


# 计算ΔAUC标准差（稳定性指标）
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

# --------------------------ΔAUC可视化--------------------------
# 图1：带误差线的ΔAUC条形图
plt.figure(figsize=(10, 6))
bar_colors = ['skyblue' if x >= 0 else 'salmon' for x in perm_importance['ΔAUC']]
plt.barh(perm_importance['Feature'], perm_importance['ΔAUC'], color=bar_colors)
plt.errorbar(
    x=perm_importance['ΔAUC'],
    y=range(len(perm_importance)),
    xerr=perm_importance['ΔAUC Std'],
    fmt='none', ecolor='black', capsize=5
)
plt.axvline(x=0, color='black', linestyle='--')  # 零线
plt.xlabel('ΔAUC (AUC Decrease)')
plt.ylabel('Features')
plt.title('Permutation Importance with Stability (ΔAUC ± Std)')
plt.gca().invert_yaxis()  # 从高到低排序
plt.tight_layout()
plt.show()

# 图2：原始AUC vs 置换后AUC对比
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

# 图3：特征相对贡献度
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

# --------------------------SHAP分析--------------------------
# 计算SHAP值
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(test_x)
expected_value = explainer.expected_value

# 计算SHAP重要性
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)

print("\n=== SHAP-based Feature Importance ===")
print(shap_importance)

# --------------------------SHAP可视化--------------------------
# 图1：SHAP摘要图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_x, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show()

# 图2：Top4特征单个依赖图
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

# 图3：4x4特征交互矩阵图（核心功能：所有Top4特征两两比较）
if len(top4_features) == 4:
    fig, axs = plt.subplots(4, 4, figsize=(24, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, main_feature in enumerate(top4_features):
        for j, interact_feature in enumerate(top4_features):
            # 绘制交互依赖图
            shap.dependence_plot(
                main_feature, shap_values, test_x,
                interaction_index=interact_feature,
                ax=axs[i, j], show=False,
                dot_size=15,  x_jitter=0.1
            )

            # 美化子图
            axs[i, j].set_title(f'{main_feature} vs {interact_feature}', fontsize=10)
            axs[i, j].set_xlabel(main_feature, fontsize=8)
            axs[i, j].set_ylabel(f'SHAP value of {main_feature}', fontsize=8)
            axs[i, j].tick_params(axis='both', labelsize=6)

            # 对角线标注
            if i == j:
                axs[i, j].text(0.5, 0.5, 'Same feature',
                               ha='center', va='center',
                               fontsize=12, color='gray', fontweight='bold')

    fig.suptitle('SHAP Interaction Matrix: Pairwise Comparison of Top 4 Features',
                 fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# 图4：单个样本Waterfall图
sample_idx = 0  # 选择第一个样本
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

# --------------------------ΔAUC与SHAP相关性分析--------------------------
# 合并两种重要性指标
corr_df = perm_importance[['Feature', 'ΔAUC']].merge(
    shap_importance, on='Feature'
)

# 相关性热图
plt.figure(figsize=(8, 6))
corr = corr_df[['ΔAUC', 'Mean |SHAP|']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between ΔAUC and Mean |SHAP|')
plt.tight_layout()
plt.show()

# 散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='ΔAUC', y='Mean |SHAP|', hue='Feature', data=corr_df, s=100)
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('ΔAUC (Permutation Importance)')
plt.ylabel('Mean |SHAP Value|')
plt.title('Feature Importance: ΔAUC vs SHAP')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------------ROC曲线--------------------------
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

# --------------------------保存结果--------------------------
# 保存预测结果和阈值分析
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

# 保存所有结果
test_result.to_csv('prediction_results.csv', index=False)
threshold_analysis.to_csv('threshold_analysis.csv', index=False)
perm_importance.to_csv('permutation_importance.csv', index=False)
shap_importance.to_csv('shap_importance.csv', index=False)

print("\nAll results saved successfully!")

# 加载新的CSV文件
new_df = pd.read_csv("input_all - 副本.csv", encoding='UTF-8')

# 提取特征数据并进行预测
new_data = new_df.iloc[:, 1:]  # 调整列索引以匹配您的数据集结构
dnew = xgb.DMatrix(new_data)
ypred_new = bst.predict(dnew)

# 设置阈值、评价指标
new_y_pred = (ypred_new >= 0.5) * 1

# 将预测结果写入新的DataFrame并保存到文件
new_test_result = pd.DataFrame({'prediction': ypred_new})
new_test_result = new_test_result.reset_index(drop=False)
new_test_result.to_csv('textXGresult3.txt', index=False)
