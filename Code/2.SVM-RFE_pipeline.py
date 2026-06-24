import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    precision_recall_curve, multilabel_confusion_matrix, 
    hamming_loss, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)

# Set global plotting configurations for editable vector graphics.
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['svg.fonttype'] = 'none'

# Define evaluation metrics with 95% CI and biological correction rules.
def calculate_metrics_with_ci(y_true, y_pred, y_prob=None, n_bootstraps=1000, alpha=0.05):
    point_estimates = {
        'Exact_Match_Ratio': accuracy_score(y_true, y_pred),
        'Hamming_Loss': hamming_loss(y_true, y_pred),
        'Micro_Precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'Micro_Recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'Micro_F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Macro_Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    if y_prob is not None:
        point_estimates['Macro_AUC'] = roc_auc_score(y_true, y_prob, average='macro')

    n_samples = len(y_true)
    rng = np.random.RandomState(42)
    bootstrapped_scores = {k: [] for k in point_estimates.keys()}

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        y_t, y_p = y_true[idx], y_pred[idx]
        
        bootstrapped_scores['Exact_Match_Ratio'].append(accuracy_score(y_t, y_p))
        bootstrapped_scores['Hamming_Loss'].append(hamming_loss(y_t, y_p))
        bootstrapped_scores['Micro_Precision'].append(precision_score(y_t, y_p, average='micro', zero_division=0))
        bootstrapped_scores['Micro_Recall'].append(recall_score(y_t, y_p, average='micro', zero_division=0))
        bootstrapped_scores['Micro_F1'].append(f1_score(y_t, y_p, average='micro', zero_division=0))
        bootstrapped_scores['Macro_Precision'].append(precision_score(y_t, y_p, average='macro', zero_division=0))
        bootstrapped_scores['Macro_Recall'].append(recall_score(y_t, y_p, average='macro', zero_division=0))
        bootstrapped_scores['Macro_F1'].append(f1_score(y_t, y_p, average='macro', zero_division=0))

        if y_prob is not None:
            try:
                bootstrapped_scores['Macro_AUC'].append(roc_auc_score(y_t, y_prob[idx], average='macro'))
            except ValueError:
                pass 

    final_results = {}
    ordered_keys = ['Exact_Match_Ratio', 'Macro_AUC'] if y_prob is not None else ['Exact_Match_Ratio']
    ordered_keys += [k for k in point_estimates.keys() if k not in ordered_keys]
    
    for k in ordered_keys:
        v = point_estimates[k]
        if len(bootstrapped_scores[k]) > 0:
            lower = np.percentile(bootstrapped_scores[k], 100 * (alpha / 2))
            upper = np.percentile(bootstrapped_scores[k], 100 * (1 - alpha / 2))
            final_results[k] = f"{v:.4f} (95% CI: {lower:.4f}-{upper:.4f})"
        else:
            final_results[k] = f"{v:.4f} (95% CI: N/A)"
            
    return final_results

def apply_biological_correction(y_true_bin, y_pred_bin, unique_classes):
    y_pred_eval = y_pred_bin.copy()
    if 'MB' not in unique_classes:
        return y_pred_eval
        
    mb_idx = list(unique_classes).index('MB')
    vs_idx = list(unique_classes).index('VS') if 'VS' in unique_classes else -1
    pb_idx = list(unique_classes).index('PB') if 'PB' in unique_classes else -1
    
    for i in range(len(y_true_bin)):
        if y_true_bin[i, mb_idx] == 1 and y_pred_eval[i, mb_idx] == 1:
            if vs_idx != -1 and y_true_bin[i, vs_idx] == 0 and y_pred_eval[i, vs_idx] == 1:
                y_pred_eval[i, vs_idx] = 0
            if pb_idx != -1 and y_true_bin[i, pb_idx] == 0 and y_pred_eval[i, pb_idx] == 1:
                y_pred_eval[i, pb_idx] = 0
                
    return y_pred_eval

# Load datasets and apply CPM+Log2 normalization.
os.makedirs('results', exist_ok=True)
train_df = pd.read_excel('train_set_70.xlsx')
test_df = pd.read_excel('test_set_30.xlsx')
sim_df = pd.read_csv('simulated_mixture_data.csv') 

gene_cols = train_df.columns[2:].tolist()
unique_types = np.sort(train_df['type'].unique())

def preprocess(X):
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return np.log2((X / sums) * 1e6 + 1)

X_train_merged = np.vstack([preprocess(train_df[gene_cols].values), sim_df[gene_cols].values])
y_train_orig_bin = MultiLabelBinarizer(classes=unique_types).fit_transform([[t] for t in train_df['type']])
y_train_final = np.vstack([y_train_orig_bin, sim_df[[f'Label_{t}' for t in unique_types]].values])

X_test_orig_norm = preprocess(test_df[gene_cols].values)
y_test_orig_bin = MultiLabelBinarizer(classes=unique_types).fit_transform([[t] for t in test_df['type']])

scaler = MinMaxScaler()
X_train_final = scaler.fit_transform(X_train_merged)
X_test_final = scaler.transform(X_test_orig_norm)

# Train SVM-RFE model and optimize hyperparameters using GridSearchCV.
print("Running SVM-RFE GridSearch...")
pipeline = Pipeline([
    ('rfe', RFE(estimator=SVC(kernel='linear', class_weight='balanced', random_state=42), step=0.05)),
    ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
])

param_grid = {
    'estimator__rfe__n_features_to_select': [0.3, 0.5, 0.8],
    'estimator__svm__C': [0.1, 1, 10],
    'estimator__svm__kernel': ['linear', 'rbf']
}

base_model = MultiOutputClassifier(pipeline)
grid_search = GridSearchCV(base_model, param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_final, y_train_final)
best_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score (f1_macro): {grid_search.best_score_:.4f}")

# Determine optimal decision thresholds using 10-fold cross-validation.
kf = KFold(n_splits=10, shuffle=True, random_state=42)
y_probs_cv = np.zeros(y_train_final.shape)

for tr_cv, val_cv in kf.split(X_train_final):
    cv_model = clone(best_model).fit(X_train_final[tr_cv], y_train_final[tr_cv])
    probs = cv_model.predict_proba(X_train_final[val_cv])
    for i in range(len(unique_types)):
        y_probs_cv[val_cv, i] = probs[i][:, 1] if probs[i].shape[1] == 2 else probs[i][:, 0]

best_thresholds = []
print("\nOptimal Thresholds:")
for i, category in enumerate(unique_types):
    p, r, t = precision_recall_curve(y_train_final[:, i], y_probs_cv[:, i])
    f1 = 2 * (p * r) / (p + r + 1e-8)
    best_t = t[np.argmax(f1[:-1])] if len(t) > 0 else 0.5
    best_thresholds.append(best_t)
    print(f" - {category}: {best_t:.4f}")

# Evaluate model performance on the independent test set.
print("\nEvaluating on independent test set...")
test_probs = best_model.predict_proba(X_test_final)
test_probs_matrix = np.array([p[:, 1] if p.shape[1] == 2 else np.zeros(p.shape[0]) for p in test_probs]).T

y_test_pred_raw = (test_probs_matrix >= np.array(best_thresholds)).astype(int)
y_test_pred_eval = apply_biological_correction(y_test_orig_bin, y_test_pred_raw, unique_types)

metrics_with_ci = calculate_metrics_with_ci(y_test_orig_bin, y_test_pred_eval, y_prob=test_probs_matrix) 
print("Test Set Metrics:")
for metric, val in metrics_with_ci.items():
    print(f" - {metric}: {val}")

# Evaluate on external validation set if available.
external_val_file = 'external_validation_set.xlsx'

if os.path.exists(external_val_file):
    print("\nEvaluating on external validation set...")
    df_actual = pd.read_excel(external_val_file)
    X_actual_scaled = scaler.transform(preprocess(df_actual[gene_cols].values)) 
    
    actual_probs = best_model.predict_proba(X_actual_scaled)
    actual_probs_matrix = np.array([p[:, 1] if p.shape[1] == 2 else np.zeros(p.shape[0]) for p in actual_probs]).T
    y_actual_pred_raw = (actual_probs_matrix >= np.array(best_thresholds)).astype(int)
    
    if 'type' in df_actual.columns:
        y_actual_list = [[p for p in re.split(r'[+,\s;]+', t.strip()) if p in unique_types] for t in df_actual['type'].astype(str)]
        y_actual_bin = MultiLabelBinarizer(classes=unique_types).fit_transform(y_actual_list)
        
        y_actual_pred_eval = apply_biological_correction(y_actual_bin, y_actual_pred_raw, unique_types)
        actual_metrics = calculate_metrics_with_ci(y_actual_bin, y_actual_pred_eval, y_prob=actual_probs_matrix) 
        print("External Validation Metrics:")
        for metric, val in actual_metrics.items():
            print(f" - {metric}: {val}")

# Perform class-specific SHAP feature importance analysis.
print("\nRunning class-specific SHAP analysis...")
np.random.seed(42)
sample_size = min(100, X_test_final.shape[0])
sample_indices = np.random.choice(X_test_final.shape[0], size=sample_size, replace=False)

X_explain_full = X_test_final[sample_indices]
X_train_full = X_train_final

for i, fluid in enumerate(unique_types):
    print(f" - Calculating SHAP values for {fluid}...")
    
    rfe_step = best_model.estimators_[i].named_steps['rfe']
    svm_step = best_model.estimators_[i].named_steps['svm']
    
    support_mask = rfe_step.support_
    selected_feature_names = np.array(gene_cols)[support_mask].tolist()
    
    X_explain_subset = X_explain_full[:, support_mask]
    X_train_subset = X_train_full[:, support_mask]
    
    X_background_subset = shap.kmeans(X_train_subset, 20)
    
    def predict_fluid_prob_subset(X):
        return svm_step.predict_proba(X)[:, 1]
    
    explainer = shap.KernelExplainer(predict_fluid_prob_subset, X_background_subset)
    shap_values = explainer.shap_values(X_explain_subset, nsamples="auto") 
    
    plt.figure(figsize=(8, 6))
    
    shap.summary_plot(
        shap_values, 
        features=X_explain_subset, 
        feature_names=selected_feature_names, 
        max_display=15,
        show=False
    )
    
    plt.title(f"SHAP Feature Impact: {fluid}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    output_filename = f'results/SVM_SHAP_Summary_{fluid}_Optimal15.svg'
    plt.savefig(output_filename, format='svg', dpi=600, bbox_inches='tight')
    plt.close() 

print("\nPipeline execution completed successfully.")