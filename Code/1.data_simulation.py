import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw dataset and extract feature columns and unique fluid types.
# Note: Ensure the raw data file is named 'modeling_data.xlsx' in the root directory.
file_path = 'modeling_data.xlsx'
df = pd.read_excel(file_path)

gene_cols = df.columns[2:].tolist()
unique_types = np.sort(df['type'].unique())

# Perform stratified 70/30 train-test split to preserve initial class distributions.
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df['type'], random_state=42
)

train_df.to_excel('train_set_70.xlsx', index=False)
test_df.to_excel('test_set_30.xlsx', index=False)
print("Train and test datasets saved successfully.")

# Define CPM+Log2 normalization function and preprocess the training data.
def preprocess(X):
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return np.log2((X / sums) * 1e6 + 1)

X_train_raw = train_df[gene_cols].values
y_train = train_df['type'].values
names_train = train_df['sample'].values
X_train_norm = preprocess(X_train_raw)

# Define simulation function to generate complex mixtures with biological noise and dropout.
def generate_exhaustive_mixtures(X_train, y_train_labels, names, types, samples_per_scenario=5):
    sim_X, sim_y, sim_info = [], [], []
    type_to_pool_idx = {t: np.where(y_train_labels == t)[0] for t in types}
    
    scenarios = [
        (2, [[1, 1], [1, 10], [10, 1], [1, 20], [20, 1], [1, 30], [30, 1]]), 
        (3, [[1, 1, 1]]), 
        (4, [[1, 1, 1, 1]]), 
        (5, [[1, 1, 1, 1, 1]])
    ]
    
    for k, ratios in scenarios:
        for combo in itertools.combinations(types, k):
            for r in ratios:
                for _ in range(samples_per_scenario):
                    selected_pool_indices = [np.random.choice(type_to_pool_idx[t]) for t in combo]
                    weights = np.array(r) / sum(r)
                    
                    mixed_expr = np.sum(X_train[selected_pool_indices] * weights[:, np.newaxis], axis=0)
                    
                    noise = np.random.normal(0, 0.05, size=mixed_expr.shape)
                    mixed_expr = mixed_expr * (1 + noise)
                    mixed_expr[mixed_expr < 0.1] = 0 
                    
                    labels = np.zeros(len(types))
                    for t in combo: 
                        labels[list(types).index(t)] = 1
                    
                    comp = "+".join([f"{names[selected_pool_indices[i]]}({round(weights[i],2)})" for i in range(k)])
                    
                    sim_X.append(mixed_expr)
                    sim_y.append(labels)
                    sim_info.append(comp)
                    
    return np.array(sim_X), np.array(sim_y), sim_info

# Execute mixture simulation and export the generated dataset.
np.random.seed(42)
X_sim, y_sim, info_sim = generate_exhaustive_mixtures(
    X_train_norm, y_train, names_train, unique_types, samples_per_scenario=10
)

sim_df = pd.DataFrame(X_sim, columns=gene_cols)
sim_df.insert(0, 'Composition', info_sim)
for i, t in enumerate(unique_types): 
    sim_df[f'Label_{t}'] = y_sim[:, i]

sim_df.to_csv('simulated_mixture_data.csv', index=False)
print("Simulated mixture dataset generated and saved successfully.")