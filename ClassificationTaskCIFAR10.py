import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

DATA_DIR = r"C:\Users\Ari\PycharmProjects\school\intro-to-ai-compendium\part2_classification\cifar-10-batches-py"
FIGURES_DIR = r"C:\Users\Ari\PycharmProjects\school\intro-to-ai-compendium\part2_classification\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


# Load data
batches = [unpickle(f'{DATA_DIR}\\data_batch_{i}') for i in range(1, 6)]
X_train_full = np.concatenate([b[b'data'] for b in batches])
y_train_full = np.concatenate([b[b'labels'] for b in batches])

test = unpickle(f'{DATA_DIR}\\test_batch')
X_test_raw = test[b'data'].astype(np.float32)
y_test = np.array(test[b'labels'])

meta = unpickle(f'{DATA_DIR}\\batches.meta')
class_names = [c.decode() for c in meta[b'label_names']]

# Split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_full.astype(np.float32), y_train_full,
    test_size=0.1, random_state=42, stratify=y_train_full
)

# Normalization (train stats only)
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0) + 1e-8

X_train = (X_train_raw - mean) / std
X_val = (X_val_raw - mean) / std

# PCA (fit on train only)
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Hyperparameter grids
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
k_values = [1, 3, 5, 7, 10, 15, 20]

# Logistic Regression
print("\nTuning Logistic Regression (C)...")
lr_val_accs = []
for C in C_values:
    m = LogisticRegression(solver='lbfgs', C=C, max_iter=2000, random_state=42, n_jobs=-1)
    m.fit(X_train_pca, y_train)
    lr_val_accs.append(accuracy_score(y_val, m.predict(X_val_pca)))
    print(f"  C={C}: {lr_val_accs[-1]:.4f}")

# Pick smallest C among ties: prefer stronger regularization when accuracy is equal
best_C_lr = min(c for c, a in zip(C_values, lr_val_accs) if a == max(lr_val_accs))
print(f"  Best C: {best_C_lr}")

# Linear SVM
print("\nTuning Linear SVM (C)...")
svm_val_accs = []
for C in C_values:
    m = LinearSVC(C=C, dual=False, max_iter=5000, random_state=42)
    m.fit(X_train_pca, y_train)
    svm_val_accs.append(accuracy_score(y_val, m.predict(X_val_pca)))
    print(f"  C={C}: {svm_val_accs[-1]:.4f}")

best_C_svm = min(c for c, a in zip(C_values, svm_val_accs) if a == max(svm_val_accs))
print(f"  Best C: {best_C_svm}")

# KNN
print("\nTuning KNN (k)...")
knn_val_accs = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    m.fit(X_train_pca, y_train)
    knn_val_accs.append(accuracy_score(y_val, m.predict(X_val_pca)))
    print(f"  k={k}: {knn_val_accs[-1]:.4f}")

best_k = min(k for k, a in zip(k_values, knn_val_accs) if a == max(knn_val_accs))
print(f"  Best k: {best_k}")

# Hyperparameter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(C_values, lr_val_accs, marker='o')
axes[0].axvline(best_C_lr, color='red', linestyle='--', label=f'Best C={best_C_lr}')
axes[0].set_xscale('log')
axes[0].set_xlabel('C')
axes[0].set_ylabel('Validation accuracy')
axes[0].set_title('Logistic Regression — C tuning')
axes[0].legend()

axes[1].plot(C_values, svm_val_accs, marker='o')
axes[1].axvline(best_C_svm, color='red', linestyle='--', label=f'Best C={best_C_svm}')
axes[1].set_xscale('log')
axes[1].set_xlabel('C')
axes[1].set_ylabel('Validation accuracy')
axes[1].set_title('Linear SVM — C tuning')
axes[1].legend()

axes[2].plot(k_values, knn_val_accs, marker='o')
axes[2].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Validation accuracy')
axes[2].set_title('KNN — k tuning')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}\\hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.close()

# Best models
best_lr = LogisticRegression(solver='lbfgs', C=best_C_lr, max_iter=2000, random_state=42, n_jobs=-1)
best_svm = LinearSVC(C=best_C_svm, dual=False, max_iter=5000, random_state=42)
best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)

best_lr.fit(X_train_pca, y_train)
best_svm.fit(X_train_pca, y_train)
best_knn.fit(X_train_pca, y_train)

y_pred_lr = best_lr.predict(X_val_pca)
y_pred_svm = best_svm.predict(X_val_pca)
y_pred_knn = best_knn.predict(X_val_pca)

lr_val_acc = accuracy_score(y_val, y_pred_lr)
svm_val_acc = accuracy_score(y_val, y_pred_svm)
knn_val_acc = accuracy_score(y_val, y_pred_knn)

print("\n--- Validation Accuracy ---")
print(f"Logistic Regression (C={best_C_lr}):  {lr_val_acc:.4f}")
print(f"Linear SVM          (C={best_C_svm}): {svm_val_acc:.4f}")
print(f"KNN                 (k={best_k}):     {knn_val_acc:.4f}")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_svm, y_pred_knn],
        ['Logistic Regression', 'Linear SVM', f'KNN (k={best_k})']
):
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}\\confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# Select best model
val_accs = [lr_val_acc, svm_val_acc, knn_val_acc]
names = ['Logistic Regression', 'Linear SVM', f'KNN (k={best_k})']
best_idx = int(np.argmax(val_accs))
best_name = names[best_idx]
print(f"\nBest model: {best_name} ({val_accs[best_idx]:.4f})")

#  Final model: fresh instance, retrained on train+val
X_full_raw = np.concatenate([X_train_raw, X_val_raw])
y_full = np.concatenate([y_train, y_val])

mean_final = X_full_raw.mean(axis=0)
std_final = X_full_raw.std(axis=0) + 1e-8
X_full = (X_full_raw - mean_final) / std_final
X_test_final = (X_test_raw - mean_final) / std_final

pca_final = PCA(n_components=100, random_state=42)
X_full_pca = pca_final.fit_transform(X_full)
X_test_pca_final = pca_final.transform(X_test_final)
print(f"Final PCA explained variance: {pca_final.explained_variance_ratio_.sum():.3f}")

if best_idx == 0:
    final_model = LogisticRegression(solver='lbfgs', C=best_C_lr, max_iter=2000, random_state=42, n_jobs=-1)
elif best_idx == 1:
    final_model = LinearSVC(C=best_C_svm, dual=False, max_iter=5000, random_state=42)
else:
    final_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)

final_model.fit(X_full_pca, y_full)

test_acc = accuracy_score(y_test, final_model.predict(X_test_pca_final))
print(f"Final test accuracy ({best_name}): {test_acc:.4f}")
