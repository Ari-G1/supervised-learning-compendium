import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ucimlrepo import fetch_ucirepo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def fig_path(name):
    return os.path.join(FIGURES_DIR, name)


# Helpers

def eval_metrics(y_true, y_pred):
    """Computes MSE, RMSE, MAE, and R²."""
    mse = mean_squared_error(y_true, y_pred)
    return {'MSE': round(mse, 4),
            'RMSE': round(np.sqrt(mse), 4),
            'MAE': round(mean_absolute_error(y_true, y_pred), 4),
            'R2': round(r2_score(y_true, y_pred), 4)}


def fit_eval(model, X_tr, y_tr, X_val, y_val):
    """Fit model, return (train_rmse, val_rmse, train_r2, val_r2)."""
    model.fit(X_tr, y_tr)
    tr = eval_metrics(y_tr, model.predict(X_tr))
    val = eval_metrics(y_val, model.predict(X_val))
    return tr['RMSE'], val['RMSE'], tr['R2'], val['R2']


def plot_pred_vs_true(pairs, suptitle, path):
    """Figure of predicted vs ground truth scatter plots, each panel shows the perfect-prediction diagonal and annotates
    RMSE and R²."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (y_true, y_pred, label) in zip(axes, pairs):
        m = eval_metrics(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.5, color='steelblue', s=25, edgecolors='none')
        lo = min(y_true.min(), y_pred.min()) - 1
        hi = max(y_true.max(), y_pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
        ax.set(xlim=(lo, hi), ylim=(lo, hi),
               xlabel='Ground Truth (MPG)', ylabel='Predicted (MPG)',
               title=f'{label}  |  RMSE={m["RMSE"]:.2f}  R²={m["R2"]:.3f}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)
    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path(path), dpi=150, bbox_inches='tight')


def plot_val_curve(x_vals, tr_rmse, val_rmse, tr_r2, val_r2,
                   xlabel, title, best_x, path):
    """Two-panel validation curve: RMSE and R²."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tr, val, ylabel) in zip(axes, [(tr_rmse, val_rmse, 'RMSE (MPG)'),
                                            (tr_r2, val_r2, 'R²')]):
        ax.plot(x_vals, tr, 'o-', color='steelblue', label=f'Train {ylabel}')
        ax.plot(x_vals, val, 's-', color='crimson', label=f'Val {ylabel}')
        ax.axvline(best_x, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Best = {best_x}')
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{title} – {ylabel}')
        ax.legend()
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(fig_path(path), dpi=150, bbox_inches='tight')


# ── 2.1  EXPLORATORY DATA ANALYSIS ───────────────────────────────────────────

auto_mpg = fetch_ucirepo(id=9)
df = pd.concat([auto_mpg.data.ids, auto_mpg.data.features, auto_mpg.data.targets], axis=1)
df = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
         'acceleration', 'model_year', 'origin', 'car_name']]
df = df.replace('?', np.nan)
df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']] = \
    df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']].apply(
        pd.to_numeric, errors='coerce')
print("\n── 2.1 EXPLORATORY DATA ANALYSIS ───────────────────────────────────────────")
print("Missing Values:")
print(df[df.isnull().any(axis=1)][['car_name', 'model_year', 'horsepower']])
print("Some Stats about the Data:")
print(df['mpg'].describe())

# MPG distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['mpg'], kde=True, color='royalblue', bins=15, edgecolor='white')
plt.axvline(df['mpg'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f"Mean:   {df['mpg'].mean():.1f}")
plt.axvline(df['mpg'].median(), color='green', linestyle='--', linewidth=1.5,
            label=f"Median: {df['mpg'].median():.1f}")
plt.xlabel('Miles Per Gallon (MPG)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.4)
sns.despine()
plt.tight_layout()
plt.savefig(fig_path('mpg_dist.png'), dpi=150, bbox_inches='tight')

# Feature–target correlations
feat_cols = ['cylinders', 'displacement', 'horsepower', 'weight',
             'acceleration', 'model_year', 'origin']
corr = df.corr(numeric_only=True)['mpg'].drop('mpg').sort_values(ascending=False)
plt.figure(figsize=(10, 6))
col_colors = ['red' if x < 0 else 'green' for x in corr.values]
bars = plt.barh(corr.index, corr.values, color=col_colors)
for bar, val in zip(bars, corr.values):
    plt.text(val + (0.02 if val >= 0 else -0.02),
             bar.get_y() + bar.get_height() / 2, f'{val:.2f}',
             va='center', ha='left' if val >= 0 else 'right', fontsize=10)
plt.xlabel('Pearson Correlation Coefficient')
plt.axvline(0, color='black', lw=1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig(fig_path('featureandMPG_corr.png'), dpi=150, bbox_inches='tight')

# Inter-feature heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df[feat_cols].corr(), dtype=bool))
sns.heatmap(df[feat_cols].corr(), annot=True, mask=mask,
            cmap='RdBu_r', center=0, fmt=".2f")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(fig_path('feature_corr.png'), dpi=150, bbox_inches='tight')

# ── 2.2  DATASET AND PREPROCESSING ───────────────────────────────────────────────────────

df = df.drop(columns=['car_name'])

# Origin correlation by group (justifying one-hot encoding)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (oval, oname) in zip(axes, [(1, 'American'), (2, 'European'), (3, 'Asian')]):
    sub = df[df['origin'] == oval]
    c = (sub.corr(numeric_only=True)['mpg']
         .drop(['mpg', 'origin']).sort_values(ascending=False))
    ax.barh(c.index, c.values, color=['red' if x < 0 else 'green' for x in c.values])
    ax.set_title(f'{oname} (n={len(sub)})')
    ax.axvline(0, color='black', lw=1)
    ax.set_xlim(-1.1, 1.1)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pearson Correlation Coefficient')
axes[0].set_ylabel('Feature')
sns.despine()
plt.tight_layout()
plt.savefig(fig_path('origin_correlations.png'), dpi=150, bbox_inches='tight')

df_eda = df.copy()
# All models will  use one-hot-encoded origin_2 (European) / origin_3 (Asian).
df = pd.get_dummies(df, columns=['origin'], drop_first=True, dtype=int)

X = df.drop(columns=['mpg'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Missing values — median from X_train only
hp_median = X_train['horsepower'].median()
X_train.loc[:, 'horsepower'] = X_train['horsepower'].fillna(hp_median)
X_test.loc[:, 'horsepower'] = X_test['horsepower'].fillna(hp_median)

# Outer scaler: fit on full X_train (only used for 2.7 test evaluation)
outer_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(outer_scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(outer_scaler.transform(X_test), columns=X.columns)

# Inner scaler: fit on X_tr only (prevents validation leakage during 2.4–2.6)
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1)
inner_scaler = StandardScaler()
X_tr = pd.DataFrame(inner_scaler.fit_transform(X_tr_raw), columns=X.columns)
X_val = pd.DataFrame(inner_scaler.transform(X_val_raw), columns=X.columns)

print("\n── 2.2 PREPROCESSING ───────────────────────────────────────────")
print(f"Sizes — train: {len(X_tr)}  val: {len(X_val)}  test: {len(X_test_scaled)}")
print("Features:", list(X.columns))

# ── 2.3  LINEAR REGRESSION BASELINE ──────────────────────────────────────────
# Fit — train/val only,  test set appears only once (in 2.7)
lr_baseline = LinearRegression()
lr_baseline.fit(X_tr, y_tr)

plot_pred_vs_true(
    [(y_tr, lr_baseline.predict(X_tr), 'Train'),
     (y_val, lr_baseline.predict(X_val), 'Validation')],
    'Linear Regression – Predicted vs Ground Truth', 'lr_pred_vs_true.png')

tr_m = eval_metrics(y_tr, lr_baseline.predict(X_tr))
val_m = eval_metrics(y_val, lr_baseline.predict(X_val))
print("\n── 2.3 Linear Regression Baseline───────────────────────────────────────────")

# Feature–target scatter plots
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
for i, col in enumerate(feat_cols):
    ax = axes[i]
    ax.scatter(df_eda[col], df_eda['mpg'], alpha=0.4, color='steelblue',
               s=18, edgecolors='none')
    mask_notna = df_eda[col].notna()
    coef, intercept = np.polyfit(
        df_eda.loc[mask_notna, col], df_eda.loc[mask_notna, 'mpg'], 1)
    x_range = np.linspace(df_eda[col].min(), df_eda[col].max(), 200)
    ax.plot(x_range, coef * x_range + intercept, color='crimson', linewidth=1.5)
    r = df_eda[[col, 'mpg']].corr().iloc[0, 1]
    ax.set_title(f'mpg vs {col}  (r={r:.2f})', fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel('mpg')
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)
axes[-1].set_visible(False)
plt.suptitle('Feature–Target Relationships (MPG)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(fig_path('feature_target_scatter.png'), dpi=150, bbox_inches='tight')

# Residual analysis
y_pred_tr = lr_baseline.predict(X_tr)
y_pred_val = lr_baseline.predict(X_val)
resid_tr = np.array(y_tr) - y_pred_tr
resid_val = np.array(y_val) - y_pred_val

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_pred_tr, resid_tr, alpha=0.5, color='steelblue',
                s=25, edgecolors='none')
axes[0].axhline(0, color='crimson', linestyle='--', linewidth=1.5)
axes[0].set(xlabel='Predicted (MPG)', ylabel='Residual',
            title='Residuals vs Predicted (Train)')
axes[0].grid(alpha=0.3)
sns.despine(ax=axes[0])

axes[1].scatter(y_pred_val, resid_val, alpha=0.5, color='steelblue',
                s=25, edgecolors='none')
axes[1].axhline(0, color='crimson', linestyle='--', linewidth=1.5)
axes[1].set(xlabel='Predicted (MPG)', ylabel='Residual',
            title='Residuals vs Predicted (Validation)')
axes[1].grid(alpha=0.3)
sns.despine(ax=axes[1])

all_resid = np.concatenate([resid_tr, resid_val])
sns.histplot(all_resid, kde=True, color='steelblue', bins=20,
             edgecolor='white', ax=axes[2])
axes[2].axvline(0, color='crimson', linestyle='--', linewidth=1.5)
axes[2].set(xlabel='Residual', ylabel='Frequency',
            title='Residual Distribution')
axes[2].grid(axis='y', alpha=0.3)
sns.despine(ax=axes[2])

plt.suptitle('Linear Regression – Residual Analysis', fontsize=13)
plt.tight_layout()
plt.savefig(fig_path('lr_residuals.png'), dpi=150, bbox_inches='tight')

print(f"  {'':6}  {'Train':>8}  {'Val':>8}")
for k in ['RMSE', 'MAE', 'R2']:
    print(f"  {k:<6}  {tr_m[k]:>8.4f}  {val_m[k]:>8.4f}")
print(f"  B–V gap  {val_m['RMSE'] - tr_m['RMSE']:>8.4f}")

# ── 2.4  POLYNOMIAL REGRESSION AND MODEL COMPLEXITY ──────────────────────────
# Pipeline: PolynomialFeatures → StandardScaler → LinearRegression.
# Scaling is required after expansion to maintain numerical stability.
# Feature count grows rapidly with degree, increasing variance and multicollinearity.
# LinearRegression is used as required.

degrees = range(1, 8)
tr_rmse_p, val_rmse_p, tr_r2_p, val_r2_p = [], [], [], []
models_poly = {}

print("\n── 2.4 Polynomial Regression and Model Complexity ───────────────────────────────────────")
for deg in degrees:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=deg, include_bias=False)),
                     ('scaler', StandardScaler()),
                     ('lr', LinearRegression())])
    tr_rmse, val_rmse, tr_r2, val_r2 = fit_eval(pipe, X_tr, y_tr, X_val, y_val)
    models_poly[deg] = pipe
    tr_rmse_p.append(tr_rmse)
    val_rmse_p.append(val_rmse)
    tr_r2_p.append(tr_r2)
    val_r2_p.append(val_r2)
    n_feat = pipe.named_steps['poly'].transform(X_tr).shape[1]
    gap = val_rmse - tr_rmse
    print(f"  deg={deg}  feats={n_feat:4d}  "
          f"Train={tr_rmse:.3f}  Val={val_rmse:.3f}  "
          f"R²_val={val_r2:.3f}  gap={gap:+.3f}")

best_degree = degrees[int(np.argmin(val_rmse_p))]
print(f"\n  Best degree: {best_degree}")

plot_val_curve(list(degrees), tr_rmse_p, val_rmse_p, tr_r2_p, val_r2_p,
               'Polynomial Degree', 'Polynomial Complexity Curve',
               best_degree, 'poly_complexity_curve.png')

best_poly = models_poly[best_degree]
plot_pred_vs_true(
    [(y_tr, best_poly.predict(X_tr), 'Train'),
     (y_val, best_poly.predict(X_val), 'Validation')],
    f'Polynomial (deg={best_degree}) – Predicted vs Ground Truth',
    'poly_pred_vs_true.png')

# ── 2.5  K-NEAREST NEIGHBORS (KNN) REGRESSION ──────────────────────────────────────
# Distance: Euclidean (Minkowski p=2). Standardization is essential — without
# it, high-range features (e.g. weight ~2000–5000) dominate distance over
# low-range ones (e.g. cylinders 4–8), effectively ignoring them.
# Bias–variance with k:
#   k=1  → zero train error (each point is its own neighbour) = max variance.
#   large k → predictions approach the training mean = high bias, underfitting.
#   Optimal k identified via validation curve below.

k_values = list(range(1, 31))
tr_rmse_k, val_rmse_k, tr_r2_k, val_r2_k = [], [], [], []
models_knn = {}

print("\n── 2.5 KNN Regression ──────────────────────────────────────────────")
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, metric='minkowski', p=2)
    tr_rmse, val_rmse, tr_r2, val_r2 = fit_eval(knn, X_tr, y_tr, X_val, y_val)
    models_knn[k] = knn
    tr_rmse_k.append(tr_rmse)
    val_rmse_k.append(val_rmse)
    tr_r2_k.append(tr_r2)
    val_r2_k.append(val_r2)
    print(f"  k={k:3d}  Train={tr_rmse:.3f}  Val={val_rmse:.3f}  "
          f"R²_val={val_r2:.3f}  gap={val_rmse - tr_rmse:+.3f}")

best_k = k_values[int(np.argmin(val_rmse_k))]
print(f"\n  Best k: {best_k}")

plot_val_curve(k_values, tr_rmse_k, val_rmse_k, tr_r2_k, val_r2_k,
               'k (Number of Neighbors)', 'KNN Validation Curve',
               best_k, 'knn_error_vs_k.png')

best_knn = models_knn[best_k]
plot_pred_vs_true(
    [(y_tr, best_knn.predict(X_tr), 'Train'),
     (y_val, best_knn.predict(X_val), 'Validation')],
    f'KNN (k={best_k}) – Predicted vs Ground Truth', 'knn_pred_vs_true.png')


# ── 2.6  OPTIMIZATION BEHAVIOR ───────────────────────────────────────────────
# Three GD variants on the same linear model (w=0 start, zero bias):
#   Batch GD   (b=n)  : one update/epoch over all data → smooth, slow to escape.
#   Mini-Batch (b=32) : balances stability and update frequency.
#   SGD        (b=1)  : fastest updates, highest gradient noise.
# Learning rates chosen for visible convergence within 200 epochs:
#   Batch lr=0.05 — full-dataset gradients are low-variance → large step safe.
#   Mini-batch lr=0.03 — slightly noisier gradient → moderate step.
#   SGD lr=0.005 — per-sample gradients very noisy → small step required.

def run_gd(X, y, X_val, y_val, lr, epochs, batch_size):
    X_a, y_a = np.array(X, dtype=float), np.array(y, dtype=float)
    Xv, yv = np.array(X_val, dtype=float), np.array(y_val, dtype=float)
    n, d = X_a.shape
    w, b = np.zeros(d), 0.0
    tr_losses, val_losses = [], []
    for _ in range(epochs):
        if batch_size < n:  # only shuffle for SGD and mini-batch
            idx = np.random.permutation(n)
            Xs, ys = X_a[idx], y_a[idx]
        else:
            Xs, ys = X_a, y_a
        for s in range(0, n, batch_size):
            Xb, yb = Xs[s:s + batch_size], ys[s:s + batch_size]
            err = Xb @ w + b - yb
            w -= lr * (2 / len(yb)) * (Xb.T @ err)
            b -= lr * (2 / len(yb)) * err.sum()
        tr_losses.append(np.mean((X_a @ w + b - y_a) ** 2))
        val_losses.append(np.mean((Xv @ w + b - yv) ** 2))
    return tr_losses, val_losses


EPOCHS = 500
np.random.seed(42)
gd_configs = [
    {'label': 'Batch GD', 'lr': 0.07, 'batch_size': len(X_tr), 'color': 'steelblue'},
    {'label': 'Mini-Batch b=32', 'lr': 0.03, 'batch_size': 32, 'color': 'seagreen'},
    {'label': 'SGD', 'lr': 0.005, 'batch_size': 1, 'color': 'crimson'},
]
epochs_x = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print("\n── 2.6 Optimization ────────────────────────────────────────────────")
for ax, cfg in zip(axes, gd_configs):
    tr_loss, val_loss = run_gd(X_tr, y_tr, X_val, y_val,
                               cfg['lr'], EPOCHS, cfg['batch_size'])
    ax.plot(epochs_x, tr_loss, color=cfg['color'], linewidth=1.5, label='Train MSE')
    ax.plot(epochs_x, val_loss, color=cfg['color'], linewidth=1.5,
            linestyle='--', alpha=0.7, label='Val MSE')
    ax.set(title=f"{cfg['label']} (lr={cfg['lr']})", xlabel='Epoch', ylabel='MSE')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)
    print(f"  {cfg['label']:20s}  Final Train={tr_loss[-1]:.3f}  Val={val_loss[-1]:.3f}")

plt.suptitle('Optimization Behavior – Loss per Epoch', fontsize=13)
plt.tight_layout()
plt.savefig(fig_path('optimization_loss_curves.png'), dpi=150, bbox_inches='tight')

# ── 2.7  MODEL COMPARISON AND FINAL EVALUATION ───────────────────────────────
# Models refit on full X_train_scaled. Test set used exactly once here.

lr_final = LinearRegression()
lr_final.fit(X_train_scaled, y_train)

poly_final = Pipeline([('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
                       ('scaler', StandardScaler()),
                       ('lr', LinearRegression())])
poly_final.fit(X_train, y_train)  # raw unscaled — pipeline handles scaling

knn_final = KNeighborsRegressor(n_neighbors=best_k, metric='minkowski', p=2)
knn_final.fit(X_train_scaled, y_train)

model_entries = [
    ('Linear Regression', lr_final, X_test_scaled),
    (f'Polynomial (deg={best_degree})', poly_final, X_test),
    (f'KNN (k={best_k})', knn_final, X_test_scaled),
]

print("\n── 2.7 Model Comparison – Test Set ──────────────────────────────────")
print(f"  {'Model':<28} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("  " + "-" * 60)
comparison = []
for name, model, X_eval in model_entries:
    m = eval_metrics(y_test, model.predict(X_eval))
    comparison.append({'Model': name, **m})
    print(f"  {name:<28} {m['MSE']:>8.4f} {m['RMSE']:>8.4f} "
          f"{m['MAE']:>8.4f} {m['R2']:>8.4f}")

comparison_df = pd.DataFrame(comparison)

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
bar_colors = ['steelblue', 'seagreen', 'tomato']
for ax, metric in zip(axes, ['MSE', 'RMSE', 'MAE', 'R2']):
    bars = ax.bar(comparison_df['Model'], comparison_df[metric],
                  color=bar_colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + comparison_df[metric].max() * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set(title=metric, ylabel=metric)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)
plt.suptitle('Model Comparison – Test Set', fontsize=13)
plt.tight_layout()
plt.savefig(fig_path('model_comparison.png'), dpi=150, bbox_inches='tight')

best_name = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
best_final, X_eval_final = dict(
    (name, (model, X_ev)) for name, model, X_ev in model_entries
)[best_name]
y_pred_final = best_final.predict(X_eval_final)
m_final = eval_metrics(y_test, y_pred_final)

plot_pred_vs_true(
    [(y_test, y_pred_final, f'Test – {best_name}')],
    f'Final Model: {best_name}', 'final_model_pred_vs_true.png')

print(f"\n  Selected: {best_name}")
for k, v in m_final.items():
    print(f"  {k:<6} {v:.4f}")
