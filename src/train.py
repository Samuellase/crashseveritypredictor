import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import warnings
import time
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def load_data(path: str) -> tuple:
    print("\n[1] Loading data...")
    df = pd.read_csv(path)
    print(f"  Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Target balance: {df['Severity_Binary'].value_counts(normalize=True).to_dict()}")

    X = df.drop('Severity_Binary', axis=1)
    y = df['Severity_Binary']
    return X, y, df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Creating enhanced features...")

    # Weather composite features
    weather_cols = ['Weather_Rain', 'Weather_Light Rain', 'Weather_Light Snow']
    existing_weather = [c for c in weather_cols if c in df.columns]
    df['Weather_Severity'] = (
        (df['Precipitation(in)'] > 0).astype(int) * 2 +
        (df['Visibility(mi)'] < 2).astype(int) * 2 +
        (df['Wind_Speed(mph)'] > 20).astype(int) +
        (df[existing_weather].sum(axis=1).clip(0, 1) if existing_weather else 0)
    )

    # Temperature features
    df['Temp_Extreme'] = ((df['Temperature(F)'] < 32) | (df['Temperature(F)'] > 95)).astype(int)
    df['Temp_Humidity_Index'] = df['Temperature(F)'] * df['Humidity(%)'] / 100

    # Time-based features
    df['Rush_Hour'] = (
        ((df['Hour'] >= 7) & (df['Hour'] <= 9)) |
        ((df['Hour'] >= 16) & (df['Hour'] <= 18))
    ).astype(int)
    df['Dangerous_Hours'] = ((df['Hour'] >= 22) | (df['Hour'] <= 4)).astype(int)
    df['Weekend_Night'] = (
        (df['IsWeekend'] == 1) & ((df['Hour'] >= 20) | (df['Hour'] <= 4))
    ).astype(int)

    # Infrastructure risk score
    df['Infrastructure_Risk'] = (
        df['Junction'] + df['Traffic_Signal'] + df['Crossing'] + df['Stop']
    ).clip(0, 4)

    # Visibility features
    df['Poor_Visibility'] = (
        (df['Visibility(mi)'] < 5) | (df['Civil_Twilight'] == 1)
    ).astype(int)

    # Interaction features
    df['Weather_Night'] = df['Weather_Severity'] * df['Dangerous_Hours']
    df['Pressure_Anomaly'] = abs(df['Pressure(in)'] - df['Pressure(in)'].mean())

    new_cols = ['Weather_Severity', 'Temp_Extreme', 'Rush_Hour',
                'Dangerous_Hours', 'Infrastructure_Risk', 'Poor_Visibility',
                'Weekend_Night', 'Weather_Night']
    print(f"  Created {len(new_cols)} new features")
    print("\n  Top new feature correlations with target:")
    for col in new_cols[:5]:
        if col in df.columns:
            corr = df[col].corr(df['Severity_Binary'])
            print(f"    {col:25s}: {corr:+.4f}")

    return df


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    print("\n[3] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def train_baseline_rf(X_train, y_train, X_test, y_test) -> RandomForestClassifier:
    print("\n[4] Training baseline Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    print(f"  Baseline accuracy: {acc:.4f}")
    return rf


def tune_random_forest(X_train, y_train, X_test, y_test) -> RandomForestClassifier:
    print("\n[5] Hyperparameter tuning Random Forest...")
    param_dist = {
        'n_estimators': [300, 400, 500],
        'max_depth': [25, 30, 35],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_dist, n_iter=12, cv=3, random_state=42, n_jobs=-1, verbose=0
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"  Completed in {(time.time() - start) / 60:.1f} minutes")
    rf_tuned = search.best_estimator_
    print(f"  Tuned RF accuracy: {rf_tuned.score(X_test, y_test):.4f}")
    return rf_tuned


def train_xgboost(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    print("\n[6] Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    model.fit(X_train, y_train, verbose=False)
    print(f"  XGBoost accuracy: {model.score(X_test, y_test):.4f}")
    return model


def train_stacking_ensemble(X_train, y_train, X_test, y_test) -> StackingClassifier:
    print("\n[7] Building stacking ensemble...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                   reg_alpha=0.1, random_state=42, n_jobs=-1)),
        ('tree', DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42))
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    print(f"  Stacking accuracy: {stacking.score(X_test, y_test):.4f}")
    return stacking


def evaluate_model(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'name': name,
        'model': model,
        'train_acc': model.score(X_train, y_train),
        'test_acc': model.score(X_test, y_test),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def get_feature_importances(result: dict, feature_names) -> pd.DataFrame:
    model = result['model']
    name = result['name']
    if 'Stacking' in name:
        importances = model.estimators_[0].feature_importances_
    else:
        importances = model.feature_importances_
    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)


def plot_results(results: list, best: dict, X, y_test, output_path: str) -> None:
    print("\n[11] Creating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    names = [r['name'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]

    # Plot 1: Model comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(names))
    width = 0.35
    ax1.bar(x_pos - width / 2, train_accs, width, label='Train', alpha=0.8, color='#3498db')
    ax1.bar(x_pos + width / 2, test_accs, width, label='Test', alpha=0.8, color='#e74c3c')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Accuracy progression
    ax2 = axes[0, 1]
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12']
    bars = ax2.barh(range(len(names)), test_accs, color=colors[:len(names)], alpha=0.8)
    ax2.set_xlabel('Test Accuracy')
    ax2.set_title('Accuracy Progression', fontweight='bold')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    for bar, val in zip(bars, test_accs):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9)
    ax2.set_xlim([0.65, 0.82])
    ax2.grid(axis='x', alpha=0.3)

    # Plot 3: ROC Curve
    ax3 = axes[0, 2]
    fpr, tpr, _ = roc_curve(y_test, best['y_proba'])
    ax3.plot(fpr, tpr, linewidth=3,
             label=f"{best['name']} (AUC={best['auc']:.3f})", color='#2ecc71')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title(f"ROC Curve - {best['name']}", fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Confusion Matrix
    ax4 = axes[1, 0]
    cm = confusion_matrix(y_test, best['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    ax4.set_xlabel('Predicted Severity')
    ax4.set_ylabel('True Severity')
    ax4.set_title(f"Confusion Matrix - {best['name']}", fontweight='bold')

    # Plot 5: Feature Importance
    ax5 = axes[1, 1]
    feat_imp = get_feature_importances(best, X.columns)
    top_features = feat_imp.head(12)
    ax5.barh(range(len(top_features)), top_features['Importance'], color='#9b59b6', alpha=0.8)
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['Feature'], fontsize=9)
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 12 Feature Importances', fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    # Plot 6: Metrics Summary
    ax6 = axes[1, 2]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values = [best['test_acc'], best['precision'], best['recall'], best['f1'], best['auc']]
    colors_m = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax6.bar(metrics, values, color=colors_m, alpha=0.8)
    ax6.set_ylabel('Score')
    ax6.set_title(f"Performance Metrics - {best['name']}", fontweight='bold')
    ax6.set_ylim([0.6, 0.85])
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    plt.suptitle('Vehicle Crash Severity Prediction - Complete Results',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    processed_path = os.path.join('data', 'processed', 'clean_crash_data.csv')
    results_output = os.path.join('results', 'final_results.png')

    if not os.path.exists(processed_path):
        print("ERROR: Processed data not found. Run src/preprocess.py first.")
        raise FileNotFoundError(processed_path)

    X, y, df = load_data(processed_path)
    df = engineer_features(df)
    X = df.drop('Severity_Binary', axis=1)
    y = df['Severity_Binary']

    X_train, X_test, y_train, y_test = split_data(X, y)

    rf_base = train_baseline_rf(X_train, y_train, X_test, y_test)
    rf_tuned = tune_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    stacking = train_stacking_ensemble(X_train, y_train, X_test, y_test)

    results = [
        evaluate_model(rf_base, X_train, X_test, y_train, y_test, 'Baseline RF'),
        evaluate_model(rf_tuned, X_train, X_test, y_train, y_test, 'Tuned RF'),
        evaluate_model(xgb_model, X_train, X_test, y_train, y_test, 'XGBoost'),
        evaluate_model(stacking, X_train, X_test, y_train, y_test, 'Stacking'),
    ]

    print("\n[8] Results comparison:")
    comparison = pd.DataFrame([{
        'Model': r['name'],
        'Train Acc': round(r['train_acc'], 4),
        'Test Acc': round(r['test_acc'], 4),
        'Overfitting': round(r['train_acc'] - r['test_acc'], 4)
    } for r in results])
    print(comparison.to_string(index=False))

    best = max(results, key=lambda r: r['test_acc'])
    baseline_acc = results[0]['test_acc']
    improvement = best['test_acc'] - baseline_acc

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL:  {best['name']}")
    print(f"ACCURACY:    {best['test_acc']:.2%}")
    print(f"IMPROVEMENT: +{improvement:.2%} ({improvement / baseline_acc * 100:.1f}% relative gain)")
    print(f"{'=' * 60}")

    print(f"\n[9] Detailed evaluation - {best['name']}...")
    print(f"  Accuracy:  {best['test_acc']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1 Score:  {best['f1']:.4f}")
    print(f"  ROC-AUC:   {best['auc']:.4f}")

    feat_imp = get_feature_importances(best, X.columns)
    print(f"\n[10] Top 15 Most Important Features:")
    for _, row in feat_imp.head(15).iterrows():
        print(f"  {row['Feature']:30s} {row['Importance']:.4f}")

    os.makedirs('results', exist_ok=True)
    plot_results(results, best, X, y_test, results_output)

    print(f"""
FINAL SUMMARY
  Dataset:    {X.shape[0]:,} crash records, {X.shape[1]} features
  Best Model: {best['name']} at {best['test_acc']:.2%} accuracy
  Methods:    Feature engineering, hyperparameter tuning, ensemble stacking
    """)


if __name__ == '__main__':
    main()
