# LIBRARIES

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import make_scorer
from sklearn.utils.class_weight import compute_sample_weight

# LOAD DATA
# Treat common textual NA markers as actual NaN
df = pd.read_csv("recipe_site_traffic_2212.csv", na_values=["NA", "N/A", "na", "Na", "n/a"])

print("First rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values (initial):")
print(df.isnull().sum())

# DATA VALIDATION

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Check numeric ranges
print("\nSummary statistics:")
print(df.describe(include="all"))


def parse_servings(val):
    """
    Extract first integer found in the string and return as float.
    If no digits found or value is NaN, return np.nan.
    Example handled: "6 as a snack,4 as a snack" -> 6.0
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    try:
        return float(nums[0])
    except Exception:
        return np.nan

df["servings"] = df["servings"].apply(parse_servings)


# Convert numeric columns that might be read as object because of textual "NA"
numeric_cols = ["calories", "carbohydrate", "sugar", "protein"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop recipes where all numeric cols AND category are missing

before_drop = len(df)
df = df.dropna(subset=numeric_cols + ["category"], how="all")
dropped = before_drop - len(df)
print(f"\nDropped {dropped} rows where all {numeric_cols} and category were missing.")


# Convert target variable 'high_traffic'
# Map "High" -> 1, else -> 0

df["high_traffic"] = df["high_traffic"].astype(str).str.strip().map({"High": 1})
df["high_traffic"] = df["high_traffic"].fillna(0).astype(int)

print("\nTarget distribution after mapping:")
print(df["high_traffic"].value_counts())

# DATA CLEANING

# Remove duplicates
df = df.drop_duplicates()


# For remaining missing numeric values, fill with median
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Fill missing category with Unknown
df["category"] = df["category"].fillna("Unknown")

# EXPLORATORY DATA ANALYSIS

sns.set_style("whitegrid")

columns = ["calories", "carbohydrate", "sugar", "protein"]

# FILTER OUT HIGH OUTLIERS (above 3 standard deviations)

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    upper_bound = mean + 3 * std
    before_filter = len(df)
    df = df[df[col] <= upper_bound]  # keep only rows <= 3 std above mean
    filtered_count = before_filter - len(df)
    print(f"Filtered {filtered_count} rows from '{col}' above 3 std dev ({upper_bound:.2f})")

# Scatter plot for each column
for col in columns:
    plt.figure()
    plt.scatter(df.index, df[col])
    plt.xlabel("Recipe Index")
    plt.ylabel(col)
    plt.title(f"Scatter Plot of {col}")
    plt.show()
    
# Category counts

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="category")
plt.xticks(rotation=45)
plt.title("Recipe Categories")
plt.tight_layout()
plt.show()



# Calories vs Protein

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="calories",
    y="protein",
    hue="high_traffic"
)
plt.title("Calories vs Protein by Traffic")
plt.tight_layout()
plt.show()

# FEATURE PREPARATION


# Drop recipe identifier (model doesn't need it)
X = df.drop(["high_traffic", "recipe"], axis=1)
y = df["high_traffic"]

categorical_features = ["category"]
numeric_features = ["calories", "carbohydrate", "sugar", "protein", "servings"]


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Train/test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Compute balanced sample weights for training to counter class imbalance
sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)


# MODEL 1: Logistic Regression (with grid search optimizing recall)

log_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=2000, solver="saga"))
])

# search hyperparams for logistic regression
log_param_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l2"],        # l2 for stability
    "model__class_weight": ["balanced"]  # ensure LR uses balanced class weights
}

# scoring focuses on recall for positive class (high_traffic = 1)
recall_pos_scorer = make_scorer(recall_score, pos_label=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_search = GridSearchCV(
    estimator=log_pipeline,
    param_grid=log_param_grid,
    scoring=recall_pos_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# Fit using sample weights (passed to final estimator inside pipeline as 'model__sample_weight')
log_search.fit(X_train, y_train, **{"model__sample_weight": sample_weight_train})

print("\nBest Logistic Regression params:", log_search.best_params_)
best_log = log_search.best_estimator_

# Predict on test set (probabilities too)
y_pred_log = best_log.predict(X_test)
y_proba_log = best_log.predict_proba(X_test)[:, 1]

print("\nLogistic Regression (default 0.5) Results")
print(classification_report(y_test, y_pred_log))
acc_log = accuracy_score(y_test, y_pred_log)
print("Accuracy:", acc_log)

# MODEL 2: Gradient Boosting with hyperparam tuning

gb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.1, 0.05],
    "model__min_samples_leaf": [1, 3]
}

gb_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_param_grid,
    scoring=recall_pos_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# For GradientBoostingClassifier we pass sample weight similarly via fit param
gb_search.fit(X_train, y_train, **{"model__sample_weight": sample_weight_train})

print("\nBest Gradient Boosting params:", gb_search.best_params_)
best_gb = gb_search.best_estimator_

y_pred_gb = best_gb.predict(X_test)
y_proba_gb = best_gb.predict_proba(X_test)[:, 1]

print("\nGradient Boosting (default 0.5) Results")
print(classification_report(y_test, y_pred_gb))
acc_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy:", acc_gb)

# FUNCTION: compute specificity and search thresholds to reach recall >= 0.8

def specificity_from_confusion(cm):
    # cm format from sklearn: [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def find_threshold_for_target_recall(y_true, y_proba, target_recall=0.80):
    """
    Scan thresholds from 0.01 to 0.99 and return the threshold that
    achieves recall >= target_recall and maximizes specificity among those thresholds.
    Returns (best_threshold, metrics_dict) or (None, best_metrics) if none reach target.
    """
    best = None
    best_spec = -1.0
    best_metrics = None
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred_t, pos_label=1)
        cm = confusion_matrix(y_true, y_pred_t)
        spec = specificity_from_confusion(cm)
        prec = precision_score(y_true, y_pred_t, zero_division=0)
        acc = accuracy_score(y_true, y_pred_t)
        if rec >= target_recall:
            # prefer threshold with highest specificity
            if spec > best_spec:
                best_spec = spec
                best = t
                best_metrics = {
                    "threshold": t,
                    "recall": rec,
                    "specificity": spec,
                    "precision": prec,
                    "accuracy": acc,
                    "confusion_matrix": cm
                }
    # if none reached target recall, return threshold that gives highest recall
    if best is None:
        best_rec = -1.0
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            rec = recall_score(y_true, y_pred_t, pos_label=1)
            cm = confusion_matrix(y_true, y_pred_t)
            spec = specificity_from_confusion(cm)
            prec = precision_score(y_true, y_pred_t, zero_division=0)
            acc = accuracy_score(y_true, y_pred_t)
            if rec > best_rec or (rec == best_rec and spec > best_spec):
                best_rec = rec
                best_spec = spec
                best = t
                best_metrics = {
                    "threshold": t,
                    "recall": rec,
                    "specificity": spec,
                    "precision": prec,
                    "accuracy": acc,
                    "confusion_matrix": cm
                }
    return best, best_metrics

# Find thresholds for both models
target_recall = 0.80

th_log, metrics_log = find_threshold_for_target_recall(y_test, y_proba_log, target_recall=target_recall)
th_gb, metrics_gb = find_threshold_for_target_recall(y_test, y_proba_gb, target_recall=target_recall)

print("\n--- Threshold tuning results (target recall = {:.0%}) ---".format(target_recall))
if metrics_log:
    print("\nLogistic Regression best threshold:", th_log)
    print("Logistic metrics at that threshold:")
    print("Recall (sensitivity):", metrics_log["recall"])
    print("Specificity (true negative rate):", metrics_log["specificity"])
    print("Precision:", metrics_log["precision"])
    print("Accuracy:", metrics_log["accuracy"])
    print("Confusion matrix:\n", metrics_log["confusion_matrix"])
else:
    print("No logistic threshold found.")

if metrics_gb:
    print("\nGradient Boosting best threshold:", th_gb)
    print("Gradient Boosting metrics at that threshold:")
    print("Recall (sensitivity):", metrics_gb["recall"])
    print("Specificity (true negative rate):", metrics_gb["specificity"])
    print("Precision:", metrics_gb["precision"])
    print("Accuracy:", metrics_gb["accuracy"])
    print("Confusion matrix:\n", metrics_gb["confusion_matrix"])
else:
    print("No gradient boosting threshold found.")

#  use the thresholded predictions for the "final" predictions 
if metrics_gb:
    chosen_t = metrics_gb["threshold"]
    y_pred_gb_thresh = (y_proba_gb >= chosen_t).astype(int)
else:
    chosen_t = 0.5
    y_pred_gb_thresh = y_pred_gb


# CONFUSION MATRIX & PLOT (for chosen GB threshold)

import matplotlib.pyplot as plt
import seaborn as sns

cm_final = confusion_matrix(y_test, y_pred_gb_thresh)
plt.figure(figsize=(6,5))
sns.heatmap(cm_final, annot=True, fmt="d", cmap="coolwarm")
plt.title(f"Confusion Matrix - Gradient Boosting (threshold={chosen_t:.2f})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE (Gradient Boosting)

# Extract feature names after preprocessing
feature_names = (
    numeric_features +
    list(
        best_gb.named_steps["prep"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
)

gb_estimator = best_gb.named_steps["model"]
importances = gb_estimator.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top Features Predicting Recipe Popularity (Gradient Boosting)")
plt.tight_layout()
plt.show()