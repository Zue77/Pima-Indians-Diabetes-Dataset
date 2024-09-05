import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (classification_report, accuracy_score, f1_score, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import skew, zscore

# Load dataset
url = 'https://raw.githubusercontent.com/Zue77/Pima-Indians-Diabetes-Dataset/main/diabetes.csv'
data = pd.read_csv(url)

# Display the total number of rows and columns
total_rows, total_columns = data.shape
print(f"Total number of rows (data points): {total_rows}")

# outcome (-1) is not a feature
print(f"Total number of columns (features): {total_columns - 1}")

# Features selection 
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age']

# Data Preprocessing
# Remove rows with zeros in critical columns
data_filtered = data[(data['Glucose'] != 0) & (data['BloodPressure'] != 0) &
                     (data['Insulin'] != 0) & (data['BMI'] != 0)]
print(f"\nNumber of entries after filtering: {data_filtered.shape[0]}")

# Print the number of people with and without diabetes
diabetes_counts = data_filtered['Outcome'].value_counts()
print(f"\nNumber of people with diabetes: {diabetes_counts.get(1, 0)}")
print(f"Number of people without diabetes: {diabetes_counts.get(0, 0)}")

# Handle Missing Values (if any)
# Check for missing values
missing_values = data_filtered.isnull().sum()
print(f"\nMissing values in each column:\n{missing_values}")

# Outlier Detection and Removal
# Z-score method for outlier detection
z_scores = np.abs(zscore(data_filtered[features]))
outliers = (z_scores > 3).any(axis=1)
data_filtered_no_outliers = data_filtered[~outliers]
print(f"\nNumber of entries after outlier removal: {data_filtered_no_outliers.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X = data_filtered_no_outliers.drop('Outcome', axis=1)
y = data_filtered_no_outliers['Outcome']
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to a DataFrame for consistency
data_scaled = pd.DataFrame(X_scaled, columns=X.columns)
data_scaled['Outcome'] = y.values

# Statistics
def print_statistics(column, data):
    avg = data[column].mean()
    min_val = data[column].min()
    max_val = data[column].max()
    print(f"\nStatistics for {column}:")
    print(f"Average {column}: {avg:.2f}")
    print(f"Min {column}: {min_val:.2f}, Max {column}: {max_val:.2f}")

for feature in features:
    print_statistics(feature, data_filtered)

# Calculate skewness for each feature
def print_skewness(column, data):
    skewness = skew(data[column])
    print(f"\nSkewness of {column}: {skewness:.2f}")

for feature in features:
    print_skewness(feature, data_filtered)

# Exploratory Data Analysis (EDA) Plots
plt.figure(figsize=(15, 10))
data_filtered.hist(bins=50, figsize=(15, 10))
plt.suptitle('Histogram of Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(15, 10))
data_filtered.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(15, 10))
plt.suptitle('Density Plots', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(15, 10))
data_filtered.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(15, 10))
plt.suptitle('Box Plots', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data_filtered.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Pair Plot
plt.figure(figsize=(12, 10))
sns.pairplot(data_filtered, hue='Outcome', markers=['o', 's'], palette='husl')
plt.suptitle('Pair Plot of Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Prepare data for modeling
X = data_filtered[features]
y = data_filtered['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with statsmodels
X_stats = sm.add_constant(X)  # Add constant for statsmodels
logit_model_stats = sm.Logit(y, X_stats)
result_stats = logit_model_stats.fit()

print("\nLogistic Regression Summary with statsmodels:")
print(result_stats.summary())

# Basic Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
y_pred_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]

# Basic K-Nearest Neighbors (with fixed k=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_pred_prob_knn = knn_model.predict_proba(X_test)[:, 1]

# Hyperparameter Tuning for Logistic Regression
logreg_tuned_params = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
logreg_tuned = GridSearchCV(LogisticRegression(max_iter=1000, solver='liblinear'), logreg_tuned_params, cv=5)
logreg_tuned.fit(X_train, y_train)

logreg_best = logreg_tuned.best_estimator_
y_pred_logreg_tuned = logreg_best.predict(X_test)
y_pred_prob_logreg_tuned = logreg_best.predict_proba(X_test)[:, 1]

# Hyperparameter Tuning for K-Nearest Neighbors
knn_tuned_params = {'n_neighbors': [3, 5, 7, 10, 15], 'weights': ['uniform', 'distance']}
knn_tuned = GridSearchCV(KNeighborsClassifier(), knn_tuned_params, cv=5)
knn_tuned.fit(X_train, y_train)

knn_best = knn_tuned.best_estimator_
y_pred_knn_tuned = knn_best.predict(X_test)
y_pred_prob_knn_tuned = knn_best.predict_proba(X_test)[:, 1]

# Plot KNN Optimal Value
k_values = knn_tuned.cv_results_['param_n_neighbors'].data
mean_test_scores = knn_tuned.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_test_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Test Score')
plt.title('K-Nearest Neighbors: Mean Test Score vs. Number of Neighbors')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Create a summary table
results = {
    'Model': ['Basic Logistic Regression', 'Basic K-Nearest Neighbors', 'Tuned Logistic Regression', 'Tuned K-Nearest Neighbors'],
    'Accuracy': [round(accuracy_score(y_test, y_pred_logreg), 4),
                 round(accuracy_score(y_test, y_pred_knn), 4),
                 round(accuracy_score(y_test, y_pred_logreg_tuned), 4),
                 round(accuracy_score(y_test, y_pred_knn_tuned), 4)],
    'F1 Score': [round(f1_score(y_test, y_pred_logreg), 4),
                 round(f1_score(y_test, y_pred_knn), 4),
                 round(f1_score(y_test, y_pred_logreg_tuned), 4),
                 round(f1_score(y_test, y_pred_knn_tuned), 4)],
    'ROC AUC': [round(roc_auc_score(y_test, y_pred_prob_logreg), 4),
                round(roc_auc_score(y_test, y_pred_prob_knn), 4),
                round(roc_auc_score(y_test, y_pred_prob_logreg_tuned), 4),
                round(roc_auc_score(y_test, y_pred_prob_knn_tuned), 4)]
}

results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)

# ROC and Precision-Recall Curves
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_prob_logreg)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_prob_knn)
fpr_logreg_tuned, tpr_logreg_tuned, _ = roc_curve(y_test, y_pred_prob_logreg_tuned)
fpr_knn_tuned, tpr_knn_tuned, _ = roc_curve(y_test, y_pred_prob_knn_tuned)

precision_logreg, recall_logreg, _ = precision_recall_curve(y_test, y_pred_prob_logreg)
precision_knn, recall_knn, _ = precision_recall_curve(y_test, y_pred_prob_knn)
precision_logreg_tuned, recall_logreg_tuned, _ = precision_recall_curve(y_test, y_pred_prob_logreg_tuned)
precision_knn_tuned, recall_knn_tuned, _ = precision_recall_curve(y_test, y_pred_prob_knn_tuned)

# ROC Curve Plot
plt.figure(figsize=(12, 8))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (Basic, AUC = {round(roc_auc_score(y_test, y_pred_prob_logreg), 4)})')
plt.plot(fpr_knn, tpr_knn, label=f'K-Nearest Neighbors (Basic, AUC = {round(roc_auc_score(y_test, y_pred_prob_knn), 4)})')
plt.plot(fpr_logreg_tuned, tpr_logreg_tuned, label=f'Tuned Logistic Regression (AUC = {round(roc_auc_score(y_test, y_pred_prob_logreg_tuned), 4)})')
plt.plot(fpr_knn_tuned, tpr_knn_tuned, label=f'Tuned K-Nearest Neighbors (AUC = {round(roc_auc_score(y_test, y_pred_prob_knn_tuned), 4)})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Precision-Recall Curve Plot
plt.figure(figsize=(12, 8))
plt.plot(recall_logreg, precision_logreg, label=f'Logistic Regression (Basic, AP = {round(average_precision_score(y_test, y_pred_prob_logreg), 4)})')
plt.plot(recall_knn, precision_knn, label=f'K-Nearest Neighbors (Basic, AP = {round(average_precision_score(y_test, y_pred_prob_knn), 4)})')
plt.plot(recall_logreg_tuned, precision_logreg_tuned, label=f'Tuned Logistic Regression (AP = {round(average_precision_score(y_test, y_pred_prob_logreg_tuned), 4)})')
plt.plot(recall_knn_tuned, precision_knn_tuned, label=f'Tuned K-Nearest Neighbors (AP = {round(average_precision_score(y_test, y_pred_prob_knn_tuned), 4)})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.show()

# Cross-Validation for Tuned Models
cv_scores_logreg_tuned = cross_val_score(logreg_best, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores for Tuned Logistic Regression: {cv_scores_logreg_tuned}")
print(f"Mean Cross-Validation Accuracy for Tuned Logistic Regression: {cv_scores_logreg_tuned.mean():.4f}")

cv_scores_knn_tuned = cross_val_score(knn_best, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores for Tuned K-Nearest Neighbors: {cv_scores_knn_tuned}")
print(f"Mean Cross-Validation Accuracy for Tuned K-Nearest Neighbors: {cv_scores_knn_tuned.mean():.4f}")

# Confusion Matrices
# Basic Logistic Regression
def plot_confusion_matrices(y_true, y_preds, labels):
    plt.figure(figsize=(15, 10))
    for i, (y_pred, label) in enumerate(zip(y_preds, labels)):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
        plt.subplot(2, 2, i+1)
        disp.plot(ax=plt.gca(), colorbar=False)
        plt.title(label)
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(y_test, [y_pred_logreg, y_pred_knn, y_pred_logreg_tuned, y_pred_knn_tuned],
                        ['Logistic Regression', 'KNN', 'Tuned Logistic Regression', 'Tuned KNN'])
