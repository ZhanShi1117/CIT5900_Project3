import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and preprocess
df = pd.read_csv("ResearchOutputs.csv", low_memory=False)
# df = pd.read_csv("fused_outputs.csv", low_memory=False)
df.columns = [col.strip().lower() for col in df.columns]

# Filter out missing values in projectstatus
df = df[df['projectstatus'].notna()]

# Build binary target: Completed = 1, else = 0
df['label'] = (df['projectstatus'].str.lower() == 'completed').astype(int)

# Select features (same as before)
features = ['outputtype', 'projectrdc', 'projectstartyear', 'projectendyear', 'outputyear']
df = df[features + ['label']]

# Fill missing values
df['outputtype'] = df['outputtype'].fillna('Unknown')
df['projectrdc'] = df['projectrdc'].fillna('Unknown')
for col in ['projectstartyear', 'projectendyear', 'outputyear']:
    df[col] = df[col].fillna(df[col].median())

# One-hot encode categorical features
cat_features = ['outputtype', 'projectrdc']
df_encoded = pd.get_dummies(df, columns=cat_features)

# Split data
X = df_encoded.drop(columns=['label'])
y = df_encoded['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Train Random Forest
# ----------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest for ProjectStatus:\n")
print(classification_report(y_test, y_pred_rf, digits=3))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Active", "Completed"])
disp_rf.plot(cmap='Blues')
plt.title("Random Forest - ProjectStatus")
plt.show()

# Feature importance - Random Forest
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45, ha='right')
plt.title("Top 10 Feature Importances (Random Forest - ProjectStatus)")
plt.tight_layout()
plt.show()


# ----------------------------
# 3. Train Logistic Regression
# ----------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("Logistic Regression for ProjectStatus:\n")
print(classification_report(y_test, y_pred_lr, digits=3))

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["Active", "Completed"])
disp_lr.plot(cmap='Greens')
plt.title("Logistic Regression - ProjectStatus")
plt.show()



