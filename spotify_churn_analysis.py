#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

#import cleaned dataset
df = pd.read_csv('spotify_cleaned_with_engagement.csv')

x = df.drop(['user_id', 'churned'], axis=1)
y = df["churned"]

#train tet split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, stratify=y, random_state=42
)

# 4. Oversample using SMOTE
sm = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#model training
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(x_train_scaled, y_train)
y_pred_logreg = logreg.predict(x_test_scaled)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

y_probs = rf.predict_proba(x_test)[:, 1]
threshold = 0.35
y_pred_custom = (y_probs > threshold).astype(int)

print("Custom Threshold Evaluation:")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

print("Logistic Regression:\n", classification_report(y_test, y_pred_logreg))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

#confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.savefig("Confusion Matrix - Random Forest")

#roc-auc curve
rf_probs = rf.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, rf_probs)
roc_auc = roc_auc_score(y_test, rf_probs)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1], 'k--')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
plt.savefig("ROC Curve - Random Forest")

#feature importance (Random Forest)
feat_importance = rf.feature_importances_
features = x.columns

sns.barplot(x=feat_importance, y=features)
plt.title("Feature Importance - Random Forest")
plt.xlabel("importance")
plt.ylabel("Feature")
plt.show()
plt.savefig("Feature Importance - Random Forest")
