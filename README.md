# 🎵 Spotify Churn Prediction Project

## 📌 Overview
This project aims to predict user churn on Spotify using behavioral and engagement data. By identifying which users are likely to leave the platform, Spotify can proactively take steps to retain them.

---

## 📊 Project Highlights

- 📁 Cleaned and engineered features from Spotify user data
- 📈 Exploratory Data Analysis (EDA) to uncover churn drivers
- 🤖 Applied classification models: Logistic Regression & Random Forest
- 🎯 Tuned models for **higher recall**, especially on churned users
- 📉 Evaluated with ROC AUC, confusion matrices, and feature importance
- 🧠 Developed a composite **Engagement Score** for early churn detection

---

## 📂 Repository Structure

📦 spotify-churn-prediction/

├── spotify_churn_analysis.py # Final model training and evaluation

├── spotify_churn_analysis_cleaning.ipynb # Data cleaning and feature engineering

├── spotify_EDA.ipynb # EDA and visualizations

├── spotify_cleaned_with_engagement.csv # Final cleaned dataset

├── assets/ # Contains saved plots (ROC, feature importance, etc.)

├── Report.pdf # Final report (optional)

├── Slides.pptx # Presentation slide deck (optional)

└── README.md # This file


---

## 🧪 Models Used

| Model               | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) |
|---------------------|----------|----------------|--------------------|------------|
| Logistic Regression | 0.75     | **0.73**        | 0.41               | **0.52**   |
| Random Forest (Tuned)| 0.78     | 0.38            | 0.41               | 0.39       |

*Class balancing and threshold tuning were applied to improve churn detection.*

---

## 📈 Key Insights

- High churn correlation with:
  - `days_since_last_login` ↑
  - `avg_daily_minutes` ↓
  - `num_playlists` ↓
- Custom **Engagement Score** effectively predicts churn probability
- Logistic Regression provided the best **recall**, essential for retention strategies

---

## 🛠 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Jupyter Notebooks
- imbalanced-learn (SMOTE)
- VS Code & Git

---

## ✅ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/spotify-churn-prediction.git
   cd spotify-churn-prediction
2. Install dependencies
   pandas==1.5.3
numpy==1.24.0
scikit-learn==1.1.3
matplotlib==3.6.3
seaborn==0.11.2
imbalanced-learn>=0.10.0
jupyter
3. Run the model
   python spotify_churn_analysis.py

📬 Suggestions for Retention
Send re-engagement prompts to users with low engagement scores

- Offer playlist suggestions and session-based rewards

- Monitor skip patterns and support ticket activity

- Prioritize premium users showing inactivity trends

📌 Author
Tanisha Mahajan
---
