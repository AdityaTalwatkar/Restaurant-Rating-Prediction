# 🍽️ Restaurant Rating Prediction

This project focuses on predicting restaurant ratings based on various features such as cuisine type, location, price range, and customer engagement (votes). The goal is to help restaurant businesses and food delivery platforms understand key factors that influence customer ratings.

---

## 📌 Project Overview

- 📊 **Model Used**: Random Forest Regressor  
- 🧪 **Data**: Synthetic dataset with 1,000 samples  
- 🛠️ **Tech Stack**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- 🎯 **Objective**: Predict `aggregate_rating` using features like votes, price range, cuisine, and location.

---

## 🧠 Key Features

- Generates a synthetic dataset of Indian restaurants with:
  - 🎫 `votes`
  - 💰 `price_range` (1: Low, 2: Medium, 3: High)
  - 🍱 `cuisine` (e.g., North Indian, South Indian, Mughlai, etc.)
  - 🏙️ `location` (major cities in India)
  - 🌟 `aggregate_rating` (target variable)

- Preprocessing includes:
  - One-hot encoding of categorical variables
  - Train-test split for model evaluation

- Model training and evaluation:
  - Random Forest Regressor
  - Performance metrics: MSE, RMSE, R² Score

- Feature importance visualization:
  - Bar chart showing top 10 features impacting ratings

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/restaurant-rating-prediction.git
cd restaurant-rating-prediction
```

### 2. Install Dependencies
Make sure you have Python 3.x installed.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Project
```bash
python restaurant_rating_prediction.py
```

---

## 📈 Sample Output

```
Model Performance:
MSE: 0.0963
RMSE: 0.3103
R-squared: 0.7836

Top 10 Most Important Features:
      Feature         Importance
0      votes           0.1723
1  price_range         0.0857
...
```

> 💡 **Insight**: Votes and price range are the most influential features in determining restaurant ratings. Restaurant owners should focus on improving service quality and affordability to attract more customers and higher ratings.

---


## 🤖 Future Improvements

- Integrate real-world restaurant data (e.g., Zomato/Yelp/Kaggle datasets)
- Hyperparameter tuning and model optimization
- Try other models like XGBoost, Gradient Boosting, or Neural Networks
- Build a web dashboard using Flask or Streamlit

---

## 👨‍💻 Author

**Aditya Rajendra Talwatkar**  


## 📜 License

This project is for educational and research purposes only. Feel free to fork and modify!

