<div align="center"> <img width="120" alt="Air quality icon" src="https://github.com/user-attachments/assets/4b35fc97-3fb2-44b3-a04a-0918dd3ee4a7" /> <h1 style="font-size:2.5rem; margin-bottom:0;">Air Quality Prediction Models</h1> <p style="font-size:1.2rem; margin-top:0;"> <b>🌫️ Machine Learning-Based Air Quality Monitoring and Prediction</b> </p> <p> <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python"/></a> <a href="https://scikit-learn.org/" target="_blank"><img src="https://img.shields.io/badge/scikit--learn-0.24-orange?logo=scikit-learn&logoColor=white"/></a> <a href="https://xgboost.readthedocs.io/en/stable/" target="_blank"><img src="https://img.shields.io/badge/XGBoost-1.7-red?logo=xgboost&logoColor=white"/></a> <img src="https://img.shields.io/badge/License-MIT-green"/> </p> </div> <p align="center" style="font-size:1.1rem;"> <b>✨ Explore, model, and predict air quality using advanced machine learning techniques. Compare models like Random Forest and XGBoost to identify the best approach. ✨</b> </p>

------------------------

# An Environmental Monitoring Project
ML models for predicting air quality and analyzing environmental pollution data.

---
## 🚀 Features
- 📊 **Data Exploration:** Analyze and visualize air quality datasets, feature distributions, and correlations.
- 🧹 **Preprocessing:** Handle missing values, normalize data, and prepare features for modeling.
- 🤖 **Modeling:** Train Random Forest, XGBoost, and other baseline models.
- 🏆 **Evaluation:** Compare model performance using metrics like RMSE, MAE, and feature importance.
- 🔮 **Prediction:** Predict Air Quality Index (AQI) for new inputs.
- 💾 **Model Saving/Loading:** Persist trained models for reuse.
- 🧪 **Testing:** Ensure data integrity and model correctness with test cases.

----
## 🌈 Notebook Preview

<p align="center"> <img width="1080" alt="Notebook-preview" src="https://github.com/user-attachments/assets/dcd1216c-a7c5-4fa5-beda-bc0399b662f9" /> <br> <i>Table of contents.</i> </p>

---

<p align="center"> <img width="1080" alt="Predict next-hour PM2.5" src="https://github.com/user-attachments/assets/691018c3-1468-47f0-a939-f4de3d2abcca" /> <br> <i>Quick Inference (Next-Hour Forecast).</i> </p>

----

## 🔎 Outputs

<p align="center"><img width="860" height="749" alt="AQI Category Confusion Matrix" src="https://github.com/user-attachments/assets/6a79cfba-32c5-41f2-9ad2-4a2cf9deec2e" />

---

<p align="center"><img width="540" height="547" alt="Scatter Plot (True vs Predicted)" src="https://github.com/user-attachments/assets/612d75d4-17d7-41da-88f0-6214e483dd37" />

---

<p align="center"><img width="1160" height="523" alt="Time Series Plot (True vs Predicted)" src="https://github.com/user-attachments/assets/c5aabce6-5a4b-457f-a0d7-16fb1f9be5c4" />
  
---

## ⚡ Quick Start

```
# 1. Clone the repository
git clone <your-repo-url>
cd Air_Quality_Prediction_Models

# 2. (Optional) Create a virtual environment
python -m venv .venv
# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook Air_Quality_Prediction_Models.ipynb
```

---

## 🧑‍💻 Usage

- Explore raw air quality datasets in the notebook.
- Perform data cleaning, feature engineering, and preprocessing.
- Train Random Forest and XGBoost models.
- Evaluate models using metrics and visualization plots.
- Make predictions for new environmental data points.
---
## 🗂 Project Structure
```
├── Air_Quality_Prediction_Models.ipynb    # Main notebook with EDA, modeling, and evaluation
├── Data/                                  # Raw and processed datasets
├── Models/                                # Saved trained models
├── Plots/                                 # Visualization outputs
├── Documents/                             # Reports, PDFs, and presentations
├── Requirements.txt                       # Python dependencies
└── README.md                              # Project documentation

```
---

## 🧪 Testing

Validate data preprocessing and model outputs with included test scripts.<br>Ensure input datasets are consistent and free of missing values.
```
pytest tests/
```
---

## 💾 Model Saving/Loading

```
import joblib

# Save model
joblib.dump(rf_model, 'models/random_forest_model.pkl')

# Load model
model = joblib.load('models/random_forest_model.pkl')
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!<br>Feel free to fork the repo, open an issue, or submit a pull request.

**How to contribute:**

- ⭐ Star this repo
- Fork the repository
- Create a new branch (git checkout -b feature/YourFeature)
- Commit your changes (git commit -m 'Add some feature')
- Push the branch (git push origin feature/YourFeature)
- Open a Pull Request

---

## 🙏 Credits


- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
- Project by [Mohammad Zakariya](https://github.com/KingMZ01)


---

## 👤 Author

Mohammad Zakariya

<a href="https://github.com/KingMZ01" target="_blank">GitHub: @Mohammad Zakariya</a>

Email: mzakariya239@gmail.com

---

<div align="center"> <img src="https://img.icons8.com/fluency/48/party-baloons.png" width="40"/> <h3>Made with ❤️ by Mohammad Zakariya</h3> <p>2025 &copy; All rights reserved.</p> </div>

