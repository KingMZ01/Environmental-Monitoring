<div align="center"> <img src="https://img.icons8.com/fluency/96/air-quality.png" width="90"/> <h1 style="font-size:2.5rem; margin-bottom:0;">Air Quality Prediction Models</h1> <p style="font-size:1.2rem; margin-top:0;"> <b>🌫️ Machine Learning-Based Air Quality Monitoring and Prediction</b> </p> <p> <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python"/></a> <a href="https://scikit-learn.org/" target="_blank"><img src="https://img.shields.io/badge/scikit--learn-0.24-orange?logo=scikit-learn&logoColor=white"/></a> <a href="https://xgboost.readthedocs.io/en/stable/" target="_blank"><img src="https://img.shields.io/badge/XGBoost-1.7-red?logo=xgboost&logoColor=white"/></a> <img src="https://img.shields.io/badge/License-MIT-green"/> </p> </div> <p align="center" style="font-size:1.1rem;"> <b>✨ Explore, model, and predict air quality using advanced machine learning techniques. Compare models like Random Forest and XGBoost to identify the best approach. ✨</b> </p>

------------------------

# Environmental-Monitoring
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

<p align="center"> <img width="1080" alt="Notebook-preview" src="https://github.com/user-attachments/assets/40c4a9e6-f68a-4b7f-8abc-07dcfb3d494e" /> <br> <i>Table of contents.</i> </p>

---

<p align="center"> <img width="1080" alt="Predict next-hour PM2.5" src="https://github.com/user-attachments/assets/dd5e5731-067a-43f1-a7df-5967f690b346" /> <br> <i>Quick Inference (Next-Hour Forecast).</i> </p>

----
## 🔎 Outputs

<p align="center"><img width="860" height="749" alt="AQI Category Confusion Matrix" src="https://github.com/user-attachments/assets/41be046a-5964-4980-80c6-c2fb0b861037" />

---

<p align="center"><img width="540" height="547" alt="Scatter Plot (True vs Predicted)" src="https://github.com/user-attachments/assets/a7c1bae1-709c-4b8d-b46d-c5ecdc1258c5" />

---

<p align="center"><img width="1160" height="523" alt="Time Series Plot (True vs Predicted)" src="https://github.com/user-attachments/assets/a1ef5080-fb8a-484f-b1e7-451b94ddd2a0" />

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

