# 🚚 Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

## 📌 Project Overview

This project focuses on predicting the risk of late deliveries in global supply chain operations using Machine Learning.

Late deliveries can affect customer satisfaction, increase operational costs, and create supply chain disruptions. The objective of this project is to analyze historical logistics data and identify shipments that may have a higher risk of being delivered late.

A **Random Forest Classifier** is developed to predict late delivery risk, and an interactive **Streamlit dashboard** is designed to present delivery insights and risk categories.

## 🎯 Project Objectives

- Analyze historical logistics and shipment data.
- Perform data cleaning and exploratory data analysis (EDA).
- Apply feature engineering to create meaningful predictive features.
- Build a Machine Learning model to predict late delivery risk.
- Evaluate model performance using standard classification metrics.
- Categorize predictions into Low, Medium, and High risk.
- Develop an interactive dashboard for delivery risk analysis.

## 📂 Dataset Information

- **Dataset:** APL Logistics
- **Total Records:** 180,519
- **Target Variable:** `Late_delivery_risk`
- **Problem Type:** Binary Classification

The target variable represents:
- `0` – On-time delivery
- `1` – Late delivery

The dataset contains information related to shipping modes, customer segments, product categories, order quantities, transaction types, and geographical locations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine Learning and model evaluation |
| Random Forest | Classification algorithm |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Jupyter Notebook | Development and experimentation |
| VS Code | Development environment |

## ⚙️ Project Workflow

### 1. Data Loading and Exploration
- Loaded the APL Logistics dataset using Pandas.
- Examined dataset structure, data types, and missing values.
- Analyzed target variable distribution.
- Identified patterns and trends in logistics operations.

### 2. Data Preprocessing
- Handled missing values.
- Removed irrelevant and potentially leakage-related columns.
- Cleaned categorical values.
- Prepared numerical and categorical features for model training.

### 3. Feature Engineering

Created additional features to improve the model's ability to identify delivery risk:

- **Shipping Pressure Index:** Represents order item quantity relative to scheduled shipping duration.
- **Order Complexity:** Categorizes orders based on item quantity into Low, Medium, and High.
- **Scheduled Duration Category:** Groups scheduled shipping duration into Short, Medium, and Long.

### 4. Model Development

- Selected Random Forest Classifier.
- Split the dataset into 80% training and 20% testing data using stratified sampling.
- Applied numerical imputation and categorical encoding.
- Built a Scikit-learn preprocessing and classification pipeline.
- Trained the model using the processed dataset.

### 5. Model Evaluation

Evaluated the model using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### 6. Risk Classification

Predicted late delivery probabilities are grouped into three categories:

| Risk Category | Probability |
|---|---|
| 🟢 Low | Below 30% |
| 🟡 Medium | 30% to below 60% |
| 🔴 High | 60% and above |

These are initial operational risk bands used for the project.

### 7. Dashboard Development

Developed an interactive Streamlit dashboard to present logistics insights, analyze shipping patterns, and communicate predicted delivery risks.

## 📊 Model Performance

The Random Forest model achieved the following results on the test dataset using a classification threshold of 0.40:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 75.31% |
| Precision | 74.71% |
| Recall | 83.11% |
| F1-Score | 78.69% |
| ROC-AUC | 84.40% |

**Note:** The classification threshold was selected to balance precision and recall, with greater emphasis on identifying potentially late deliveries.

## 📁 Project Structure

```text
APL_Logistics_Late_Delivery_Prediction/
│
├── dataset/
│   ├── APL_Logistics.csv
│   ├── APL_Logistics_cleaned.csv
│   └── APL_Logistics_feature_engineered.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Development.ipynb
│
├── models/
│   └── late_delivery_model.pkl
│
├── visuals/
│   ├── feature_importance.png
│   └── feature_importance.csv
│
├── Dashboard/
│   └── APP/
│       └── app.py
│
├── reports/
│
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-link>
```

### Step 2: Navigate to the Project Folder

```bash
cd APL_Logistics_Late_Delivery_Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment (Windows)

```powershell
.venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Dashboard

```bash
python -m streamlit run Dashboard/APP/app.py
```

The dashboard will open in your default browser.

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Exploratory Data Analysis and data cleaning.
- Feature engineering and data preprocessing.
- Building classification models using Random Forest.
- Model evaluation and threshold tuning.
- Understanding feature importance.
- Developing interactive dashboards using Streamlit.
- Applying Machine Learning to a real-world logistics problem.

## ⚠️ Limitations and Future Improvements

- Validate whether all input features are available before shipment dispatch.
- Perform order-level or time-based validation when suitable identifiers or timestamps are available.
- Improve model generalization using additional logistics data.
- Explore advanced classification algorithms and hyperparameter tuning.
- Add more detailed shipment-level prediction and operational recommendation features.

The current results are preliminary because the evaluation uses a random row-level split, and repeated order records may affect the reported performance.

## 👩‍💻 Project Details

**Internship Organization:** Unified Mentor

**Project Domain:** Data Science & Machine Learning

**Project Title:** Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

**Tools:** Python, Machine Learning, Scikit-learn, Streamlit, Pandas, NumPy

---

⭐ If you find this project interesting, feel free to explore the repository!

**Developed as part of my Data Science and Machine Learning internship at Unified Mentor.**
