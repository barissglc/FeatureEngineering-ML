```markdown
# MTA Artificial Intelligence Modeling

This project was developed under the **Milli Teknoloji Akademisi Artificial Intelligence Expertise Program**. The study aims to analyze time series data, perform feature engineering, and achieve the best results using various machine learning models.

## 🚀 Project Content
This project consists of the following components:
- **Data Processing**: Filling in missing data, scaling, and transformations.
- **Feature Engineering**: Extracting time series features using tsfresh.
- **Modeling**: Utilizing XGBoost, CatBoost, RandomForest, and StackingClassifier.
- **Hyperparameter Optimization**: Finding the best parameters with GridSearchCV.
- **Model Performance Analysis**: Evaluating with metrics such as Accuracy, F1 Score, and ROC-AUC.

---

## 📌 Libraries Used
The following libraries were utilized in this project:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import yfinance as yf
import tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import pandas_ta as ta
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
```

---

## 📊 Methods Applied
Below are the details of the methods applied in this project:

### **1. Data Preprocessing**
- Filling in missing values and cleaning unnecessary columns.
- Applying Label Encoding and One-Hot Encoding.
- Scaling features using StandardScaler.

### **2. Feature Engineering**
- Extracting time series features with the **tsfresh** library.
- Reducing dimensionality using **PCA (Principal Component Analysis)**.
- Selecting the most important features using the **SelectKBest** method.

### **3. Modeling**
- **RandomForestClassifier**: Used especially to handle imbalanced data.
- **XGBoost & CatBoost**: Two of the best gradient boosting models were employed.
- **StackingClassifier**: Combined different models to achieve more robust predictions.

### **4. Model Optimization**
- Determining the best hyperparameters using **GridSearchCV**.
- Balancing the data with **ADASYN (Adaptive Synthetic Sampling)**.

### **5. Model Performance and Evaluation**
- Analyzing model predictions with a **Confusion Matrix**.
- Comparing model performance using metrics such as **ROC-AUC, Accuracy, and F1 Score**.

---

## 📈 Results and Conclusions
- **StackingClassifier** achieved higher accuracy and F1 scores compared to individual models.
- The use of **PCA** reduced model performance by causing the loss of important features.
- Features extracted with **tsfresh** significantly enhanced the model's predictive power.
- The **ADASYN method** facilitated more successful predictions on imbalanced datasets.

---

## 🛠 Installation and Usage
To run this project on your computer, follow the steps below:

### 1. Install Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn requests yfinance tsfresh xgboost catboost scikit-learn imbalanced-learn pandas_ta
```

### 2. Run the Notebook
Open and run this file using Jupyter Notebook or Colab.
```bash
jupyter notebook YZUP_VYU_Baris_Gulec.ipynb
```

### 3. Train and Test the Model
Execute the code blocks provided in the notebook to train and test the model.

---

## 📚 Resources
- **scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **tsfresh**: [https://tsfresh.readthedocs.io/](https://tsfresh.readthedocs.io/)
- **XGBoost**: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **CatBoost**: [https://catboost.ai/](https://catboost.ai/)

---

## 📩 Contact
If you have any questions about this project, please reach out:
- **Name Surname**: Barış Güleç
- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/barisgulec)
- **Email**: barispvp@hotmail.com
```

