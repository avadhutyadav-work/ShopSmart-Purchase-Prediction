# ShopSmart – Purchase Prediction

A machine learning project that predicts whether an online-store visitor will make a purchase, based on their session behaviour. The model is deployed as a **Streamlit** web app.

**Live app:** _add your Streamlit link here_

---

## Problem Statement
ShopSmart, an e-commerce company, cannot accurately predict which visitors will buy. This leads to inefficient marketing and lost revenue.

The dataset has **12,330 user sessions** (one session per visitor) with numerical and categorical features about browsing behaviour. The goal is to build a **Decision Tree classifier** that predicts `Revenue` (purchase or no purchase).

The data is **imbalanced** (only a small share of visitors buy), so the model is judged with the **F1 score**, with a benchmark of **0.55**.

## Project Steps
1. **EDA** – null values, duplicates, class balance, box plots, purchase rate by visitor type and month, correlation heatmap
2. **Preprocessing** – label encoding of categorical columns, train/test split (80/20, stratified), standard scaling
3. **Model** – Decision Tree with `class_weight="balanced"` to handle the imbalance
4. **Pruning** – cost-complexity pruning (`ccp_alpha`) with `max_depth` and `min_samples_leaf` tuning, using GridSearchCV with 5-fold cross-validation scored by F1
5. **Evaluation** – F1 score, precision, recall, confusion matrix
6. **Deployment** – Streamlit app

## Results
| Metric | Value |
|---|---|
| F1 score | _add your value_ |
| Precision | _add your value_ |
| Recall | _add your value_ |
| Benchmark F1 | 0.55 |

## Features Used
| Feature | Description |
|---|---|
| Administrative, Informational, ProductRelated | Number of pages visited in each category |
| *_Duration | Time spent on each page category |
| BounceRates, ExitRates | Share of visitors who leave from the page |
| PageValues | Average value of the pages visited before a purchase |
| SpecialDay | Closeness of the visit to a special day |
| Month | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | Visitor's technical and location details |
| VisitorType | New, Returning or Other visitor |
| Weekend | Whether the visit was on a weekend |
| **Revenue** | **Target:** whether the visitor made a purchase |

## Repository Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `model.pkl` | Trained (pruned) Decision Tree |
| `scaler.pkl` | StandardScaler fitted on the training data |
| `columns.pkl` | Feature names, in the order the model expects |
| `requirements.txt` | Python libraries needed |
| `Shop_smart_ecommerce.ipynb` | Notebook with EDA, training and evaluation (optional, if you upload it) |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

## Tech Stack
Python, pandas, scikit-learn, seaborn, matplotlib, Streamlit

## Author
_Your name_
