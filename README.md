# Public Procurement Auction Price Decrease Prediction

## Executive Summary

This project develops a machine learning pipeline for predicting price decreases in public procurement auctions.

The model uses auction-level features such as starting price, bid count, auction duration, step size, participant activity, and lot descriptions to forecast the final price decrease. The project combines data cleaning, feature engineering, text processing, CatBoost models, Ridge regression with TF-IDF features, cross-validation, and ensemble blending.

The final solution achieved:

| Metric | Value |
|--------|-------|
| Validation Metric | Mean Absolute Error (MAE) |
| Final OOF MAE | 4.4888 |
| Kaggle / External MAE | 4.6634 |

The project demonstrates an end-to-end machine learning workflow: from raw auction data preprocessing to model validation and final submission generation.

---

## Business Problem

Public procurement auctions are designed to reduce purchasing costs through competitive bidding.

However, the final price decrease can vary significantly depending on auction structure, starting price, number of bidders, auction duration, bidding rules, and lot characteristics.

The business problem addressed in this project is:

**Can we predict the expected price decrease in a public procurement auction using auction-level features?**

A reliable prediction model can support:

- procurement planning;
- budget forecasting;
- auction design optimization;
- early identification of low-competition auctions;
- better understanding of factors associated with stronger price reductions.

---

## Project Objective

The objective was to build a machine learning model that predicts the expected price decrease in public procurement auctions.

The target variable is:

| Target | Description |
|--------|-------------|
| PriceDynamicsString | Final price decrease in the auction |

The evaluation metric is:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error between predicted and actual price decrease |

---

## Dataset

The dataset contains structured, categorical, and textual auction features.

### Main Feature Groups

| Feature Group | Examples |
|--------------|----------|
| Price Features | StartingPrice, MinStep, MaxStep |
| Auction Activity | BidCount, AuctionDurationSec |
| Participant Features | number, sleep, active participants |
| Categorical Features | StepType, WithStep, CanImproveBid, WithSecondPhase |
| Text Features | LotName |
| Target Variable | PriceDynamicsString |

---

## Methodology

### 1. Data Cleaning

A custom cleaning function `strict_fix()` was developed to prepare raw auction data.

Main steps:

- handled missing and invalid values;
- replaced negative values with missing values;
- corrected inconsistent minimum and maximum bid steps;
- fixed inconsistent participant activity fields;
- clipped extreme values using quantiles;
- applied square-root transformation to reduce long tails;
- normalized text fields in auction lot names;
- ensured consistent numeric types across train and test data.

---

### 2. Feature Engineering

A custom feature engineering function `add_features()` was used to create auction-level predictive signals.

Key engineered features:

| Feature | Description |
|---------|-------------|
| log_price | Log-transformed starting price |
| duration_h | Auction duration in hours |
| bids_per_h | Number of bids per auction hour |
| actives | Estimated active participants |
| activity_ratio | Share of active participants |
| bid_density | Bids per active participant |
| step_mean | Average bid step |
| step_rel | Bid step relative to starting price |
| step_range | Difference between maximum and minimum bid step |
| log_bidcount | Log-transformed bid count |
| bids_x_step | Interaction between bid count and relative bid step |
| efficiency | Active participants relative to auction duration |
| bid_speed | Bids per second |
| dur_per_bid | Auction duration per bid |
| price_per_participant | Starting price per active participant |
| bids_per_sleep | Bid count relative to inactive participants |

These features were designed to capture auction intensity, competition level, price structure, and participant behavior.

---

### 3. Exploratory Data Analysis

The analysis included:

- before/after cleaning distribution checks;
- histograms of transformed numerical features;
- correlation matrix;
- pairplot of core auction variables;
- inspection of engineered feature distributions.

The EDA confirmed that engineered features such as `bids_per_h`, `activity_ratio`, and `price_per_participant` added useful signals beyond the original auction variables.

---

## Modeling Approach

### CatBoost Models

Several CatBoostRegressor models were trained using different hyperparameter configurations.

CatBoost was selected because it performs well with:

- tabular data;
- categorical variables;
- nonlinear relationships;
- heterogeneous feature types;
- robust regression tasks.

The CatBoost models were trained using K-Fold out-of-fold validation.

| Model | Role |
|-------|------|
| CatBoost A | Stable baseline model |
| CatBoost B | Deeper and more flexible model |
| CatBoost C | Tuned model configuration |
| CatBoost D | Regularized stabilizer |

The CatBoost predictions were blended by optimizing validation MAE.

---

### Ridge Regression with TF-IDF

A separate Ridge regression model was trained to capture textual patterns from `LotName`.

The Ridge pipeline included:

- median imputation for numerical features;
- robust scaling;
- one-hot encoding of categorical features;
- TF-IDF vectorization of lot names;
- RidgeCV with cross-validation.

Text features were represented using TF-IDF n-grams.

This model was added because tree-based models may not fully capture useful signals from textual lot descriptions.

---

### Final Ensemble

The final prediction combined:

- CatBoost ensemble predictions;
- Ridge + TF-IDF predictions.

The blend weight was selected by minimizing out-of-fold MAE.

This ensemble approach improved stability and reduced dependence on a single model type.

---

## Results

| Model Component | Purpose |
|----------------|---------|
| CatBoost Ensemble | Captures nonlinear tabular and categorical patterns |
| Ridge + TF-IDF | Captures text-based signals from lot names |
| Final Blend | Combines tabular and text-based predictive signals |

Final performance:

| Result | MAE |
|--------|-----|
| Out-of-Fold Validation | 4.4888 |
| Kaggle / External Score | 4.6634 |

---

## Reproducibility

The full pipeline is automated:

```text
Raw Data
→ Data Cleaning
→ Feature Engineering
→ Exploratory Analysis
→ CatBoost Modeling
→ Ridge + TF-IDF Modeling
→ Ensemble Blending
→ Submission Generation
```

Reproducibility measures:

- fixed random seed (`random_state=42`);
- K-Fold validation;
- automated preprocessing functions;
- final predictions exported to `submission.csv`.

---

## Business Value

This project shows how machine learning can support procurement analytics.

Potential use cases include:

- forecasting expected auction savings;
- identifying auctions with weak expected competition;
- improving procurement budget planning;
- evaluating auction design parameters;
- supporting data-driven procurement strategy.

The model does not replace procurement experts, but it can provide an analytical layer for understanding which auction characteristics are associated with stronger price reductions.

---

## Limitations

Several limitations should be considered:

- The project was developed in an exam / competition setting.
- The dataset structure depends on the provided train/test files.
- External validation depends on the competition scoring system.
- The model focuses on predictive performance, not causal interpretation.
- Additional business context could improve interpretability of auction dynamics.

Future improvements could include feature importance analysis, SHAP interpretation, model monitoring, and deployment as a lightweight prediction tool.

---

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- RidgeCV
- TF-IDF
- K-Fold Cross-Validation
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Skills Demonstrated

| Machine Learning | Data Engineering | NLP & Text Analytics | Business Analytics |
|------------------|------------------|----------------------|--------------------|
| Regression Modeling | Data Cleaning | Text Normalization | Procurement Analytics |
| CatBoost | Feature Engineering | TF-IDF Vectorization | Auction Performance Analysis |
| Ensemble Learning | Pipeline Automation | N-gram Features | Price Decrease Forecasting |
| Cross-Validation | Train/Test Processing | Text-Based Prediction Signals | Budget Planning Support |
| MAE Optimization | Reproducible Workflow | Ridge Regression on Text | Decision Support |

---

## Author

**Maksim Dumanov**

HSE University — International Programme in Business and Economics

Research Areas: Data Analytics • Machine Learning • Business Analytics • Procurement Analytics
