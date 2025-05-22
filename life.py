# life.py — versión definitiva del modelo más avanzado para Life Expectancy Challenge

import sys
import subprocess

# Instalador automático de dependencias

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in ["optuna", "xgboost", "lightgbm", "catboost"]:
    install_if_missing(package)

# === Librerías ===
import numpy as np
import pandas as pd
import optuna
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

# === Configuración ===
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# === Carga de datos ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

if "Unnamed: 0" in train.columns:
    train.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
if "Unnamed: 0" in test.columns:
    test.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# === Preprocesamiento ===
cols_zero_as_na = ['Schooling', 'Income composition of resources', 'percentage expenditure']
for col in cols_zero_as_na:
    train[col] = train[col].replace(0, np.nan)
    test[col] = test[col].replace(0, np.nan)

train['Population'] = train['Population'].replace(1386542.0, np.nan)
test['Population'] = test['Population'].replace(1386542.0, np.nan)

train['Status'] = train['Status'].map({'Developing': 0, 'Developed': 1})
test['Status'] = test['Status'].map({'Developing': 0, 'Developed': 1})

# Target Encoding con KFold
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
train_country_te = np.zeros(len(train))
for tr_idx, val_idx in kf.split(train):
    fold_mean = train.iloc[tr_idx].groupby('Country')['Life expectancy'].mean()
    train_country_te[val_idx] = train.iloc[val_idx]['Country'].map(fold_mean)
global_mean = train['Life expectancy'].mean()
train_country_te = np.where(np.isnan(train_country_te), global_mean, train_country_te)
train['Country_TE'] = train_country_te

country_mean_full = train.groupby('Country')['Life expectancy'].mean()
test['Country_TE'] = test['Country'].map(country_mean_full).fillna(global_mean)

# === Selección de features ===
# Lista completa de columnas esperadas
all_features = ['Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure',
                'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness 1-19 years',
                'thinness 5-9 years', 'Income composition of resources', 'Schooling', 'Country_TE']

# Filtrar solo las columnas que realmente existen en el dataframe
features = [col for col in all_features if col in train.columns]

# Asignar variables de entrada
X_train = train[features].copy()
X_test = test[features].copy()

y_train = train['Life expectancy'].copy()
X_test = test[features].copy()

# Imputaciones (KNN + Iterative)
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train)
X_test_knn = knn_imputer.transform(X_test)

imp_estimator = ExtraTreesRegressor(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
iter_imputer = IterativeImputer(estimator=imp_estimator, max_iter=10, initial_strategy='mean',
                                imputation_order='ascending', random_state=RANDOM_SEED)
X_train_imp = iter_imputer.fit_transform(X_train_knn)
X_test_imp = iter_imputer.transform(X_test_knn)

X_train_imp_df = pd.DataFrame(X_train_imp, columns=features)
X_test_imp_df = pd.DataFrame(X_test_imp, columns=features)

# Outliers y escalado robusto
lower_bounds = X_train_imp_df.quantile(0.01)
upper_bounds = X_train_imp_df.quantile(0.99)
X_train_imp_df = X_train_imp_df.clip(lower_bounds, upper_bounds, axis=1)
X_test_imp_df = X_test_imp_df.clip(lower_bounds, upper_bounds, axis=1)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imp_df)
X_test_scaled = scaler.transform(X_test_imp_df)

# === Ingeniería de características polinómicas ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# === Selección de características con LightGBM ===
lgb_temp = LGBMRegressor(n_estimators=200, random_state=RANDOM_SEED)
lgb_temp.fit(X_train_poly, y_train)
importances = lgb_temp.booster_.feature_importance(importance_type='gain')
topN = 150
top_indices = np.argsort(importances)[::-1][:topN]
X_train_sel = X_train_poly[:, top_indices]
X_test_sel = X_test_poly[:, top_indices]

# === Entrenamiento del modelo con Stacking ===
model_xgb = XGBRegressor(**{'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.85,
                            'colsample_bytree': 0.85, 'gamma': 0.1, 'min_child_weight': 2,
                            'reg_alpha': 0.5, 'reg_lambda': 1.0, 'random_state': RANDOM_SEED, 'tree_method': 'hist'})

model_lgb = LGBMRegressor(**{'n_estimators': 650, 'num_leaves': 64, 'learning_rate': 0.03, 'feature_fraction': 0.85,
                              'bagging_fraction': 0.85, 'lambda_l1': 0.1, 'lambda_l2': 0.2, 'min_child_samples': 10,
                              'random_state': RANDOM_SEED, 'n_jobs': -1})

model_cat = CatBoostRegressor(**{'iterations': 700, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0,
                                  'random_strength': 1.0, 'bagging_temperature': 0.3, 'random_state': RANDOM_SEED, 'verbose': 0})

model_hist = HistGradientBoostingRegressor(**{'max_iter': 700, 'max_depth': 7, 'learning_rate': 0.03,
                                              'l2_regularization': 0.3, 'early_stopping': False,
                                              'random_state': RANDOM_SEED})

meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
stacking_reg = StackingRegressor(
    estimators=[
        ('xgb', model_xgb),
        ('lgb', model_lgb),
        ('cat', model_cat),
        ('hist', model_hist)
    ],
    final_estimator=meta_model,
    cv=10,
    n_jobs=-1
)

stacking_reg.fit(X_train_sel, y_train)
final_cv_scores = cross_val_score(stacking_reg, X_train_sel, y_train, cv=10, scoring='r2')
print(f"Stacking Regressor 10-fold CV R²: {np.mean(final_cv_scores):.5f} ± {np.std(final_cv_scores):.5f}")

# === Predicción final ===
test_predictions = stacking_reg.predict(X_test_sel)
output = pd.DataFrame({'ID': test['ID'], 'Life expectancy': test_predictions})
output.to_csv("results_ultra_model.csv", index=False)
print("✅ Archivo generado: results_ultra_model.csv")
