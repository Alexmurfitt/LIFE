import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Cargar datos
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Verificar nombre exacto de la columna objetivo
target_col = [col for col in train.columns if "life" in col.lower() and "expect" in col.lower()][0]

# Extraer y eliminar ID
test_ids = test["Unnamed: 0"]
train.drop(columns=["Unnamed: 0"], inplace=True)
test.drop(columns=["Unnamed: 0"], inplace=True)

# Separar target y features
X = train.drop(columns=[target_col])
y = train[target_col]

# Identificar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines de preprocesado
numeric_transformer = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Modelos base
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
cat = CatBoostRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, verbose=0)

# Ensemble
ensemble = VotingRegressor([
    ("xgb", xgb),
    ("lgbm", lgbm),
    ("cat", cat)
])

# Pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ensemble)
])

# Entrenar modelo
pipeline.fit(X, y)

# Predecir sobre test
preds = pipeline.predict(test)

# Generar archivo de predicción
# Guardar CSV
submission = pd.DataFrame({
    "ID": np.arange(1, len(preds) + 1),
    "Life expectancy": preds
})
submission.to_csv("results_predictor.csv", index=False)
print("✅ Archivo generado: results_predictor.csv")