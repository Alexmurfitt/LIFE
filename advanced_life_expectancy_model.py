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

# Cargar los datos
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Detectar columna target robustamente
target_col = [col for col in train.columns if "life" in col.lower() and "expectancy" in col.lower()][0]

# Extraer y eliminar columnas irrelevantes
test_ids = test["Unnamed: 0"] if "Unnamed: 0" in test.columns else test.index
train.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
test.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# Separar X e y
X = train.drop(columns=[target_col])
y = train[target_col]

# Detectar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines de preprocesamiento
numeric_transformer = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Modelos base
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
cat = CatBoostRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, verbose=0)

# Ensamble
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

# Entrenamiento
pipeline.fit(X, y)

# Predicción
preds = pipeline.predict(test)

# Guardar CSV de predicciones
submission = pd.DataFrame({
    "ID": np.arange(1, len(preds) + 1),
    "Life expectancy": preds
})
submission.to_csv("results_advanced.csv", index=False)
print("✅ Archivo generado: results_advanced.csv")