
"""
Recomendación: utilizar un linter como 'pycodestyle' o 'black'
para validar el cumplimiento del estándar PEP8 antes de entregas.
Instalación sugerida: pip install pycodestyle black
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Carga y preparación de datos
data = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.sort_values(by=['Date', 'Hour'], inplace=True)

# Variables temporales adicionales
data['Month'] = data['Date'].dt.month
data['Weekday'] = data['Date'].dt.weekday

# Variables seleccionadas
features = [
    'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)',
    'Snowfall (cm)', 'Seasons', 'Holiday', 'Functioning Day',
    'Month', 'Weekday'
]

target = 'Rented Bike Count'

# División temporal: 80% entrenamiento y 20% prueba
train_size = int(len(data) * 0.8)
X_train = data.iloc[:train_size][features]
y_train = data.iloc[:train_size][target]
X_test = data.iloc[train_size:][features]
y_test = data.iloc[train_size:][target]

# Variables categóricas y numéricas
numeric_features = [
    'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)',
    'Snowfall (cm)', 'Month', 'Weekday'
]

categorical_features = ['Seasons', 'Holiday', 'Functioning Day']

# Pipeline de preprocesamiento sin imputación (no necesaria)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Definir modelos con hiperparámetros
models = {
    'KNN': (KNeighborsRegressor(), {'model__n_neighbors': [3, 5, 7]}),
    'Ridge': (Ridge(), {'model__alpha': [0.1, 1.0, 10.0]}),
    'RandomForest': (
        RandomForestRegressor(random_state=42),
        {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
    )
}

results = {}

# Función para entrenar y evaluar modelos
for name, (model, params) in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    grid_search = GridSearchCV(
        pipeline, params, cv=5,
        scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    results[name] = (best_model, rmse_test, y_test_pred)

# Seleccionar el mejor modelo según RMSE
best_model_name = min(results, key=lambda k: results[k][1])
best_model, best_rmse, best_predictions = results[best_model_name]

# Guardar el mejor modelo
with open('model_prediction_bikerpro.pk', 'wb') as f:
    pickle.dump(best_model, f)

# Gráfica correcta de comparación Real vs Predicciones (temporal)
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Real', linewidth=2)
plt.plot(best_predictions, label='Predicción', linewidth=2, alpha=0.7)
plt.title(f'Comparación Demanda Real vs Predicción - Modelo {best_model_name}')
plt.xlabel('Tiempo (orden cronológico)')
plt.ylabel('Demanda de bicicletas')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('comparative_actual_model_test_set.png')
plt.close()

# Mostrar resultado final en consola
print(
    f"Mejor modelo: {best_model_name}, RMSE conjunto de prueba: {best_rmse:.2f}"
)
