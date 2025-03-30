
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Cargar el archivo CSV
data = pd.read_csv('/mnt/data/SeoulBikeData.csv', encoding='ISO-8859-1')

# Preprocesamiento
# Convertir las columnas categóricas en variables numéricas
data['Seasons'] = data['Seasons'].map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3})
data['Holiday'] = data['Holiday'].map({'No Holiday': 0, 'Holiday': 1})
data['Functioning Day'] = data['Functioning Day'].map({'Yes': 1, 'No': 0})

# Separar las características y la variable objetivo
X = data.drop(columns=['Rented Bike Count', 'Date'])
y = data['Rented Bike Count']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento de las variables numéricas y categóricas
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline para preprocesar los datos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores faltantes
    ('scaler', StandardScaler())  # Escalado de las características numéricas
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación de valores faltantes
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificación one-hot de variables categóricas
])

# Combinar ambos pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear un pipeline completo para cada modelo
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Modelos a entrenar
models = {
    'KNN': KNeighborsRegressor(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor()
}

# Calibración de hiperparámetros utilizando GridSearchCV para cada modelo
param_grid_knn = {'model__n_neighbors': [3, 5, 7, 9], 'model__weights': ['uniform', 'distance']}
param_grid_ridge = {'model__alpha': [0.1, 1, 10, 100]}
param_grid_rf = {'model__n_estimators': [50, 100, 150], 'model__max_depth': [None, 10, 20]}

# Función para realizar GridSearch y evaluación del modelo
def train_and_evaluate(model_name, model, param_grid):
    print(f"Entrenando modelo {model_name}...")
    
    grid_search = GridSearchCV(create_pipeline(model), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    # Predicciones y evaluación
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calcular RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"RMSE en conjunto de entrenamiento para {model_name}: {rmse_train}")
    print(f"RMSE en conjunto de prueba para {model_name}: {rmse_test}")
    
    return best_model, rmse_train, rmse_test, y_train_pred, y_test_pred

# Entrenar y evaluar todos los modelos
results = {}
for model_name, model in models.items():
    if model_name == 'KNN':
        param_grid = param_grid_knn
    elif model_name == 'Ridge':
        param_grid = param_grid_ridge
    elif model_name == 'RandomForest':
        param_grid = param_grid_rf
    
    best_model, rmse_train, rmse_test, y_train_pred, y_test_pred = train_and_evaluate(model_name, model, param_grid)
    results[model_name] = {
        'best_model': best_model,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# Guardar el mejor modelo (por ejemplo, el de menor RMSE en el conjunto de prueba)
best_model_overall = min(results.items(), key=lambda x: x[1]['rmse_test'])[1]['best_model']
with open('model_prediction_bikerpro.pk', 'wb') as f:
    pickle.dump(best_model_overall, f)

# Generar las gráficas de comparación
def plot_comparison(actual, predicted, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Real')
    plt.plot(predicted, label='Predicho', linestyle='--')
    plt.title(title)
    plt.xlabel('Observaciones')
    plt.ylabel('Demanda de bicicletas')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Generar las gráficas de comparación
plot_comparison(y_train, results['KNN']['y_train_pred'], "Comparativa Real vs Predicción - Conjunto de Entrenamiento", "comparative_actual_model_train_set.png")
plot_comparison(y_test, results['KNN']['y_test_pred'], "Comparativa Real vs Predicción - Conjunto de Prueba", "comparative_actual_model_test_set.png")
