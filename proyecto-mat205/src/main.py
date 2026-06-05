import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    label_encoders = {}
    categorical_cols = ['Tipo_Vehiculo', 'Condicion_Clima']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df[['Distancia_KM', 'Temperatura_C', 'Tipo_Vehiculo', 'Condicion_Clima']]
    y = df['Nivel_Riesgo']
    
    return X, y, label_encoders

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Métricas de Evaluación (Datos de Prueba):")
    print(f"Precisión: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_risk(model, encoders, distancia, temperatura, vehiculo, clima):
    vehiculo_encoded = encoders['Tipo_Vehiculo'].transform([vehiculo])[0]
    clima_encoded = encoders['Condicion_Clima'].transform([clima])[0]
    
    features = np.array([[distancia, temperatura, vehiculo_encoded, clima_encoded]])
    prediction = model.predict(features)
    
    return prediction[0]

if __name__ == "__main__":
    filepath = '../data/transporte_ganado.csv'
    
    try:
        X, y, encoders = load_and_preprocess_data(filepath)
        modelo_nb = train_model(X, y)
        
        print("-" * 50)
        print("INFERENCIA EN TIEMPO REAL")
        print("-" * 50)
        
        test_distancia = 450
        test_temperatura = 37
        test_vehiculo = 'Jaula_Abierta'
        test_clima = 'Soleado'
        
        riesgo = predict_risk(modelo_nb, encoders, test_distancia, test_temperatura, test_vehiculo, test_clima)
        
        print(f"Parámetros de entrada:")
        print(f"Distancia: {test_distancia} KM | Temperatura: {test_temperatura}°C | Vehículo: {test_vehiculo} | Clima: {test_clima}")
        print(f">>> RIESGO PREDICHO: {riesgo.upper()} <<<")
        print("-" * 50)
        
    except FileNotFoundError:
        print("Error: Asegúrate de ejecutar el script desde la carpeta 'src' y que el CSV exista en 'data/'.")