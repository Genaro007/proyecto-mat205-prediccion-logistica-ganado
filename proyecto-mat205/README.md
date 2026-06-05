# Inferencia Predictiva en Transporte de Ganado

Este repositorio contiene la implementación de un modelo de Machine Learning desarrollado para la materia MAT205. El sistema utiliza el algoritmo **Naive Bayes Gaussiano** para predecir el nivel de riesgo logístico en el transporte de ganado interdepartamental en Bolivia, procesando datos de naturaleza híbrida (variables continuas y categóricas).

## Arquitectura del Proyecto

- `data/transporte_ganado.csv`: Dataset con registros históricos de rutas.
- `src/main.py`: Script principal que procesa los datos, entrena el modelo probabilístico y ejecuta inferencias en tiempo real.
- `requirements.txt`: Dependencias necesarias para el entorno de Python.

## Variables del Modelo

**Predictoras (X):**
- `Distancia_KM` (Continua)
- `Temperatura_C` (Continua)
- `Tipo_Vehiculo` (Categórica: Jaula_Abierta, Termico_Controlado)
- `Condicion_Clima` (Categórica: Soleado, Lluvia, Nublado)

**Objetivo (Y):**
- `Nivel_Riesgo` (Bajo, Medio, Alto)

## Ejecución

1. Clonar el repositorio.
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt