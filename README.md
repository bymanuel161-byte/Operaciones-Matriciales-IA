# Operaciones-Matriciales-IA
Aplicación de Operaciones Matriciales en IA - Sistemas de Recomendación con SVD
# Aplicación de Operaciones Matriciales en Inteligencia Artificial

Sistemas de Recomendación mediante Filtrado Colaborativo con SVD

## Descripción

Este proyecto implementa un sistema de recomendación basado en filtrado colaborativo 
utilizando Descomposición en Valores Singulares (SVD) para predecir preferencias de usuarios.

## Archivos

- `sistema_recomendacion_svd.py`: Implementación completa del sistema

## Requisitos

numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0

## Instalación

pip install -r requirements.txt
python sistema_recomendacion_svd.py

## Uso

from sistema_recomendacion_svd import SistemaRecomendacion
import numpy as np

Crear sistema
sistema = SistemaRecomendacion(n_components=3)

Entrenar con matriz de ratings
matriz_ratings = np.array([...])
sistema.entrenar(matriz_ratings)

Obtener recomendaciones
recomendaciones = sistema.recomendar_top_n(usuario_id=0, n=5)

## Resultados

- **RMSE:** 0.4457
- **MAE:** 0.3432
- **Información retenida (k=3):** 95.50%

## Autor

Manuel Ortiz - ID: 2025-1659

## Curso

Álgebra Lineal Aplicada a Inteligencia Artificial

## Fecha

Diciembre 2025
