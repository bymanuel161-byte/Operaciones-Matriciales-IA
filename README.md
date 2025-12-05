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

######################################################################
# SISTEMA DE RECOMENDACIÓN - FILTRADO COLABORATIVO CON SVD
######################################################################

PASO 1: Creando matriz de ratings
======================================================================

Matriz de Ratings (0 = película no vista):
          Inception  Matrix  Interstellar  Terminator  Avatar  Blade Runner
Ana               5       3             0           1       0             4
Carlos            4       0             0           1       2             3
Diana             1       1             0           5       4             0
Eduardo           1       0             0           4       0             2
Fernanda          0       1             5           4       0             5


PASO 2: Entrenando el modelo
======================================================================
ENTRENANDO MODELO DE RECOMENDACIÓN
======================================================================

Dimensiones de la matriz: 5 usuarios × 6 items
Sparsity: 36.67%

Aplicando Singular Value Decomposition (SVD)...

Matriz U (usuarios en espacio latente): (5, 5)
Matriz Sigma (valores singulares): (5, 5)
Matriz V^T (items en espacio latente): (5, 6)

Valores Singulares:
  σ_1 = 11.4122
  σ_2 = 6.2891
  σ_3 = 5.7152
  σ_4 = 2.3455
  σ_5 = 2.0108

Reduciendo dimensionalidad a k=3 componentes...

Información retenida con k=3: 95.50%

======================================================================
MODELO ENTRENADO EXITOSAMENTE
======================================================================


PASO 3: Generando predicciones
======================================================================

Predicciones para películas NO VISTAS:
----------------------------------------------------------------------

Ana:
  Interstellar: 0.13 estrellas (predicción)
  Avatar: 0.40 estrellas (predicción)

Carlos:
  Matrix: 1.39 estrellas (predicción)
  Interstellar: 0.00 estrellas (predicción)

Diana:
  Interstellar: 0.00 estrellas (predicción)
  Blade Runner: 0.18 estrellas (predicción)

Eduardo:
  Matrix: 0.61 estrellas (predicción)
  Interstellar: 1.05 estrellas (predicción)
  Avatar: 1.33 estrellas (predicción)

Fernanda:
  Inception: 0.00 estrellas (predicción)
  Avatar: 0.00 estrellas (predicción)


PASO 4: Generando recomendaciones Top-3
======================================================================

Recomendaciones para Ana:
  1. Avatar: 0.40 estrellas
  2. Interstellar: 0.13 estrellas
  3. Blade Runner: 0.00 estrellas

Recomendaciones para Carlos:
  1. Matrix: 1.39 estrellas
  2. Interstellar: 0.00 estrellas
  3. Avatar: 0.00 estrellas

Recomendaciones para Diana:
  1. Blade Runner: 0.18 estrellas
  2. Interstellar: 0.00 estrellas
  3. Avatar: 0.00 estrellas

Recomendaciones para Eduardo:
  1. Avatar: 1.33 estrellas
  2. Interstellar: 1.05 estrellas
  3. Matrix: 0.61 estrellas

Recomendaciones para Fernanda:
  1. Inception: 0.00 estrellas
  2. Avatar: 0.00 estrellas
  3. Blade Runner: 0.00 estrellas


PASO 5: Evaluación del modelo
======================================================================

Métricas de rendimiento:
  RMSE (Root Mean Square Error): 0.4457
  MAE (Mean Absolute Error): 0.3432


PASO 6: Matriz completa de predicciones
======================================================================

Matriz de Predicciones (todos los valores):
          Inception  Matrix  Interstellar  Terminator  Avatar  Blade Runner
Ana            5.14    1.98          0.13        0.77    0.40          4.30
Carlos         3.80    1.39         -0.36        1.48    1.21          2.60
Diana          1.06    0.43         -0.31        5.22    3.70          0.18
Eduardo        0.96    0.61          1.05        3.09    1.33          1.77
Fernanda      -0.01    0.96          4.71        4.26   -0.39          5.02


######################################################################
# DEMOSTRACIÓN COMPLETADA
######################################################################


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
