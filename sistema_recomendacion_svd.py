
"""
APLICACIÓN PRÁCTICA: Sistema de Recomendación con Filtrado Colaborativo
Autor: Manuel Ortiz
Curso: Álgebra Lineal Aplicada a IA
Tema: Operaciones Matriciales en Sistemas de Recomendación
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt

class SistemaRecomendacion:
    """
    Implementación de un sistema de recomendación basado en
    Filtrado Colaborativo usando Descomposición en Valores Singulares (SVD)
    """

    def __init__(self, n_components=3):
        """
        Inicializar el sistema de recomendación

        Args:
            n_components: Número de componentes latentes a usar
        """
        self.n_components = n_components
        self.U = None
        self.sigma = None
        self.Vt = None
        self.matriz_predicciones = None

    def entrenar(self, matriz_ratings):
        """
        Entrenar el modelo usando SVD

        Args:
            matriz_ratings: Matriz numpy de usuarios x items
        """
        print("=" * 70)
        print("ENTRENANDO MODELO DE RECOMENDACIÓN")
        print("=" * 70)

        # Guardar dimensiones originales
        n_usuarios, n_items = matriz_ratings.shape
        print(f"\nDimensiones de la matriz: {n_usuarios} usuarios × {n_items} items")

        # Calcular sparsity
        total_ratings = n_usuarios * n_items
        ratings_conocidos = np.count_nonzero(matriz_ratings)
        sparsity = 100 * (1 - ratings_conocidos / total_ratings)
        print(f"Sparsity: {sparsity:.2f}%")

        # Aplicar SVD
        print("\nAplicando Singular Value Decomposition (SVD)...")
        self.U, sigma_valores, self.Vt = svd(matriz_ratings, full_matrices=False)

        # Crear matriz diagonal Sigma
        self.sigma = np.diag(sigma_valores)

        print(f"\nMatriz U (usuarios en espacio latente): {self.U.shape}")
        print(f"Matriz Sigma (valores singulares): {self.sigma.shape}")
        print(f"Matriz V^T (items en espacio latente): {self.Vt.shape}")

        # Mostrar valores singulares
        print(f"\nValores Singulares:")
        for i, val in enumerate(sigma_valores):
            print(f"  σ_{i+1} = {val:.4f}")

        # Reducción de dimensionalidad
        print(f"\nReduciendo dimensionalidad a k={self.n_components} componentes...")
        U_reducida = self.U[:, :self.n_components]
        Sigma_reducida = self.sigma[:self.n_components, :self.n_components]
        Vt_reducida = self.Vt[:self.n_components, :]

        # Calcular matriz de predicciones
        self.matriz_predicciones = U_reducida @ Sigma_reducida @ Vt_reducida

        # Calcular porcentaje de información retenida
        varianza_total = np.sum(sigma_valores**2)
        varianza_retenida = np.sum(sigma_valores[:self.n_components]**2)
        porcentaje_retenido = 100 * varianza_retenida / varianza_total

        print(f"\nInformación retenida con k={self.n_components}: {porcentaje_retenido:.2f}%")
        print("\n" + "=" * 70)
        print("MODELO ENTRENADO EXITOSAMENTE")
        print("=" * 70)

    def predecir_rating(self, usuario_id, item_id):
        """
        Predecir rating para un usuario e item específico

        Args:
            usuario_id: Índice del usuario
            item_id: Índice del item

        Returns:
            Rating predicho
        """
        if self.matriz_predicciones is None:
            raise ValueError("Modelo no entrenado. Llama a entrenar() primero.")

        prediccion = self.matriz_predicciones[usuario_id, item_id]

        # Limitar predicción al rango válido [0, 5]
        prediccion = np.clip(prediccion, 0, 5)

        return prediccion

    def recomendar_top_n(self, usuario_id, n=5, items_ya_vistos=None):
        """
        Recomendar top N items para un usuario

        Args:
            usuario_id: Índice del usuario
            n: Número de recomendaciones
            items_ya_vistos: Lista de items ya consumidos por el usuario

        Returns:
            Lista de tuplas (item_id, rating_predicho)
        """
        if self.matriz_predicciones is None:
            raise ValueError("Modelo no entrenado. Llama a entrenar() primero.")

        # Obtener predicciones para este usuario
        predicciones_usuario = self.matriz_predicciones[usuario_id, :]

        # Si hay items ya vistos, ponerles rating muy bajo
        if items_ya_vistos is not None:
            predicciones_usuario = predicciones_usuario.copy()
            for item_idx in items_ya_vistos:
                predicciones_usuario[item_idx] = -999

        # Obtener índices de top N items
        top_items_idx = np.argsort(predicciones_usuario)[::-1][:n]

        # Crear lista de recomendaciones
        recomendaciones = [
            (item_idx, np.clip(predicciones_usuario[item_idx], 0, 5))
            for item_idx in top_items_idx
        ]

        return recomendaciones

    def evaluar_modelo(self, matriz_original):
        """
        Evaluar el modelo calculando RMSE y MAE

        Args:
            matriz_original: Matriz original de ratings

        Returns:
            Dict con métricas
        """
        # Calcular error solo en ratings conocidos (no-cero)
        mascara = matriz_original != 0

        diferencias = matriz_original[mascara] - self.matriz_predicciones[mascara]

        rmse = np.sqrt(np.mean(diferencias**2))
        mae = np.mean(np.abs(diferencias))

        return {
            'RMSE': rmse,
            'MAE': mae
        }


def demo_sistema_recomendacion():
    """
    Demostración completa del sistema de recomendación
    """
    print("\n\n")
    print("#" * 70)
    print("# SISTEMA DE RECOMENDACIÓN - FILTRADO COLABORATIVO CON SVD")
    print("#" * 70)
    print()

    # 1. Crear matriz de ratings de ejemplo
    print("PASO 1: Creando matriz de ratings")
    print("=" * 70)

    # Matriz 5 usuarios × 6 películas
    # Ratings: 1-5 estrellas, 0 = no visto
    matriz_ratings = np.array([
        [5, 3, 0, 1, 0, 4],  # Usuario 0
        [4, 0, 0, 1, 2, 3],  # Usuario 1
        [1, 1, 0, 5, 4, 0],  # Usuario 2
        [1, 0, 0, 4, 0, 2],  # Usuario 3
        [0, 1, 5, 4, 0, 5]   # Usuario 4
    ])

    nombres_peliculas = [
        "Inception", "Matrix", "Interstellar", 
        "Terminator", "Avatar", "Blade Runner"
    ]

    nombres_usuarios = [
        "Ana", "Carlos", "Diana", "Eduardo", "Fernanda"
    ]

    # Crear DataFrame para visualización
    df_ratings = pd.DataFrame(
        matriz_ratings,
        index=nombres_usuarios,
        columns=nombres_peliculas
    )

    print("\nMatriz de Ratings (0 = película no vista):")
    print(df_ratings)
    print()

    # 2. Entrenar el modelo
    print("\nPASO 2: Entrenando el modelo")
    sistema = SistemaRecomendacion(n_components=3)
    sistema.entrenar(matriz_ratings)

    # 3. Hacer predicciones
    print("\n\nPASO 3: Generando predicciones")
    print("=" * 70)

    print("\nPredicciones para películas NO VISTAS:")
    print("-" * 70)

    for i, usuario in enumerate(nombres_usuarios):
        print(f"\n{usuario}:")
        for j, pelicula in enumerate(nombres_peliculas):
            if matriz_ratings[i, j] == 0:  # Solo películas no vistas
                rating_pred = sistema.predecir_rating(i, j)
                print(f"  {pelicula}: {rating_pred:.2f} estrellas (predicción)")

    # 4. Generar recomendaciones Top-N
    print("\n\nPASO 4: Generando recomendaciones Top-3")
    print("=" * 70)

    for i, usuario in enumerate(nombres_usuarios):
        # Encontrar películas ya vistas
        items_vistos = [j for j in range(len(nombres_peliculas)) 
                       if matriz_ratings[i, j] != 0]

        # Obtener recomendaciones
        recomendaciones = sistema.recomendar_top_n(i, n=3, items_ya_vistos=items_vistos)

        print(f"\nRecomendaciones para {usuario}:")
        for rank, (item_idx, rating) in enumerate(recomendaciones, 1):
            print(f"  {rank}. {nombres_peliculas[item_idx]}: {rating:.2f} estrellas")

    # 5. Evaluar el modelo
    print("\n\nPASO 5: Evaluación del modelo")
    print("=" * 70)

    metricas = sistema.evaluar_modelo(matriz_ratings)
    print(f"\nMétricas de rendimiento:")
    print(f"  RMSE (Root Mean Square Error): {metricas['RMSE']:.4f}")
    print(f"  MAE (Mean Absolute Error): {metricas['MAE']:.4f}")

    # 6. Visualizar matriz completa de predicciones
    print("\n\nPASO 6: Matriz completa de predicciones")
    print("=" * 70)

    df_predicciones = pd.DataFrame(
        np.round(sistema.matriz_predicciones, 2),
        index=nombres_usuarios,
        columns=nombres_peliculas
    )

    print("\nMatriz de Predicciones (todos los valores):")
    print(df_predicciones)

    print("\n\n" + "#" * 70)
    print("# DEMOSTRACIÓN COMPLETADA")
    print("#" * 70)
    print()

    return sistema, matriz_ratings


if __name__ == "__main__":
    # Ejecutar demostración
    sistema, matriz_original = demo_sistema_recomendacion()

   
