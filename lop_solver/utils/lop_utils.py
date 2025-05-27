"""
Этот файл хранит функции, которые будут использованы во многих методах решения.
"""

import numpy as np
from numba import njit
from typing import List

@njit
def calculate_lop_cost(matrix: np.ndarray, ordering: List[int]) -> int:
    """Вычисляет значение LOP для данной перестановки."""
    cost = 0
    n = len(ordering)
    for i in range(n):
        for j in range(i + 1, n):
            cost += matrix[ordering[i], ordering[j]]
    return cost

def calculate_lop_cost_fast(matrix: np.ndarray, ordering: List[int]) -> int:
    ordered_matrix = matrix[np.ix_(ordering, ordering)] # Сортировка матрицы
    return np.sum(np.triu(ordered_matrix, k=1)) # Фильтрация + суммиирование

def generate_neighborhood(ordering: List[int]) -> List[List[int]]:
    """Генерирует соседние перестановки (для локального поиска)."""
    neighborhood = []
    # Пример: все возможные swap-соседи
    for i in range(len(ordering)):
        for j in range(i + 1, len(ordering)):
            new_ordering = ordering.copy()
            new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]
            neighborhood.append(new_ordering)
    return neighborhood

def is_valid_permutation(ordering: List[int], n: int) -> bool:
    """Проверяет, является ли ordering валидной перестановкой для матрицы nxn."""
    return sorted(ordering) == list(range(n))