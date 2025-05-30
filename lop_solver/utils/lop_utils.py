"""
Этот файл хранит функции, которые будут использованы во многих методах решения.
"""

import numpy as np
import random
from numba import njit
from typing import List, Optional

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



def generate_neighborhood(ordering: List[int], 
                         neighborhood_type: str = 'swap',
                         size: Optional[int] = None) -> List[List[int]]:
    """
    Генерирует окрестность решения для алгоритма Великого потопа
    
    Параметры:
    -----------
    ordering : List[int]
        Текущая перестановка
    neighborhood_type : str
        Тип окрестности ('swap', 'insert', 'reverse')
    size : int, optional
        Максимальный размер окрестности (если None - полная окрестность)
    
    Возвращает:
    ------------
    List[List[int]]: Список соседних решений
    """
    n = len(ordering)
    neighborhood = []
    
    if neighborhood_type == 'swap':
        # Окрестность обмена: все возможные обмены двух элементов
        for i in range(n):
            for j in range(i+1, n):
                new_ordering = ordering.copy()
                new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]
                neighborhood.append(new_ordering)
    
    elif neighborhood_type == 'insert':
        # Окрестность вставки: перемещение элемента на другую позицию
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_ordering = ordering.copy()
                item = new_ordering.pop(i)
                new_ordering.insert(j, item)
                neighborhood.append(new_ordering)
    
    elif neighborhood_type == 'reverse':
        # Окрестность инверсии: инверсия подпоследовательности
        for i in range(n):
            for j in range(i+2, n+1):
                new_ordering = ordering.copy()
                new_ordering[i:j] = reversed(new_ordering[i:j])
                neighborhood.append(new_ordering)
    
    # Ограничение размера окрестности
    if size is not None and len(neighborhood) > size:
        neighborhood = random.sample(neighborhood, size)
    
    return neighborhood