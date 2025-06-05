"""
Этот файл хранит функции, которые будут использованы во многих методах решения.
"""

import numpy as np
import random
from numba import njit
from typing import List, Optional, Tuple

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



def _generate_neighborhood(ordering: List[int], 
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


def generate_random_neighbor(solution: List[int], neighborhood_type: str) -> Tuple[List[int], int, int]:
    n = len(solution)
    if neighborhood_type == 'swap':
        i, j = random.sample(range(n), 2)
        neighbor = solution.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor, i, j
    elif neighborhood_type == 'insert':
        i, j = random.sample(range(n), 2)
        neighbor = solution.copy()
        elem = neighbor.pop(i)
        if j > i:
            j -= 1
        neighbor.insert(j, elem)
        return neighbor, i, j
    elif neighborhood_type == 'reverse':
        i, j = sorted(random.sample(range(n), 2))
        neighbor = solution.copy()
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        return neighbor, i, j
    else:
        raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")
    

def calc_insert_cost_(matrix: np.ndarray, old_cost: int, old_ordering: List[int], i: int, j: int):
    if i == j:
        return 0
    if i < j:

        a_i = old_ordering[i]
        a_j = old_ordering[j]
        
        # Разница для пары (i, j)
        # delta = matrix[a_j, a_i] - matrix[a_i, a_j]
        delta = 0
        
        # Суммирование по всем k между i и j
        for k in range(i + 1, j):
            a_k = old_ordering[k]
            # delta += matrix[a_j, a_k] - matrix[a_i, a_k]
            # delta += matrix[a_k, a_i] - matrix[a_k, a_j]
            delta += matrix[a_k, a_i] - matrix[a_i, a_k]
            
        return old_cost + delta
    if i > j:
        a_i = old_ordering[i]
        a_j = old_ordering[j]
        
        # Разница для пары (i, j)
        # delta = matrix[a_j, a_i] - matrix[a_i, a_j]
        delta = 0
        
        # Суммирование по всем k между i и j
        for k in range(j, i-1):
            a_k = old_ordering[k]
            # delta += matrix[a_j, a_k] - matrix[a_i, a_k]
            # delta += matrix[a_k, a_i] - matrix[a_k, a_j]
            delta += matrix[a_i, a_k] - matrix[a_k, a_i]
            
        return old_cost + delta


def calc_insert_cost(matrix, old_cost, ordering, i, j):
    if i == j:
        return 0
        
    x = ordering[i]  # Элемент для перемещения
    
    if j < i:  # Перемещение влево
        indices = range(j, i)
        sign = 1
    else:  # j > i (перемещение вправо)
        indices = range(i + 1, j + 1)
        sign = -1
    
    total = 0
    for k in indices:
        y = ordering[k]
        total += matrix[x, y] - matrix[y, x]
    
    return sign * total + old_cost