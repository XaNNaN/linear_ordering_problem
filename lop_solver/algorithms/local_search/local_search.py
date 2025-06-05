# import numpy as np
# import random
# import time
# from algorithms.abs_solvers import HeuristicSolver
# from typing import List, Optional, Tuple, Dict, Callable

# import os
# import sys


# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

# class LocalSearchAlgorithm(HeuristicSolver):
#     """
#     Реализация алгоритма локального поиска для задачи линейного упорядочивания (LOP)
    
#     Параметры:
#     -----------
#     matrix : np.ndarray
#         Матрица весов
#     initial_solution : List[int], optional
#         Начальное решение (если не задано, генерируется случайно)
#     max_iter : int, optional
#         Максимальное количество итераций (по умолчанию 1000)
#     neighborhood_type : str, optional
#         Тип окрестности ('swap', 'insert', 'reverse') (по умолчанию 'swap')
#     improvement_strategy : str, optional
#         Стратегия улучшения ('best', 'first', 'random') (по умолчанию 'random')
#     max_neighbors : int, optional
#         Максимальное количество соседей для оценки (по умолчанию None)
#     """
    
#     def __init__(self, matrix: np.ndarray, 
#                  initial_solution: Optional[List[int]] = None,
#                  max_iter: int = 1000,
#                  neighborhood_type: str = 'swap',
#                  improvement_strategy: str = 'random',
#                  max_neighbors: Optional[int] = None):
#         self.matrix = matrix
#         self.n = matrix.shape[0]
#         self.max_iter = max_iter
#         self.neighborhood_type = neighborhood_type
#         self.improvement_strategy = improvement_strategy
#         self.cost_history = []
#         self.execution_time = 0.0
#         self.iterations = 0
        
#         # Инициализация решения
#         if initial_solution:
#             self.current_solution = initial_solution.copy()
#         else:
#             self.current_solution = list(range(self.n))
#             random.shuffle(self.current_solution)
        
#         self.current_cost = self.calc_cost(self.current_solution)
#         self.best_solution = self.current_solution.copy()
#         self.best_cost = self.current_cost
        
#         # Настройка max_neighbors
#         if max_neighbors is None:
#             if improvement_strategy == 'best':
#                 # Для best improvement используем всю окрестность
#                 if neighborhood_type == 'swap':
#                     self.max_neighbors = self.n * (self.n - 1) // 2
#                 elif neighborhood_type == 'insert':
#                     self.max_neighbors = (self.n - 1) ** 2
#                 elif neighborhood_type == 'reverse':
#                     self.max_neighbors = self.n * (self.n - 1) // 2
#             else:
#                 self.max_neighbors = min(100, self.n * 10)  # Ограничение для first/random
#         else:
#             self.max_neighbors = max_neighbors
        
#         # Словарь функций генерации соседей
#         self.neighbor_generators = {
#             'swap': self.generate_swap_neighbor,
#             'insert': self.generate_insert_neighbor,
#             'reverse': self.generate_reverse_neighbor
#         }
        
#         # Словарь функций расчета приращения
#         self.delta_calculators = {
#             'swap': self.calc_swap_delta,
#             'insert': self.calc_insert_delta,
#             'reverse': self.calc_reverse_delta
#         }

#     def calc_cost(self, solution: List[int]) -> int:
#         """Вычисление стоимости решения"""
#         ordered_matrix = self.matrix[np.ix_(solution, solution)] # Сортировка матрицы
#         return np.sum(np.triu(ordered_matrix, k=1)) # Фильтрация + суммиирование

#     def generate_swap_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
#         """Генерация соседа swap"""
#         n = len(solution)
#         i, j = random.sample(range(n), 2)
#         neighbor = solution.copy()
#         neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#         return neighbor, i, j

#     def generate_insert_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
#         """Генерация соседа insert"""
#         n = len(solution)

#         i, j = random.sample(range(n), 2)
#         neighbor = solution.copy()
#         elem = neighbor.pop(i)
#         if j > i:
#             j -= 1
#         neighbor.insert(j, elem)
#         return neighbor, i, j


#     def generate_reverse_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
#         """Генерация соседа reverse"""
#         n = len(solution)
#         i, j = sorted(random.sample(range(n), 2))
#         neighbor = solution.copy()
#         neighbor[i:j+1] = reversed(neighbor[i:j+1])
#         return neighbor, i, j


#     def calc_swap_delta(self, solution: List[int], i: int, j: int) -> int:
#         """Расчет приращения для swap"""
#         a, b = min(i, j), max(i, j)
#         delta = 0
#         # Элементы между i и j
#         for k in range(a + 1, b):
#             delta += self.matrix[solution[k], solution[a]] - self.matrix[solution[a], solution[k]]
#             delta += self.matrix[solution[b], solution[k]] - self.matrix[solution[k], solution[b]]
#         # Пара i-j
#         delta += self.matrix[solution[b], solution[a]] - self.matrix[solution[a], solution[b]]
#         # Обработка соседних элементов
#         if b - a == 1:
#             delta += self.matrix[solution[a], solution[b]] - self.matrix[solution[b], solution[a]]
#         return delta

#     def calc_insert_delta(self, solution: List[int], i: int, j: int) -> int:
#         """Расчет приращения для insert"""
#         elem = solution[i]
#         new_sol = solution.copy()
#         new_sol.pop(i)
#         new_sol.insert(j, elem)
#         return self.calc_cost(new_sol) - self.current_cost

#     def calc_reverse_delta(self, solution: List[int], i: int, j: int) -> int:
#         """Расчет приращения для reverse"""
#         delta = 0
#         # Вклад обращенного сегмента
#         for a in range(i, j + 1):
#             for b in range(a + 1, j + 1):
#                 delta += self.matrix[solution[b], solution[a]] - self.matrix[solution[a], solution[b]]
#         return delta

#     def solve(self) -> None:
#         """Основной метод выполнения алгоритма"""
#         start_time = time.time()
#         self.cost_history = [self.current_cost]
        
#         for _ in range(self.max_iter):
#             candidate_solution = None
#             candidate_cost = None
#             improve = 0
            
#             # Стратегия Best Improvement
#             if self.improvement_strategy == 'best':
#                 best_delta = -float('inf')
#                 best_neighbor = None
#                 neighbors_evaluated = 0
                
#                 # Генерация соседей
#                 for _ in range(self.max_neighbors):
#                     neighbor, i, j = self.neighbor_generators[self.neighborhood_type](self.current_solution)
#                     neighbor_cost = self._calc_cost(self.matrix, neighbor)
#                     neighbors_evaluated += 1
                    
#                     if neighbor_cost > self.current_cost:
#                         self.current_cost = neighbor_cost
#                         self.current_neighbor = neighbor
                
#                 if self.current_cost > self.best_cost:
#                     self.best_solution = self.current_neighbor
#                     self.best_cost = self.current_cost
#                     improve = 1
            
#             # Стратегии First/Random Improvement
#             else:
#                 for _ in range(self.max_neighbors):
#                     neighbor, i, j = self.neighbor_generators[self.neighborhood_type](self.current_solution)
#                     neighbor_cost = self._calc_cost(self.matrix, neighbor)
                    
#                     if neighbor_cost > self.current_cost:
#                         self.current_neighbor = neighbor
#                         self.current_cost = neighbor_cost
#                         improve = 1
#                         break
            
#             # Обновление решения
#             if improve == 0:
#                 self.cost = self.best_cost
#                 break  # Локальный оптимум достигнут
        
#         self.execution_time = time.time() - start_time

#     def get_stats(self) -> Dict:
#         """Статистика выполнения алгоритма"""
#         return {
#             'matrix_size': self.n,
#             'iterations': self.iterations,
#             'best_cost': self.best_cost,
#             'execution_time': self.execution_time,
#             'neighborhood_type': self.neighborhood_type,
#             'improvement_strategy': self.improvement_strategy,
#             'max_neighbors': self.max_neighbors,
#             'cost_history': self.cost_history,
#             'solution': self.best_solution
#         }
    


# if __name__ == "__main__":
#     # Генерация тестовой матрицы
#     matrix = np.random.randint(0, 100, size=(50, 50))

#     # Инициализация алгоритма
#     ls = LocalSearchAlgorithm(
#         matrix,
#         neighborhood_type='swap',
#         improvement_strategy='random',
#         max_iter=500
#     )

#     # Запуск алгоритма
#     ls.solve()

#     # Получение результатов
#     stats = ls.get_stats()
#     print(f"Best cost: {stats['best_cost']}")
#     print(f"Executed in {stats['execution_time']:.4f} seconds")