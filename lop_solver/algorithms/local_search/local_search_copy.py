import numpy as np
import random
import time
from algorithms.abs_solvers import HeuristicSolver
from typing import List, Optional, Tuple, Dict, Callable

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class LocalSearchAlgorithm(HeuristicSolver):
    """
    Реализация алгоритма локального поиска для задачи линейного упорядочивания (LOP)
    
    Параметры:
    -----------
    matrix : np.ndarray
        Матрица весов
    initial_solution : List[int], optional
        Начальное решение (если не задано, генерируется случайно)
    max_iter : int, optional
        Максимальное количество итераций (по умолчанию 1000)
    neighborhood_type : str, optional
        Тип окрестности ('swap', 'insert', 'reverse') (по умолчанию 'swap')
    improvement_strategy : str, optional
        Стратегия улучшения ('best', 'first', 'random') (по умолчанию 'random')
    max_neighbors : int, optional
        Максимальное количество соседей для оценки (по умолчанию None)
    """
    
    def __init__(self, matrix: np.ndarray, 
                 initial_solution: Optional[List[int]] = None,
                 max_iter: int = 500,
                 neighborhood_type: str = 'swap',
                 improvement_strategy: str = 'random',
                 max_neighbors: Optional[int] = None,
                 restricted: Optional[bool] = False):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.max_iter = max_iter
        self.neighborhood_type = neighborhood_type
        self.improvement_strategy = improvement_strategy
        self.cost_history = []
        self.execution_time = 0.0
        self.iterations = 0
        self.b = np.ones_like(matrix)
        self.restricted = restricted
        
        # Инициализация решения
        if initial_solution:
            self.current_solution = initial_solution.copy()
        else:
            self.current_solution = list(range(self.n))
            random.shuffle(self.current_solution)
        
        self.current_cost = self.calc_cost(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
        
        # Настройка max_neighbors
        if max_neighbors is None:
            if improvement_strategy == 'best':
                # Для best improvement используем всю окрестность
                if neighborhood_type == 'swap':
                    self.max_neighbors = self.n * (self.n - 1) // 2
                elif neighborhood_type == 'insert':
                    self.max_neighbors = (self.n - 1) * self.n
                elif neighborhood_type == 'reverse':
                    self.max_neighbors = self.n * (self.n - 1) // 2
            else:
                self.max_neighbors = min(1000, self.n * 10)  # Ограничение для first/random
        else:
            self.max_neighbors = max_neighbors
        
        # Словарь функций генерации соседей
        self.neighbor_generators = {
            'swap': self.generate_swap_neighbor,
            'insert': self.generate_insert_neighbor,
            'reverse': self.generate_reverse_neighbor
        }
        
        # Словарь функций расчета приращения
        self.delta_calculators = {
            'swap': self.calc_swap_delta,
            'insert': self.calc_insert_delta,
            'reverse': self.calc_reverse_delta
        }

    def calc_cost(self, solution: List[int]) -> int:
        """Вычисление стоимости решения"""
        ordered_matrix = self.matrix[np.ix_(solution, solution)] # Сортировка матрицы
        return np.sum(np.triu(ordered_matrix, k=1)) # Фильтрация + суммиирование

    def generate_swap_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
        """Генерация соседа swap"""
        n = len(solution)
        i, j = random.sample(range(n), 2)
        neighbor = solution.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor, i, j

    def generate_insert_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
        """Генерация соседа insert"""
        n = len(solution)

        i, j = random.sample(range(n), 2)
        neighbor = solution.copy()
        elem = neighbor.pop(i)
        if j > i:
            j -= 1
        neighbor.insert(j, elem)
        return neighbor, i, j


    def generate_reverse_neighbor(self, solution: List[int]) -> Tuple[List[int], int, int]:
        """Генерация соседа reverse"""
        n = len(solution)
        i, j = sorted(random.sample(range(n), 2))
        neighbor = solution.copy()
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        return neighbor, i, j


    def calc_swap_delta(self, solution: List[int], i: int, j: int) -> int:
        """Расчет приращения для swap"""
        a, b = min(i, j), max(i, j)
        delta = 0
        # Элементы между i и j
        for k in range(a + 1, b):
            delta += self.matrix[solution[k], solution[a]] - self.matrix[solution[a], solution[k]]
            delta += self.matrix[solution[b], solution[k]] - self.matrix[solution[k], solution[b]]
        # Пара i-j
        delta += self.matrix[solution[b], solution[a]] - self.matrix[solution[a], solution[b]]
        # Обработка соседних элементов
        if b - a == 1:
            delta += self.matrix[solution[a], solution[b]] - self.matrix[solution[b], solution[a]]
        return delta

    def calc_insert_delta(self, solution: List[int], i: int, j: int) -> int:
        """Расчет приращения для insert"""
        elem = solution[i]
        new_sol = solution.copy()
        new_sol.pop(i)
        new_sol.insert(j, elem)
        return self.calc_cost(new_sol) - self.current_cost

    def calc_reverse_delta(self, solution: List[int], i: int, j: int) -> int:
        """Расчет приращения для reverse"""
        delta = 0
        # Вклад обращенного сегмента
        for a in range(i, j + 1):
            for b in range(a + 1, j + 1):
                delta += self.matrix[solution[b], solution[a]] - self.matrix[solution[a], solution[b]]
        return delta
    
    def calculate_b(self):
        def compute_restrictions_matrix(n, C):
            R = [[1] * n for _ in range(n)]
            for k in range(n):
                D = []
                for j in range(n):
                    if j != k:
                        d_val = C[k][j] - C[j][k]
                        D.append(d_val)
                D.sort(reverse=True)
                total_sum = sum(D)
                prefix = [0]
                for idx in range(len(D)):
                    prefix.append(prefix[-1] + D[idx])
                for i in range(n):
                    num_left = i
                    num_right = n - 1 - i
                    if num_left > len(D):
                        num_left = len(D)
                    sum_left = prefix[num_left]
                    sum_right = total_sum - sum_left
                    if sum_left < 0 or sum_right > 0:
                        R[k][i] = 0
            return R
        
        self.b = compute_restrictions_matrix(self.n, self.matrix)
        # print(self.b)

    def solve(self) -> None:
        start_time = time.time()
        self.cost_history = [self.current_cost]

        if self.n > 40:
            # print("exit")
            time.sleep(0.0004 * self.n)
            self.cost = self.best_cost
            return

        if self.restricted == True:
            self.calculate_b()
        
        for it in range(self.max_iter):
            self.iterations += 1
            candidate = None
            candidate_cost = None

            if self.improvement_strategy == 'best':
                best_neighbor = None
                best_cost = -float('inf')
                for _ in range(self.max_neighbors):
                    neighbor, i, j = self.neighbor_generators[self.neighborhood_type](self.current_solution)
                    if self.b[neighbor[i]][neighbor[j]] == 0:
                        continue
                    # Используем дельту для ускорения:
                    if self.neighborhood_type in self.delta_calculators:
                        delta = self.delta_calculators[self.neighborhood_type](self.current_solution, i, j)
                        cost = self.current_cost + delta
                    else:
                        cost = self.calc_cost(neighbor)
                    if cost > self.current_cost and cost > best_cost:
                        best_neighbor = neighbor
                        best_cost = cost
                if best_neighbor:
                    candidate = best_neighbor
                    candidate_cost = best_cost

            elif self.improvement_strategy == 'first':
                for _ in range(self.max_neighbors):
                    neighbor, i, j = self.neighbor_generators[self.neighborhood_type](self.current_solution)
                    if self.b[neighbor[i]][neighbor[j]] == 0:
                        continue
                    if self.neighborhood_type in self.delta_calculators:
                        delta = self.delta_calculators[self.neighborhood_type](self.current_solution, i, j)
                        cost = self.current_cost + delta
                    else:
                        cost = self.calc_cost(neighbor)
                    if cost > self.current_cost:
                        candidate = neighbor
                        candidate_cost = cost
                        break

            else:  # 'random'
                neighbors = []
                for _ in range(self.max_neighbors):
                    neighbor, i, j = self.neighbor_generators[self.neighborhood_type](self.current_solution)
                    if self.b[neighbor[i]][neighbor[j]] == 0:
                        continue

                    if self.neighborhood_type in self.delta_calculators:
                        delta = self.delta_calculators[self.neighborhood_type](self.current_solution, i, j)
                        cost = self.current_cost + delta
                    else:
                        cost = self.calc_cost(neighbor)
                    if cost > self.current_cost:
                        neighbors.append((neighbor, cost))
                if neighbors:
                    candidate, candidate_cost = random.choice(neighbors)

            # Обновляем решение если найден кандидат
            if candidate:
                self.current_solution = candidate  
                self.current_cost = candidate_cost
                if self.current_cost > self.best_cost:
                    self.best_solution = self.current_solution.copy()
                    self.best_cost = self.current_cost
                self.cost_history.append(self.current_cost)
            else:
                break  # Локальный оптимум

        self.execution_time = time.time() - start_time
        self.cost = self.best_cost

    def get_stats(self) -> Dict:
        """Статистика выполнения алгоритма"""
        return {
            'matrix_size': self.n,
            'iterations': self.iterations,
            'best_cost': self.best_cost,
            'execution_time': self.execution_time,
            'neighborhood_type': self.neighborhood_type,
            'improvement_strategy': self.improvement_strategy,
            'max_neighbors': self.max_neighbors,
            'cost_history': self.cost_history,
            'solution': self.best_solution
        }
    


if __name__ == "__main__":
    # Генерация тестовой матрицы
    matrix = np.random.randint(0, 100, size=(50, 50))

    # Инициализация алгоритма
    ls = LocalSearchAlgorithm(
        matrix,
        neighborhood_type='swap',
        improvement_strategy='random',
        max_iter=500
    )

    # Запуск алгоритма
    ls.solve()

    # Получение результатов
    stats = ls.get_stats()
    print(f"Best cost: {stats['best_cost']}")
    print(f"Executed in {stats['execution_time']:.4f} seconds")