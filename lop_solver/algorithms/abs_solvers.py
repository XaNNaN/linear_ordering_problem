import numpy as np
import time

from abc import ABC, abstractmethod
from typing import List, Tuple
from utils.lop_utils import calculate_lop_cost_fast


class BaseSolver(ABC):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.solution = None
        self.cost = -np.inf
        self.execution_time = 0
        self.iterations = 0

    @abstractmethod
    def solve(self) -> None:
        """Основной метод решения (должен быть реализован в подклассах)"""
        pass

    def get_ordering(self) -> List[int]:
        """Возвращает найденное решение"""
        return self.solution
    
    def _calc_cost(self, matrix: np.ndarray, ordering: List[int]) -> int:
        """Метод для вычисления стоимости"""
        return calculate_lop_cost_fast(matrix, ordering)

    def get_cost(self) -> float:
        """Возвращает стоимость решения"""
        return self.cost

    def get_stats(self) -> dict:
        """Возвращает статистику работы"""
        return {
            "solver": self.__class__.__name__,
            "cost": self.cost,
            "time": self.execution_time,
            "iterations": self.iterations
        }
    

# class ExactSolver(BaseSolver):
#     def __init__(self, matrix: np.ndarray):
#         super().__init__(matrix)
#         self.nodes_explored = 0

#     def get_stats(self) -> dict:
#         stats = super().get_stats()
#         stats.update({"nodes_explored": self.nodes_explored})
#         return stats
    

class HeuristicSolver(BaseSolver):
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.improvement_history = []

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({"improvement": self.improvement_history})
        return stats