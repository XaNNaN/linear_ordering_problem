import numpy as np
import time

from typing import List, Optional
from algorithms.abs_solvers import HeuristicSolver

class BeckerAlgorithm(HeuristicSolver):
    """
    Реализация алгоритма Беккера для LOP с двумя вариантами:
    1. Оригинальный (полный пересчет q_i на каждом шаге) - O(n³)
    2. Оптимизированный (однократный расчет q_i) - O(n²)
    """
    
    def __init__(self, matrix: np.ndarray, optimized: bool = True):
        super().__init__(matrix)
        self.optimized = optimized  # Флаг для выбора версии алгоритма
        self.q_history = []  # История значений q_i для анализа

    def solve(self) -> None:
        start_time = time.time()
        
        if self.optimized:
            self.solution = self._optimized_becker()
        else:
            self.solution = self._original_becker()
            
        self.cost = self._calc_cost(self.matrix, self.solution)
        self.execution_time = time.time() - start_time

    def _original_becker(self) -> List[int]:
        """Оригинальный алгоритм Беккера O(n³)"""
        n = self.n
        remaining = list(range(n))
        ordering = []
        
        while remaining:
            q_values = []
            for i in remaining:
                # Вычисляем q_i = sum(c_ik) / sum(c_kt) для t в remaining
                sum_row = sum(self.matrix[i, k] for k in remaining)
                sum_col = sum(self.matrix[k, i] for k in remaining)
                q_values.append(sum_row / sum_col if sum_col != 0 else 0)
            
            max_idx = np.argmax(q_values)
            selected = remaining.pop(max_idx)
            ordering.append(selected)
            self.q_history.append(q_values.copy())
            
        return ordering

    def _optimized_becker(self) -> List[int]:
        """Оптимизированная версия O(n²)"""
        n = self.n
        q_values = np.zeros(n)
        
        # Однократный расчет q_i
        for i in range(n):
            sum_row = np.sum(self.matrix[i, :])
            sum_col = np.sum(self.matrix[:, i])
            q_values[i] = sum_row / sum_col if sum_col != 0 else 0
        
        self.q_history = q_values.copy()
        return list(np.argsort(q_values)[::-1])  # Сортировка по убыванию

    def get_stats(self) -> dict:
        stats = super().get_stats()
        opt = "Optimized" if self.optimized else "Original"
        stats.update({
            "solver": stats["solver"] + " " + opt,
            "method": "Becker's Algorithm",
            "optimized": self.optimized,
            "q_values": self.q_history
        })
        return stats