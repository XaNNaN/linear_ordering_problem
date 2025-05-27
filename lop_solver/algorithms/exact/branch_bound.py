import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linprog
import time

from algorithms.abs_solvers import ExactSolver

class BranchAndBound(ExactSolver):
    """
    Реализация метода ветвей и границ для LOP:
    1. Использует LP-релаксацию для получения верхних границ
    2. Ветвление по переменным с дробными значениями ~0.5
    3. Проверяет транзитивность и отсекает недопустимые решения
    """
    
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.node_history = []  # Для анализа работы алгоритма
        self.best_cost = - np.inf

    def solve(self, time_limit: float = 3600) -> None:
        start_time = time.time()
        
        # Инициализация
        root_node = {
            'constraints': [],  # Список фиксированных переменных
            'lp_solution': None,
            'upper_bound': np.inf
        }
        
        active_nodes = [root_node]
        
        while active_nodes and (time.time() - start_time) < time_limit:
            # Выбор узла с максимальной верхней границей
            current_node = max(active_nodes, key=lambda x: x['upper_bound'])
            active_nodes.remove(current_node)
            
            # Решение LP-релаксации
            lp_sol = self._solve_lp_relaxation(current_node['constraints'])
            
            # Проверка на целочисленность и транзитивность
            if self._is_integer_and_transitive(lp_sol):
                cost = self._calculate_exact_cost(lp_sol)
                if cost > self.best_cost:
                    self.best_cost = cost
                    self.solution = self._extract_ordering(lp_sol)
                continue
            
            # Отсечение по границе
            current_ub = self._calculate_lp_cost(lp_sol)
            if current_ub <= self.best_cost:
                continue
            
            # Ветвление
            branch_var = self._select_branching_variable(lp_sol)
            
            # Создание подзадач
            left_node = {
                'constraints': current_node['constraints'] + [(branch_var[0], branch_var[1], 1)],
                'upper_bound': current_ub
            }
            
            right_node = {
                'constraints': current_node['constraints'] + [(branch_var[1], branch_var[0], 1)],
                'upper_bound': current_ub
            }
            
            active_nodes.extend([left_node, right_node])
            self.nodes_explored += 1
            self.node_history.append({
                'node': current_node,
                'branch_var': branch_var,
                'lp_sol': lp_sol
            })
        
        self.execution_time = time.time() - start_time

    def _solve_lp_relaxation(self, constraints: List[Tuple[int, int, int]]) -> np.ndarray:
        """Решает LP-релаксацию с учетом фиксированных переменных"""
        n = self.n
        c = self.matrix.flatten()  # Целевая функция
        
        # Базовые ограничения x_ij + x_ji = 1
        A_eq, b_eq = self._build_basic_constraints()
        
        # Добавляем фиксированные переменные
        for i, j, val in constraints:
            constr = np.zeros((n, n))
            constr[i, j] = 1
            A_eq = np.vstack([A_eq, constr.flatten()])
            b_eq = np.append(b_eq, val)
        
        # Решаем LP
        res = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
        return res.x.reshape((n, n))

    def _build_basic_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Строит матрицу ограничений x_ij + x_ji = 1"""
        n = self.n
        A_eq = []
        b_eq = []
        
        for i in range(n):
            for j in range(i+1, n):
                constr = np.zeros((n, n))
                constr[i, j] = 1
                constr[j, i] = 1
                A_eq.append(constr.flatten())
                b_eq.append(1)
                
        return np.array(A_eq), np.array(b_eq)

    def _is_integer_and_transitive(self, solution: np.ndarray) -> bool:
        """Проверяет целочисленность и транзитивность"""
        if not np.all(np.isclose(solution, 0) | np.isclose(solution, 1)):
            return False
            
        # Проверка транзитивности
        n = solution.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i == j or j == k or i == k:
                        continue
                    if solution[i,j] + solution[j,k] - solution[i,k] > 1 + 1e-6:
                        return False
        return True

    def _select_branching_variable(self, solution: np.ndarray) -> Tuple[int, int]:
        """Выбирает переменную с дробным значением, ближайшим к 0.5"""
        fractional = np.abs(solution - 0.5)
        np.fill_diagonal(fractional, -1)  # Игнорируем диагональ
        return np.unravel_index(np.argmax(fractional), solution.shape)

    def _calculate_lp_cost(self, solution: np.ndarray) -> float:
        """Вычисляет стоимость LP-решения"""
        return np.sum(self.matrix * solution)

    def _extract_ordering(self, solution: np.ndarray) -> List[int]:
        """Извлекает перестановку из матрицы решений"""
        return list(np.argsort(np.sum(solution, axis=1))[::-1])

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            'method': 'Branch and Bound with LP Relaxation',
            'nodes_explored': self.nodes_explored,
            'node_history': self.node_history[:10]  # Первые 10 узлов для анализа
        })
        return stats