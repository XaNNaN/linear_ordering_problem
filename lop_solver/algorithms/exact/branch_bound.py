import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linprog
import time
import itertools
import heapq
from algorithms.abs_solvers import ExactSolver

class BranchAndBound(ExactSolver):
    """
    Исправленная реализация метода ветвей и границ без бесконечного цикла
    """
    
    class Node:
        __slots__ = ('constraints', 'upper_bound', 'depth')
        
        def __init__(self, constraints, upper_bound, depth):
            self.constraints = constraints  # [(i, j, value)]
            self.upper_bound = upper_bound
            self.depth = depth
            
        def __lt__(self, other):
            # Для max-heap: узлы с большей верхней границей имеют высший приоритет
            return self.upper_bound > other.upper_bound

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.n = matrix.shape[0]
        self.total_vars = self.n * self.n
        self.best_solution = None
        self.best_cost = -np.inf
        self.base_constraints = self._build_base_constraints()
        self.tol = 1e-5
        self.visited_nodes = set()  # Для избежания дубликатов
        self.node_queue = []  # Очередь с приоритетом

    def solve(self, time_limit: float = 3600) -> None:
        start_time = time.time()
        
        # Инициализация корневого узла
        root_node = self.Node(constraints=[], upper_bound=np.inf, depth=0)
        heapq.heappush(self.node_queue, root_node)
        self.visited_nodes.add(self._node_hash(root_node))
        
        while self.node_queue and (time.time() - start_time) < time_limit:
            current_node = heapq.heappop(self.node_queue)
            
            # Пропуск узлов, которые не могут улучшить решение
            if current_node.upper_bound <= self.best_cost + self.tol:
                continue
                
            # Решение LP-релаксации
            lp_sol = self._solve_lp_relaxation(current_node.constraints)
            
            # Пропуск узла если решение не найдено
            if lp_sol is None:
                continue
            
            # Расчет верхней границы
            current_ub = self._calculate_lp_cost(lp_sol)
            
            # Обновление верхней границы узла
            current_node.upper_bound = current_ub
            
            # Пропуск узлов, которые не могут улучшить решение (после точного расчета)
            if current_ub <= self.best_cost + self.tol:
                continue
            
            # Проверка на целочисленность и транзитивность
            if self._is_integer_and_transitive(lp_sol):
                cost = self._calculate_exact_cost(lp_sol)
                if cost > self.best_cost:
                    self.best_cost = cost
                    self.best_solution = self._extract_ordering(lp_sol)
                continue
            
            # Ветвление
            branch_var = self._select_branching_variable(lp_sol, current_node.constraints)
            if branch_var is None:
                continue
            
            # Создание подзадач
            for value in [1, 0]:
                new_constraints = current_node.constraints + [(branch_var[0], branch_var[1], value)]
                new_node = self.Node(
                    constraints=new_constraints,
                    upper_bound=current_ub,  # Наследуем текущую верхнюю границу
                    depth=current_node.depth + 1
                )
                node_hash = self._node_hash(new_node)
                
                # Добавляем только новые узлы
                if node_hash not in self.visited_nodes:
                    heapq.heappush(self.node_queue, new_node)
                    self.visited_nodes.add(node_hash)
                    self.nodes_explored += 1
        
        self.execution_time = time.time() - start_time
        if self.best_solution is not None:
            self.solution = self.best_solution
            self.cost = self.best_cost

    def _node_hash(self, node: Node) -> str:
        """Создает уникальный хеш для узла на основе его ограничений"""
        constraints_str = ",".join(f"{i},{j},{v}" for i, j, v in sorted(node.constraints))
        return f"{node.depth}|{constraints_str}"

    def _build_base_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Строит базовые ограничения: x_ij + x_ji = 1 и x_ii = 0"""
        n = self.n
        A_eq = []
        b_eq = []
        
        # 1. Ограничения x_ij + x_ji = 1 для i < j
        for i in range(n):
            for j in range(i+1, n):
                constr = np.zeros(self.total_vars)
                idx_ij = i * n + j
                idx_ji = j * n + i
                constr[idx_ij] = 1
                constr[idx_ji] = 1
                A_eq.append(constr)
                b_eq.append(1)
        
        # 2. Ограничения диагонали x_ii = 0
        for i in range(n):
            constr = np.zeros(self.total_vars)
            idx = i * n + i
            constr[idx] = 1
            A_eq.append(constr)
            b_eq.append(0)
            
        return np.array(A_eq), np.array(b_eq)

    def _solve_lp_relaxation(self, constraints: List[Tuple[int, int, int]]) -> Optional[np.ndarray]:
        """Решает LP-релаксацию с учетом фиксированных переменных"""
        n = self.n
        c = -self.matrix.flatten()  # Минимизируем -c^T x
        
        # Начинаем с базовых ограничений
        A_eq = self.base_constraints[0].copy()
        b_eq = self.base_constraints[1].copy()
        
        # Добавляем фиксированные переменные
        for (i, j, val) in constraints:
            constr = np.zeros(self.total_vars)
            idx = i * n + j
            constr[idx] = 1
            A_eq = np.vstack([A_eq, constr])
            b_eq = np.append(b_eq, val)
        
        # Решаем LP
        try:
            res = linprog(c, 
                          A_eq=A_eq, 
                          b_eq=b_eq, 
                          bounds=(0, 1), 
                          method='highs',
                          options={'presolve': True, 'time_limit': 5})
            
            if res.success:
                return res.x.reshape((n, n))
            else:
                return None
        except Exception as e:
            print(f"LP solving error: {str(e)}")
            return None

    def _is_integer_and_transitive(self, solution: np.ndarray) -> bool:
        """Проверяет целочисленность и транзитивность"""
        # Проверка целочисленности (игнорируем диагональ)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if not (np.isclose(solution[i, j], 0, atol=self.tol) or 
                        np.isclose(solution[i, j], 1, atol=self.tol)):
                    return False
        
        # Проверка транзитивности
        for i, j, k in itertools.product(range(self.n), repeat=3):
            if i == j or j == k or i == k:
                continue
            if solution[i, j] + solution[j, k] - solution[i, k] > 1 + self.tol:
                return False
        return True

    def _select_branching_variable(self, solution: np.ndarray, 
                                  constraints: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int]]:
        """
        Выбирает переменную с дробным значением, ближайшей к 0.5,
        игнорируя уже фиксированные переменные
        """
        min_dist = float('inf')
        candidate = None
        
        # Создаем множество уже фиксированных переменных
        fixed_vars = {(i, j) for i, j, _ in constraints}
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j or (i, j) in fixed_vars:
                    continue
                    
                dist = abs(solution[i, j] - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    candidate = (i, j)
        return candidate

    def _calculate_lp_cost(self, solution: np.ndarray) -> float:
        """Вычисляет стоимость LP-решения"""
        # Учитываем только верхний треугольник (i < j)
        cost = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                cost += self.matrix[i, j] * solution[i, j]
        return cost

    def _calculate_exact_cost(self, solution: np.ndarray) -> float:
        """Вычисляет точную стоимость для целочисленного решения"""
        ordering = self._extract_ordering(solution)
        return self._calc_cost(self.matrix, ordering)

    def _extract_ordering(self, solution: np.ndarray) -> List[int]:
        """Извлекает перестановку из матрицы решений"""
        # Вычисляем сумму "исходящих" связей для каждого элемента
        outgoing = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    outgoing[i] += solution[i, j]
        
        return list(np.argsort(outgoing)[::-1])

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            'method': 'Branch and Bound with LP Relaxation',
            'nodes_explored': self.nodes_explored,
            'nodes_remaining': len(self.node_queue)
        })
        return stats