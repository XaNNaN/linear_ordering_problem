import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Dict, Optional
import time
from collections import deque
import itertools
from algorithms.abs_solvers import ExactSolver
import heapq
from utils.lop_utils import calculate_lop_cost_fast

class BranchAndCut(ExactSolver):
    
    class Node:
        __slots__ = ('constraints_eq', 'constraints_ub', 'lp_solution', 'upper_bound', 'depth')
        
        def __init__(self, constraints_eq, constraints_ub, lp_solution, upper_bound, depth):
            self.constraints_eq = constraints_eq  # [(i, j, value)]
            self.constraints_ub = constraints_ub  # [(i, j, k)] для x_ij + x_jk - x_ik <= 1
            self.lp_solution = lp_solution
            self.upper_bound = upper_bound
            self.depth = depth
            
        def __lt__(self, other):
            return self.upper_bound > other.upper_bound  # Для max-heap

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.n = matrix.shape[0]
        self.total_vars = self.n * self.n
        self.tol = 1e-5
        self.cut_count = 0
        self.node_queue = []
        self.visited_nodes = set()
        # Кэш для хранения фиксированных переменных
        self.fixed_vars = np.full((self.n, self.n), -1)

    def solve(self, time_limit: float = 3600) -> None:
        start_time = time.time()
        self._initialize_root_node()
        
        while self.node_queue and time.time() - start_time < time_limit:
            node = heapq.heappop(self.node_queue)
            
            # Решение LP с текущими ограничениями
            x_mat, upper_bound = self._solve_lp(node)
            if x_mat is None:
                continue
                
            # Отсечение по границе
            if upper_bound <= self.cost + self.tol:
                continue
                
            # Проверка целочисленности и транзитивности
            if self._is_integer_solution(x_mat) and self._is_transitive(x_mat):
                cost = self._calculate_exact_cost(x_mat)
                if cost > self.cost:
                    self.cost = cost
                    self.solution = self._extract_ordering(x_mat)
                continue
                
            # Ветвление
            self._branch(node, x_mat)
            
        self.execution_time = time.time() - start_time

    def _initialize_root_node(self):
        n = self.n
        constraints_eq = []
        
        # Базовые ограничения (x_ij + x_ji = 1)
        for i in range(n):
            for j in range(i+1, n):
                constraints_eq.append((i, j, None))
                
        root_node = self.Node(
            constraints_eq=constraints_eq,
            constraints_ub=[],
            lp_solution=None,
            upper_bound=np.inf,
            depth=0
        )
        heapq.heappush(self.node_queue, root_node)
        self.visited_nodes.add(self._node_hash(root_node))

    def _solve_lp(self, node: Node) -> Tuple[Optional[np.ndarray], float]:
        n = self.n
        iter_count = 0
        max_iter = 5  # Максимум итераций добавления отсечений
        
        while iter_count < max_iter:
            # Формирование задачи LP
            c, A_eq, b_eq, A_ub, b_ub = self._build_lp_problem(node)
            
            # Решение LP
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=(0, 1), method='highs', options={'presolve': True})
            
            if not res.success:
                return None, -np.inf
                
            x_flat = res.x
            x_mat = x_flat.reshape((n, n))
            upper_bound = -res.fun
            
            # Поиск нарушенных транзитивных ограничений
            new_cuts = self._find_violated_triangle_cuts(x_mat, node.constraints_ub)
            
            if not new_cuts:
                return x_mat, upper_bound
                
            # Добавление новых отсечений
            node.constraints_ub.extend(new_cuts)
            self.cut_count += len(new_cuts)
            iter_count += 1
            
        return x_mat, upper_bound

    def _build_lp_problem(self, node: Node):
        n = self.n
        c = -self.matrix.flatten()  # Минимизируем -c^T x
        
        # Инициализация списков ограничений
        A_eq_list = []
        b_eq_list = []
        A_ub_list = []
        b_ub_list = []
        
        # Обработка ограничений типа равенств
        for i, j, value in node.constraints_eq:
            if value is not None:
                # Фиксированная переменная
                constr = np.zeros(self.total_vars)
                constr[i*n + j] = 1
                A_eq_list.append(constr)
                b_eq_list.append(value)
            else:
                # Парное ограничение
                constr = np.zeros(self.total_vars)
                constr[i*n + j] = 1
                constr[j*n + i] = 1
                A_eq_list.append(constr)
                b_eq_list.append(1.0)
                
        # Транзитивные отсечения
        for i, j, k in node.constraints_ub:
            constr = np.zeros(self.total_vars)
            constr[i*n + j] = 1
            constr[j*n + k] = 1
            constr[i*n + k] = -1
            A_ub_list.append(constr)
            b_ub_list.append(1.0)
            
        # Ограничения диагонали (x_ii = 0)
        for i in range(n):
            constr = np.zeros(self.total_vars)
            constr[i*n + i] = 1
            A_eq_list.append(constr)
            b_eq_list.append(0.0)
            
        # Преобразование в массивы правильной формы
        A_eq = np.array(A_eq_list) if A_eq_list else np.zeros((0, self.total_vars))
        b_eq = np.array(b_eq_list) if b_eq_list else np.zeros(0)
        A_ub = np.array(A_ub_list) if A_ub_list else np.zeros((0, self.total_vars))
        b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)
        
        return c, A_eq, b_eq, A_ub, b_ub

    def _find_violated_triangle_cuts(self, x_mat: np.ndarray, existing_cuts: list):
        n = self.n
        new_cuts = []
        
        for i, j, k in itertools.product(range(n), repeat=3):
            if i == j or j == k or i == k:
                continue
            if (i, j, k) in existing_cuts:
                continue
            if x_mat[i, j] + x_mat[j, k] - x_mat[i, k] > 1 + self.tol:
                new_cuts.append((i, j, k))
                
        return new_cuts[:10]  # Ограничиваем количество новых отсечений

    def _is_integer_solution(self, x_mat: np.ndarray) -> bool:
        return np.all(np.abs(x_mat - np.round(x_mat)) < self.tol)

    def _is_transitive(self, x_mat: np.ndarray) -> bool:
        n = self.n
        for i, j, k in itertools.product(range(n), repeat=3):
            if i == j or j == k or i == k:
                continue
            if x_mat[i, j] + x_mat[j, k] - x_mat[i, k] > 1 + self.tol:
                return False
        return True

    def _calculate_exact_cost(self, x_mat: np.ndarray) -> float:
        ordering = self._extract_ordering(x_mat)
        return self._calc_cost(self.matrix, ordering)

    def _extract_ordering(self, x_mat: np.ndarray) -> List[int]:
        n = x_mat.shape[0]
        # Используем сумму по строкам как меру "ранга"
        row_sums = np.sum(x_mat, axis=1)
        return list(np.argsort(row_sums)[::-1])

    def _branch(self, node: Node, x_mat: np.ndarray):
        i, j = self._select_branching_variable(x_mat)
        depth = node.depth + 1
        
        # Левая ветвь: x_ij = 1
        left_constraints = node.constraints_eq.copy()
        left_constraints.append((i, j, 1))
        left_node = self.Node(
            constraints_eq=left_constraints,
            constraints_ub=node.constraints_ub.copy(),
            lp_solution=None,
            upper_bound=node.upper_bound,
            depth=depth
        )
        self._add_node(left_node)
        
        # Правая ветвь: x_ij = 0
        right_constraints = node.constraints_eq.copy()
        right_constraints.append((i, j, 0))
        right_node = self.Node(
            constraints_eq=right_constraints,
            constraints_ub=node.constraints_ub.copy(),
            lp_solution=None,
            upper_bound=node.upper_bound,
            depth=depth
        )
        self._add_node(right_node)

    def _select_branching_variable(self, x_mat: np.ndarray) -> Tuple[int, int]:
        n = self.n
        min_dist = 0.5
        candidate = (0, 1)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = abs(x_mat[i, j] - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    candidate = (i, j)
                    
        return candidate

    def _add_node(self, node: Node):
        node_hash = self._node_hash(node)
        if node_hash not in self.visited_nodes:
            heapq.heappush(self.node_queue, node)
            self.visited_nodes.add(node_hash)
            self.nodes_explored += 1

    def _node_hash(self, node: Node) -> str:
        eq_hash = hash(tuple((i, j, v) for i, j, v in node.constraints_eq if v is not None))
        ub_hash = hash(tuple(node.constraints_ub))
        return f"{eq_hash}_{ub_hash}"

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            'method': 'Branch and Cut',
            'cuts_added': self.cut_count,
            'nodes_explored': self.nodes_explored,
            'nodes_remaining': len(self.node_queue)
        })
        return stats