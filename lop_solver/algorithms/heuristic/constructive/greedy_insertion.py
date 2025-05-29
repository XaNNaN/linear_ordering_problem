import numpy as np
import random
import time
from typing import List, Optional, Callable, Tuple
from algorithms.abs_solvers import HeuristicSolver


class GreedySolver(HeuristicSolver):
    """
    Реализация жадного алгоритма и его модификаций для задачи LOP.
    
    Параметры:
    -----------
    matrix : np.ndarray
        Матрица весов
    method : str, optional
        Метод решения ('basic', 'best_insertion', 'look_ahead', 'random', 'weighted', 'reverse')
    look_ahead_steps : int, optional
        Глубина предпросмотра для метода look_ahead (по умолчанию 2)
    k : int, optional
        Количество кандидатов для случайного выбора (по умолчанию 3)
    weights : np.ndarray, optional
        Пользовательские веса элементов (если None, вычисляются автоматически)
    reverse : bool, optional
        Использовать обратное построение порядка (по умолчанию False)
    """
    
    def __init__(self, matrix: np.ndarray, 
                 method: str = 'basic',
                 look_ahead_steps: int = 2,
                 k: int = 3,
                 weights: Optional[np.ndarray] = None,
                 reverse: bool = False):
        super().__init__(matrix)
        self.method = method
        self.look_ahead_steps = look_ahead_steps
        self.k = k
        self.weights = weights
        self.reverse = reverse
        self.insertion_history = []
        self.reverse_status = "Reversed" if self.reverse else "Straight"
        
        # Автоматический расчет весов при необходимости
        if method == 'weighted' and weights is None:
            self.weights = self._calculate_default_weights()
    
    def solve(self) -> None:
        start_time = time.time()
        
        if self.reverse:
            # Обратное построение: создаем временный обратный порядок
            rev_order = []
            while len(rev_order) < self.n:
                self._add_next_element_reverse(rev_order)
            self.solution = rev_order[::-1]  # Инвертируем порядок
        else:
            # Прямое построение
            self.solution = []
            while len(self.solution) < self.n:
                self._add_next_element(self.solution)
        
        self.cost = self._calculate_exact_cost(self.solution)
        self.execution_time = time.time() - start_time
    
    def _add_next_element(self, current_order: List[int]) -> None:
        """Добавляет следующий элемент в текущий порядок"""
        candidates = [i for i in range(self.n) if i not in current_order]
        
        if self.method == 'basic':
            best_candidate = self._basic_greedy(current_order, candidates)
            current_order.append(best_candidate)
            
        elif self.method == 'best_insertion':
            best_candidate, best_pos = self._best_insertion(current_order, candidates)
            current_order.insert(best_pos, best_candidate)
            self.insertion_history.append((best_candidate, best_pos))
            
        elif self.method == 'look_ahead':
            best_candidate = self._look_ahead_greedy(current_order, candidates)
            current_order.append(best_candidate)
            
        elif self.method == 'random':
            best_candidate = self._random_greedy(current_order, candidates)
            current_order.append(best_candidate)
            
        elif self.method == 'weighted':
            best_candidate = self._weighted_greedy(current_order, candidates)
            current_order.append(best_candidate)
    
    def _add_next_element_reverse(self, rev_order: List[int]) -> None:
        """Добавляет следующий элемент в обратный порядок (для reverse=True)"""
        candidates = [i for i in range(self.n) if i not in rev_order]
        
        if self.method == 'basic':
            best_candidate = self._basic_greedy_reverse(rev_order, candidates)
            rev_order.append(best_candidate)
            
        elif self.method == 'best_insertion':
            best_candidate, best_pos = self._best_insertion_reverse(rev_order, candidates)
            rev_order.insert(best_pos, best_candidate)
            self.insertion_history.append((best_candidate, best_pos))
            
        elif self.method == 'look_ahead':
            best_candidate = self._look_ahead_greedy_reverse(rev_order, candidates)
            rev_order.append(best_candidate)
            
        elif self.method == 'random':
            best_candidate = self._random_greedy_reverse(rev_order, candidates)
            rev_order.append(best_candidate)
            
        elif self.method == 'weighted':
            best_candidate = self._weighted_greedy_reverse(rev_order, candidates)
            rev_order.append(best_candidate)
    
    # Базовый жадный алгоритм (прямой)
    def _basic_greedy(self, current_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_delta = -np.inf
        
        for candidate in candidates:
            # Вклад = сумма весов от всех элементов в порядке к кандидату
            delta = sum(self.matrix[j, candidate] for j in current_order)
            
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
                
        return best_candidate
    
    # Базовый жадный алгоритм (обратный)
    def _basic_greedy_reverse(self, rev_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_delta = -np.inf
        
        for candidate in candidates:
            # Вклад = сумма весов от кандидата ко всем элементам в обратном порядке
            delta = sum(self.matrix[candidate, j] for j in rev_order)
            
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
                
        return best_candidate
    
    # Лучшая вставка (прямая)
    def _best_insertion(self, current_order: List[int], candidates: List[int]) -> Tuple[int, int]:
        best_candidate = -1
        best_pos = -1
        best_delta = -np.inf
        
        for candidate in candidates:
            for pos in range(len(current_order) + 1):
                # Вычисляем вклад при вставке на позицию pos
                left = current_order[:pos]    # Элементы перед кандидатом
                right = current_order[pos:]   # Элементы после кандидата
                
                delta = (sum(self.matrix[j, candidate] for j in left) +
                         sum(self.matrix[candidate, j] for j in right))
                
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = candidate
                    best_pos = pos
                    
        return best_candidate, best_pos
    
    # Лучшая вставка (обратная)
    def _best_insertion_reverse(self, rev_order: List[int], candidates: List[int]) -> Tuple[int, int]:
        best_candidate = -1
        best_pos = -1
        best_delta = -np.inf
        
        for candidate in candidates:
            for pos in range(len(rev_order) + 1):
                # Вычисляем вклад при вставке на позицию pos
                top = rev_order[:pos]    # Элементы, которые будут после кандидата в итоговом порядке
                bottom = rev_order[pos:] # Элементы, которые будут перед кандидатом в итоговом порядке
                
                delta = (sum(self.matrix[candidate, j] for j in top) +
                         sum(self.matrix[j, candidate] for j in bottom))
                
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = candidate
                    best_pos = pos
                    
        return best_candidate, best_pos
    
    # Жадный с предпросмотром (прямой)
    def _look_ahead_greedy(self, current_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_total = -np.inf
        
        for candidate in candidates:
            # Симулируем добавление кандидата
            new_order = current_order + [candidate]
            total = self._simulate_look_ahead(new_order, self.look_ahead_steps - 1)
            
            if total > best_total:
                best_total = total
                best_candidate = candidate
                
        return best_candidate
    
    # Жадный с предпросмотром (обратный)
    def _look_ahead_greedy_reverse(self, rev_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_total = -np.inf
        
        for candidate in candidates:
            # Симулируем добавление кандидата
            new_rev_order = rev_order + [candidate]
            total = self._simulate_look_ahead_reverse(new_rev_order, self.look_ahead_steps - 1)
            
            if total > best_total:
                best_total = total
                best_candidate = candidate
                
        return best_candidate
    
    def _simulate_look_ahead(self, current_order: List[int], depth: int) -> float:
        """Рекурсивное моделирование предпросмотра (прямой)"""
        if depth == 0 or len(current_order) == self.n:
            return self._estimate_cost(current_order)
        
        best_total = -np.inf
        candidates = [i for i in range(self.n) if i not in current_order]
        
        for candidate in candidates:
            new_order = current_order + [candidate]
            total = self._simulate_look_ahead(new_order, depth - 1)
            best_total = max(best_total, total)
            
        return best_total
    
    def _simulate_look_ahead_reverse(self, rev_order: List[int], depth: int) -> float:
        """Рекурсивное моделирование предпросмотра (обратный)"""
        if depth == 0 or len(rev_order) == self.n:
            return self._estimate_cost_reverse(rev_order)
        
        best_total = -np.inf
        candidates = [i for i in range(self.n) if i not in rev_order]
        
        for candidate in candidates:
            new_rev_order = rev_order + [candidate]
            total = self._simulate_look_ahead_reverse(new_rev_order, depth - 1)
            best_total = max(best_total, total)
            
        return best_total
    
    # Случайный выбор из топ-k (прямой)
    def _random_greedy(self, current_order: List[int], candidates: List[int]) -> int:
        deltas = []
        for candidate in candidates:
            delta = sum(self.matrix[j, candidate] for j in current_order)
            deltas.append(delta)
        
        # Выбираем топ-k кандидатов
        indices = np.argsort(deltas)[-self.k:]
        top_candidates = [candidates[i] for i in indices]
        return random.choice(top_candidates)
    
    # Случайный выбор из топ-k (обратный)
    def _random_greedy_reverse(self, rev_order: List[int], candidates: List[int]) -> int:
        deltas = []
        for candidate in candidates:
            delta = sum(self.matrix[candidate, j] for j in rev_order)
            deltas.append(delta)
        
        # Выбираем топ-k кандидатов
        indices = np.argsort(deltas)[-self.k:]
        top_candidates = [candidates[i] for i in indices]
        return random.choice(top_candidates)
    
    # Взвешенный жадный (прямой)
    def _weighted_greedy(self, current_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_weighted_delta = -np.inf
        
        for candidate in candidates:
            delta = sum(self.matrix[j, candidate] for j in current_order)
            weighted_delta = delta * self.weights[candidate]
            
            if weighted_delta > best_weighted_delta:
                best_weighted_delta = weighted_delta
                best_candidate = candidate
                
        return best_candidate
    
    # Взвешенный жадный (обратный)
    def _weighted_greedy_reverse(self, rev_order: List[int], candidates: List[int]) -> int:
        best_candidate = -1
        best_weighted_delta = -np.inf
        
        for candidate in candidates:
            delta = sum(self.matrix[candidate, j] for j in rev_order)
            weighted_delta = delta * self.weights[candidate]
            
            if weighted_delta > best_weighted_delta:
                best_weighted_delta = weighted_delta
                best_candidate = candidate
                
        return best_candidate
    
    def _calculate_default_weights(self) -> np.ndarray:
        """Вычисляет веса по умолчанию как сумму исходящих дуг"""
        return np.sum(self.matrix, axis=1)
    
    def _estimate_cost(self, partial_order: List[int]) -> float:
        """Оценка стоимости частичного решения (оптимизированная)"""
        # Для экономии времени используем только добавленные элементы
        n_partial = len(partial_order)
        if n_partial == 0:
            return 0
        
        # Создаем подматрицу для добавленных элементов
        submatrix = self.matrix[np.ix_(partial_order, partial_order)]
        return np.sum(np.triu(submatrix, k=1))
    
    def _estimate_cost_reverse(self, rev_order: List[int]) -> float:
        """Оценка стоимости для обратного частичного решения"""
        # Для обратного порядка: сначала преобразуем в прямой
        order = rev_order[::-1]
        return self._estimate_cost(order)
    
    def _calculate_exact_cost(self, ordering: List[int]) -> float:
        """Точный расчет стоимости для полного порядка"""
        ordered_matrix = self.matrix[np.ix_(ordering, ordering)]
        return np.sum(np.triu(ordered_matrix, k=1))
    
    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            "solver": stats["solver"] + " " + self.method + " " + self.reverse_status,
            'method': f'Greedy ({self.method})',
            'reverse': self.reverse,
            'look_ahead_steps': self.look_ahead_steps if self.method == 'look_ahead' else None,
            'k': self.k if self.method == 'random' else None,
            'insertions': self.insertion_history
        })
        return stats