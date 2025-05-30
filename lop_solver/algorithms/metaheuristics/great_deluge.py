import numpy as np
import random
import time
from typing import List, Tuple, Optional
from algorithms.abs_solvers import HeuristicSolver
from utils.lop_utils import generate_neighborhood

class GreatDelugeAlgorithm(HeuristicSolver):
    """
    Реализация алгоритма "Великого потопа" для задачи линейного упорядочивания (LOP)
    
    Параметры:
    -----------
    matrix : np.ndarray
        Матрица весов
    initial_solution : List[int], optional
        Начальное решение (если не задано, генерируется случайно)
    max_iter : int, optional
        Максимальное количество итераций (по умолчанию 10000)
    rain_speed : float, optional
        Скорость "дождя" (снижения уровня воды) (по умолчанию 0.999)
    initial_water_level_factor : float, optional
        Фактор начального уровня воды относительно начального решения (по умолчанию 1.05)
    neighborhood_type : str, optional
        Тип окрестности ('swap', 'insert', 'reverse') (по умолчанию 'swap')
    """
    
    def __init__(self, matrix: np.ndarray, 
                 initial_solution: Optional[List[int]] = None,
                 max_iter: int = 10000,
                 rain_speed: float = 0.999,
                 initial_water_level_factor: float = 1.05,
                 neighborhood_type: str = 'swap'):
        super().__init__(matrix)
        self.max_iter = max_iter
        self.rain_speed = rain_speed
        self.initial_water_level_factor = initial_water_level_factor
        self.neighborhood_type = neighborhood_type
        self.water_level_history = []
        self.cost_history = []
        
        # Инициализация начального решения
        if initial_solution:
            self.current_solution = initial_solution.copy()
        else:
            self.current_solution = list(range(self.n))
            random.shuffle(self.current_solution)
            
        self.current_cost = self._calc_cost(matrix, self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
        
        # Установка начального уровня воды
        self.water_level = self.current_cost * initial_water_level_factor

    def solve(self) -> None:
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # Генерация соседних решений
            neighborhood = generate_neighborhood(
                self.current_solution, 
                neighborhood_type=self.neighborhood_type
            )
            
            # Выбор случайного соседа
            candidate_solution = random.choice(neighborhood)
            candidate_cost = self._calc_cost(self.matrix, candidate_solution)
            
            # Принятие решения о переходе
            if candidate_cost > self.current_cost:
                # Переход к лучшему решению
                self.current_solution = candidate_solution
                self.current_cost = candidate_cost
                
                # Обновление лучшего решения
                if candidate_cost > self.best_cost:
                    self.best_solution = candidate_solution
                    self.best_cost = candidate_cost
            elif candidate_cost > self.water_level:
                # Принятие решения с ухудшением, но выше уровня воды
                self.current_solution = candidate_solution
                self.current_cost = candidate_cost
            
            # Понижение уровня воды
            # if self.current_cost <= self.water_level:
            #     dry = self.rain_speed * (self.water_level - self.current_cost) 
            #     self.water_level -= dry
            # else:
            self.water_level *= self.rain_speed # базовый
            
            # Сохранение истории для анализа
            self.water_level_history.append(self.water_level)
            self.cost_history.append(self.current_cost)
            self.iterations += 1
            
            # Ранняя остановка, если уровень воды упал ниже лучшего решения
            if self.water_level < self.best_cost * 0.95:
                break
        
        # Фиксация лучшего решения
        self.solution = self.best_solution
        self.cost = self.best_cost
        self.execution_time = time.time() - start_time

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            'method': 'Great Deluge Algorithm',
            'rain_speed': self.rain_speed,
            'initial_water_level_factor': self.initial_water_level_factor,
            'neighborhood_type': self.neighborhood_type,
            'final_water_level': self.water_level,
            'water_level_history': self.water_level_history,
            'cost_history': self.cost_history
        })
        return stats
    


import matplotlib.pyplot as plt

def plot_gda_performance(gda: GreatDelugeAlgorithm):
    plt.figure(figsize=(12, 6))
    
    # График стоимости
    plt.subplot(1, 2, 1)
    plt.plot(gda.cost_history, 'b-', label='Текущая стоимость')
    plt.plot([gda.best_cost] * len(gda.cost_history), 'r--', label='Лучшая стоимость')
    plt.xlabel('Итерация')
    plt.ylabel('Стоимость')
    plt.title('Динамика стоимости')
    plt.legend()
    plt.grid(True)
    
    # График уровня воды
    plt.subplot(1, 2, 2)
    plt.plot(gda.water_level_history, 'g-')
    plt.xlabel('Итерация')
    plt.ylabel('Уровень воды')
    plt.title('Динамика уровня воды')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
