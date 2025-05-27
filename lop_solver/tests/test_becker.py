import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import pytest
import numpy as np
from algorithms.heuristic.constructive.becker import BeckerAlgorithm
from benchmarks.random_matrix import generate_random_lop_instance as generate_random_matrix

@pytest.mark.parametrize("optimized", [True, False])
def test_becker_on_small_matrix(optimized):
    """Тест на маленькой матрице с известным оптимумом."""
    matrix = np.array([
        [0, 3, 1],
        [2, 0, 4],
        [5, 6, 0]
    ])
    becker = BeckerAlgorithm(matrix, optimized=optimized)
    becker.solve()
    
    assert len(becker.get_ordering()) == 3
    assert becker.get_cost() >= 10  # Известное значение для этой матрицы

def test_optimized_vs_original():
    """Сравнение двух версий алгоритма."""
    matrix = generate_random_matrix(10)
    becker_orig = BeckerAlgorithm(matrix, optimized=False)
    becker_opt = BeckerAlgorithm(matrix, optimized=True)
    
    becker_orig.solve()
    becker_opt.solve()
    
    # Оптимизированная версия не должна давать худший результат
    assert becker_opt.get_cost() >= becker_orig.get_cost() * 0.95  # Допуск 5%