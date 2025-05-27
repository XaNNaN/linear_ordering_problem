""" Этот метод 
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from itertools import permutations
from typing import List, Tuple

from utils.lop_utils import calculate_lop_cost_fast


def brute_force_lop(matrix: np.ndarray) -> Tuple[List[int], int]:
    """Полный перебор всех перестановок для поиска оптимального решения."""
    n = matrix.shape[0]
    best_ordering = None
    best_cost = -np.inf

    for perm in permutations(range(n)):
        current_cost = calculate_lop_cost_fast(matrix, perm)
        if current_cost > best_cost:
            best_cost = current_cost
            best_ordering = perm

    return list(best_ordering), best_cost