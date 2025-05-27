import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import time
import numpy as np
from algorithms.heuristic.constructive.becker import BeckerAlgorithm
from benchmarks.random_matrix import generate_random_lop_instance as generate_random_matrix

def test_runtime_comparison():
    """Сравнение времени выполнения на матрицах разного размера."""
    sizes = [10, 20, 50]
    results = []
    
    for n in sizes:
        matrix = generate_random_matrix(n)
        
        start = time.time()
        becker = BeckerAlgorithm(matrix, optimized=False)
        becker.solve()
        orig_time = time.time() - start
        
        start = time.time()
        becker = BeckerAlgorithm(matrix, optimized=True)
        becker.solve()
        opt_time = time.time() - start
        
        results.append((n, orig_time, opt_time))
    
    print("\nRuntime Comparison:")
    print("{:<10} {:<15} {:<15}".format("Size", "Original (s)", "Optimized (s)"))
    for n, t1, t2 in results:
        print("{:<10} {:<15.4f} {:<15.4f}".format(n, t1, t2))