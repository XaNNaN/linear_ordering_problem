import numpy as np
import sys
import os

import algorithms.exact
import algorithms.exact.branch_cut

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from algorithms.heuristic.constructive.becker import BeckerAlgorithm
from algorithms.exact.branch_bound import BranchAndBound
from algorithms.exact.branch_cut import BranchAndCut
from algorithms.heuristic.constructive.greedy_insertion import GreedySolver 
from algorithms.metaheuristics.great_deluge import GreatDelugeAlgorithm, plot_gda_performance
from metrics.evaluation import SolverEvaluator
from benchmarks.lolib import load_lolib_matrix
from benchmarks.random_matrix import generate_random_lop_instance

def main():
    evaluator = SolverEvaluator([
        lambda m: BeckerAlgorithm(m, optimized=False),
        lambda m: BeckerAlgorithm(m, optimized=True),
        lambda m: GreedySolver(m, method='basic'),
        lambda m: GreedySolver(m, method='best_insertion'),
        lambda m: GreedySolver(m, method='look_ahead'),
        lambda m: GreedySolver(m, method='random'),
        lambda m: GreatDelugeAlgorithm(m,max_iter=5000, rain_speed=0.997, initial_water_level_factor=1.9) 
        # BranchAndBound,
        # BranchAndCut,
        # GreedyInsertion,
        # LocalSearch,
    ])
    

    # Тест на маленькой матрице (n=5)
    small_matrix = generate_random_lop_instance(5)
    evaluator.evaluate(small_matrix, repetitions=5)
    evaluator.print_report()

    evaluator.clear_results()
    # Тест на бенчмарке LOLIB (пример)
    try:
        project_root = os.path.dirname(SCRIPT_DIR)
        file_path = os.path.join(project_root, "lop_solver", "data", "lolib", "IO", "N-t59f11xx")
        lolib_matrix = load_lolib_matrix(file_path)

        evaluator.evaluate(lolib_matrix, repetitions=5)
        evaluator.print_report()
    except FileNotFoundError:
        print("LOLIB file not found. Skipping.")

    greedy = GreedySolver(lolib_matrix, method='best_insertion')
    greedy.solve()
    gda = GreatDelugeAlgorithm(lolib_matrix, initial_solution=greedy.get_ordering(), max_iter=5000, rain_speed=0.997, initial_water_level_factor=2)
    gda.solve()
    plot_gda_performance(gda)
    
if __name__ == "__main__":
    main()