import numpy as np

from typing import List
from algorithms.abs_solvers import BaseSolver

class SolverEvaluator:
    def __init__(self, solvers: List[BaseSolver]):
        self.solvers = solvers
        self.results = []

    def evaluate(self, matrix: np.ndarray, repetitions: int = 1) -> None:
        for solver_class in self.solvers:
            total_time = 0
            best_cost = -np.inf
            
            for _ in range(repetitions):
                solver = solver_class(matrix)
                solver.solve()
                total_time += solver.execution_time
                if solver.cost > best_cost:
                    best_cost = solver.cost
                    best_stats = solver.get_stats()
            
            self.results.append({
                "solver": best_stats["solver"],
                "avg_time": total_time / repetitions,
                "best_cost": best_cost,
                "details": best_stats
            })

    def get_results(self) -> List[dict]:
        return sorted(self.results, key=lambda x: x["best_cost"], reverse=True)

    def print_report(self):
        print("{:<45} {:<15} {:<15}".format("Solver", "Best Cost", "Avg Time"))
        for res in self.get_results():
            print("{:<45} {:<15.2f} {:<15.4f}".format(
                res["solver"], res["best_cost"], res["avg_time"]))

    def clear_results(self):
        self.results = []