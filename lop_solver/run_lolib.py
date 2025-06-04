import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


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

import concurrent.futures

import shutil

import logging
logging.basicConfig(filename='benchmark_lolib.log', level=logging.INFO)

import concurrent.futures
from functools import partial

# Явно определяем функции для создания решателей
def create_becker_non_opt(matrix):
    return BeckerAlgorithm(matrix, optimized=False)

def create_becker_opt(matrix):
    return BeckerAlgorithm(matrix, optimized=True)

def create_greedy_basic(matrix):
    return GreedySolver(matrix, method='basic')

def create_greedy_best_insertion(matrix):
    return GreedySolver(matrix, method='best_insertion')

def create_greedy_look_ahead(matrix):
    return GreedySolver(matrix, method='look_ahead')

def create_greedy_random(matrix):
    return GreedySolver(matrix, method='random')

def create_great_deluge(matrix):
    return GreatDelugeAlgorithm(
        matrix, 
        max_iter=5000, 
        rain_speed=0.997, 
        initial_water_level_factor=1.9, 
        neighborhood_type='insert'
    )


# Функция для запуска решателя с ограничением времени
def run_solver_with_timeout(solver_factory, matrix, timeout=3):
    start_time = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: solver_factory(matrix).solve())
            result = future.result(timeout=timeout)
            solver = future.result()
            exec_time = time.time() - start_time
            return solver.cost, exec_time, None
    except concurrent.futures.TimeoutError:
        return float('nan'), timeout, "Timeout"
    except Exception as e:
        return float('nan'), time.time() - start_time, str(e)


def main():
    # Определяем решатели
    solvers = {
        'Becker (non-opt)': create_becker_non_opt,
        'Becker (opt)': create_becker_opt,
        'Greedy (basic)': create_greedy_basic,
        'Greedy (best_insertion)': create_greedy_best_insertion,
        'Greedy (look_ahead)': create_greedy_look_ahead,
        'Greedy (random)': create_greedy_random,
        'GreatDeluge': create_great_deluge
    }

    # Определяем пути
    project_root = os.path.dirname(SCRIPT_DIR)
    lolib_data_root = os.path.join(project_root, "lop_solver", "data", "lolib")
    results_root = os.path.join(project_root, "results", "lolib")
    
    # Создаем папку для результатов
    os.makedirs(results_root, exist_ok=True)
    
    # Находим все наборы данных LOLIB
    datasets = [d for d in os.listdir(lolib_data_root) 
                if os.path.isdir(os.path.join(lolib_data_root, d))]
    
    if not datasets:
        print("No LOLIB datasets found in directory")
        return
    
    print(f"Found {len(datasets)} LOLIB datasets for parallel processing")
    
    # Создаем частичную функцию для обработки набора данных
    process_func = partial(
        process_dataset, 
        solvers=solvers, 
        lolib_data_root=lolib_data_root,
        results_root=results_root
    )
    
    # Параллельная обработка наборов данных
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for dataset in datasets:
            future = executor.submit(
                process_dataset,
                dataset,
                solvers,
                lolib_data_root,
                results_root
            )
            futures[future] = dataset
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(datasets), desc="Processing datasets"):
            dataset = futures[future]
            try:
                dataset_results = future.result()
                all_results.append(dataset_results)
                print(f"✅ Completed dataset: {dataset}")
            except Exception as e:
                print(f"❌ Error processing dataset {dataset}: {str(e)}")
    
    # Объединяем результаты всех наборов данных
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Генерируем сводный отчет
        generate_summary_report(combined_df, results_root)
        
        # Пример визуализации для одной матрицы
        example_matrix = load_lolib_matrix(
            os.path.join(lolib_data_root, datasets[0], os.listdir(os.path.join(lolib_data_root, datasets[0]))[0]
        ))
        gda = GreatDelugeAlgorithm(
            example_matrix, 
            max_iter=3000, 
            rain_speed=0.997, 
            initial_water_level_factor=1, 
            neighborhood_type='insert'
        )
        gda.solve()
        plot_gda_performance(gda, os.path.join(results_root, "example_gda_performance.png"))
    else:
        print("No results to process")

def process_dataset(dataset: str, solvers: dict, lolib_data_root: str, results_root: str) -> pd.DataFrame:
    """
    Обрабатывает один набор данных LOLIB (выполняется в отдельном процессе)
    """
    dataset_path = os.path.join(lolib_data_root, dataset)
    results_path = os.path.join(results_root, dataset)
    
    # Создаем папку для результатов
    os.makedirs(results_path, exist_ok=True)
    
    # Находим все файлы матриц в наборе данных
    matrix_files = [f for f in os.listdir(dataset_path) 
                   if  os.path.isfile(os.path.join(dataset_path, f))]
    
    if not matrix_files:
        print(f"⚠️ No matrix files found in dataset: {dataset}")
        return pd.DataFrame()
    
    # Результаты для этого набора данных
    dataset_results = []
    
    # Обрабатываем каждую матрицу
    for matrix_file in matrix_files:
        file_path = os.path.join(dataset_path, matrix_file)
        matrix_name = os.path.splitext(matrix_file)[0]
        
        try:
            # Загружаем матрицу
            matrix = load_lolib_matrix(file_path)
            n = matrix.shape[0]
            
            # Запускаем тестирование
            matrix_results = run_matrix_benchmark(solvers, matrix, matrix_name, n)
            
            # Сохраняем результаты для этой матрицы
            matrix_results.to_csv(os.path.join(results_path, f"{matrix_name}.csv"), index=False)
            
            # Добавляем к результатам набора данных
            dataset_results.append(matrix_results)

            
            # Сохраняем графики производительности для GreatDeluge
            try:
                gda_solver = create_great_deluge(matrix)
                start_time = time.time()
                gda_solver.solve()
                exec_time = time.time() - start_time
                if exec_time > 3:
                    print(f"GreatDeluge exceeded time limit for {matrix_name}: {exec_time:.2f}s")
                
                plot_path = os.path.join(results_path, f"{matrix_name}_gda_performance.png")
                plot_gda_performance(gda_solver, save_path=plot_path)
            except Exception as e:
                print(f"Error generating GDA plot for {matrix_name}: {str(e)}")
            
        except Exception as e:
            print(f"Error processing {matrix_file} in {dataset}: {str(e)}")
    
    # Объединяем результаты всех матриц в наборе
    dataset_df = pd.concat(dataset_results, ignore_index=True) if dataset_results else pd.DataFrame()

    # Генерируем финальный отчет для датасета
    if not dataset_df.empty:
        generate_dataset_report(dataset_df, results_path, dataset)

    return dataset_df


def run_matrix_benchmark(solvers: dict, matrix: np.ndarray, 
                         matrix_name: str, size: int, repetitions: int = 5) -> pd.DataFrame:
    """
    Запускает тестирование всех решателей на одной матрице
    """
    results = {
        'matrix': [],
        'size': [],
        'solver': [],
        'repetition': [],
        'cost': [],
        'time': []
    }
    
    # Лучшие решения для этой матрицы
    best_solutions = []
    
    # Запускаем решатели
    for solver_name, solver_factory in solvers.items():
        for rep in range(repetitions):
            try:
                solver = solver_factory(matrix)
                start_time = time.time()
                solver.solve()
                exec_time = time.time() - start_time
                
                # Сохраняем результаты
                results['matrix'].append(matrix_name)
                results['size'].append(size)
                results['solver'].append(solver_name)
                results['repetition'].append(rep)
                results['cost'].append(solver.cost)
                results['time'].append(exec_time)
                
                # Сохраняем лучшее решение
                best_solutions.append(solver.cost)
                
            except Exception as e:
                print(f"Error running {solver_name} on {matrix_name}: {str(e)}")
    
    # Создаем DataFrame
    df = pd.DataFrame(results)
    
    # Добавляем отклонение от лучшего решения
    if best_solutions:
        best_cost = max(best_solutions)
        df['deviation'] = (best_cost - df['cost']) / best_cost * 100.0
        df['best_cost'] = best_cost
    else:
        df['deviation'] = float('nan')
        df['best_cost'] = float('nan')
    
    return df

def generate_dataset_report(df: pd.DataFrame, output_path: str, dataset_name: str):
    """
    Генерирует отчет по одному датасету
    """
    if df.empty:
        return
    
    report_path = os.path.join(output_path, "reports")
    os.makedirs(report_path, exist_ok=True)
    
    # 1. Отчет по решателям
    solver_report = df.groupby('solver').agg(
        num_matrices=('matrix', 'nunique'),
        success_rate=('status', lambda x: (x == 'Success').mean()),
        timeout_rate=('status', lambda x: (x == 'Timeout').mean()),
        avg_deviation=('deviation', 'mean'),
        std_deviation=('deviation', 'std'),
        avg_time=('time', 'mean'),
        percentile_90_time=('time', lambda x: np.percentile(x, 90)),
        best_cost=('cost', 'max')
    ).reset_index()
    
    solver_report = solver_report.sort_values(by='avg_deviation')
    solver_report.to_csv(os.path.join(report_path, f"{dataset_name}_solver_performance.csv"), index=False)
    
    # 2. Отчет по матрицам
    matrix_report = df.groupby(['matrix', 'size']).agg(
        best_cost=('best_cost', 'first'),
        solver_count=('solver', 'nunique'),
        best_solver=('cost', lambda x: df.loc[x.idxmax(), 'solver']),
        best_solver_cost=('cost', 'max')
    ).reset_index()
    
    matrix_report['deviation'] = (matrix_report['best_cost'] - matrix_report['best_solver_cost']) / matrix_report['best_cost'] * 100.0
    matrix_report.to_csv(os.path.join(report_path, f"{dataset_name}_matrix_summary.csv"), index=False)
    
    # 3. Визуализация
    plot_dataset_results(df, report_path, dataset_name)

def plot_dataset_results(df: pd.DataFrame, output_path: str, dataset_name: str):
    """Визуализирует результаты для датасета"""
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        plt.scatter(
            solver_df['size'], 
            solver_df['deviation'],
            label=solver,
            alpha=0.5
        )
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Deviation from Best Solution (%)')
    plt.title(f'Solution Quality vs Matrix Size - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'{dataset_name}_quality_vs_size.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        plt.scatter(
            solver_df['size'], 
            solver_df['time'],
            label=solver,
            alpha=0.5
        )
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')
    plt.title(f'Execution Time vs Matrix Size - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'{dataset_name}_time_vs_size.png'))
    plt.close()
    
    # График распределения времени по решателям
    plt.figure(figsize=(14, 8))
    df.boxplot(column='time', by='solver', grid=True, vert=True, showfliers=False)
    plt.title(f'Execution Time Distribution - {dataset_name}')
    plt.suptitle('')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{dataset_name}_time_distribution.png'))
    plt.close()

def generate_summary_report(df: pd.DataFrame, results_path: str):
    """
    Генерирует сводный отчет по всем наборам данных
    """
    if df.empty:
        print("No data to generate report")
        return
    
    # Создаем папку для отчетов
    report_path = os.path.join(results_path, "reports")
    os.makedirs(report_path, exist_ok=True)
    
    # 1. Сводный отчет по методам
    solver_report = df.groupby('solver').agg(
        num_matrices=('matrix', 'nunique'),
        avg_deviation=('deviation', 'mean'),
        std_deviation=('deviation', 'std'),
        min_deviation=('deviation', 'min'),
        max_deviation=('deviation', 'max'),
        success_rate=('deviation', lambda x: (x == 0).mean()),
        avg_time=('time', 'mean'),
        std_time=('time', 'std'),
        percentile_90_time=('time', lambda x: np.percentile(x, 90))
    ).reset_index()
    
    solver_report = solver_report.sort_values(by='avg_deviation')
    solver_report.to_csv(os.path.join(report_path, "solver_performance.csv"), index=False)
    
    # 2. Отчет по матрицам
    matrix_report = df.groupby(['matrix', 'size']).agg(
        best_cost=('best_cost', 'first'),
        solver_count=('solver', 'nunique'),
        best_solver=('cost', lambda x: df.loc[x.idxmax(), 'solver']),
        best_solver_cost=('cost', 'max')
    ).reset_index()
    
    matrix_report['deviation'] = (matrix_report['best_cost'] - matrix_report['best_solver_cost']) / matrix_report['best_cost'] * 100.0
    matrix_report.to_csv(os.path.join(report_path, "matrix_summary.csv"), index=False)
    
    # 3. Визуализация
    plot_summary_results(df, report_path)
    
    print("\nSummary report generated:")
    print(f" - Solver performance: {os.path.join(report_path, 'solver_performance.csv')}")
    print(f" - Matrix summary: {os.path.join(report_path, 'matrix_summary.csv')}")
    print(f" - Visualizations saved in: {report_path}")

def plot_summary_results(df: pd.DataFrame, output_path: str):
    """
    Генерирует визуализации для сводного отчета
    """
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        plt.scatter(
            solver_df['size'], 
            solver_df['deviation'],
            label=solver,
            alpha=0.5
        )
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Deviation from Best Solution (%)')
    plt.title('Solution Quality vs Matrix Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'quality_vs_size.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        plt.scatter(
            solver_df['size'], 
            solver_df['time'],
            label=solver,
            alpha=0.5
        )
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')
    plt.title('Execution Time vs Matrix Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'time_vs_size.png'))
    plt.close()
    
    plt.figure(figsize=(14, 8))
    df.boxplot(column='deviation', by='solver', grid=True, vert=False)
    plt.title('Deviation Distribution by Solver')
    plt.suptitle('')
    plt.xlabel('Deviation from Best Solution (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'deviation_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(14, 8))
    df.boxplot(column='time', by='solver', grid=True, vert=False, showfliers=False)
    plt.title('Execution Time Distribution by Solver (without outliers)')
    plt.suptitle('')
    plt.xlabel('Execution Time (seconds)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'time_distribution.png'))
    plt.close()

def plot_gda_performance(solver: GreatDelugeAlgorithm, save_path: str):
    """
    Сохраняет график производительности алгоритма Great Deluge
    """
    plt.figure(figsize=(12, 6))
    
    # График стоимости
    plt.subplot(1, 2, 1)
    plt.plot(solver.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.grid(True)
    
    # График уровня воды
    plt.subplot(1, 2, 2)
    plt.plot(solver.water_level_history)
    plt.xlabel('Iteration')
    plt.ylabel('Water Level')
    plt.title('Water Level History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()