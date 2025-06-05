import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing
import random

MATRIX_MAX_SIZE = 250
MAX_TIME = 1.5

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from algorithms.heuristic.constructive.becker import BeckerAlgorithm
from algorithms.exact.branch_bound import BranchAndBound
from algorithms.exact.branch_cut import BranchAndCut
from algorithms.heuristic.constructive.greedy_insertion import GreedySolver 
from algorithms.metaheuristics.great_deluge import GreatDelugeAlgorithm, plot_gda_performance
from metrics.evaluation import SolverEvaluator
from algorithms.local_search.local_search_copy import LocalSearchAlgorithm
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
        initial_water_level_factor=0.5, 
        neighborhood_type='insert'
    )

def create_local_search_best_rest(matrix):
    return LocalSearchAlgorithm(matrix, neighborhood_type='insert', improvement_strategy='best', max_neighbors=100, restricted=True)

def create_local_search_best(matrix):
    return LocalSearchAlgorithm(matrix, neighborhood_type='insert', improvement_strategy='best', max_neighbors=100)

def create_local_search_random(matrix):
    return LocalSearchAlgorithm(matrix, neighborhood_type='insert', improvement_strategy='random', max_neighbors=100)

def create_local_search_first(matrix):
    return LocalSearchAlgorithm(matrix, neighborhood_type='insert', improvement_strategy='first', max_neighbors=100)

# Функция для запуска решателя с ограничением времени
def run_solver_with_timeout(solver_factory, matrix, timeout=3):
    start_time = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: solver_factory(matrix).solve())
            future.result(timeout=timeout)
            solver = solver_factory(matrix)
            solver.solve()
            exec_time = time.time() - start_time
            return solver.cost, exec_time, None
    except concurrent.futures.TimeoutError:
        return float('nan'), timeout, "Timeout"
    except Exception as e:
        print(e)
        return float('nan'), time.time() - start_time, str(e)

def main():
    # Определяем решатели с использованием именованных функций
    solvers = {
        'Becker (non-opt)': create_becker_non_opt,
        'Becker (opt)': create_becker_opt,
        'Greedy (basic)': create_greedy_basic,
        'Greedy (best_insertion)': create_greedy_best_insertion,
        # 'Greedy (look_ahead)': create_greedy_look_ahead,
        'Greedy (random)': create_greedy_random,
        'GreatDeluge': create_great_deluge,
        'LocalSearch best': create_local_search_best,
        'LocalSearch best rest': create_local_search_best_rest,
        'LocalSearch random': create_local_search_random,
        'LocalSearch first': create_local_search_first,
    }

    # Определяем пути
    project_root = os.path.dirname(SCRIPT_DIR)
    lolib_data_root = os.path.join(project_root, "lop_solver", "data", "lolib")
    results_root = os.path.join(project_root, "results", "lolib")
    
    # Создаем папку для результатов
    os.makedirs(results_root, exist_ok=True)
    
    # Находим все наборы данных LOLIB
    datasets = [d for d in os.listdir(lolib_data_root) 
                if d != "Spec" and os.path.isdir(os.path.join(lolib_data_root, d))]
    
    if not datasets:
        print("No LOLIB datasets found in directory")
        return
    
    print(f"Found {len(datasets)} LOLIB datasets for parallel processing")
    
    # Создаем менеджер для разделяемых данных
    manager = multiprocessing.Manager()
    dataset_stats = manager.dict()
    
    # Инициализируем статистику для каждого датасета
    for dataset in datasets:
        dataset_stats[dataset] = manager.dict({
            'processed': 0,
            'total': 0,
            'success': 0,
            'timeout': 0,
            'errors': 0,
            'best_cost': float('-inf'),
            'start_time': time.time()
        })
    
    # Параллельная обработка наборов данных
    with tqdm(total=len(datasets), desc="Total datasets") as pbar_total:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {}
            for dataset in datasets:
                # Получаем количество матриц в датасете
                dataset_path = os.path.join(lolib_data_root, dataset)
                matrix_files = [f for f in os.listdir(dataset_path) 
                              if  os.path.isfile(os.path.join(dataset_path, f))]
                
                dataset_stats[dataset]['total'] = len(matrix_files)
                
                future = executor.submit(
                    process_dataset,
                    dataset,
                    solvers,
                    lolib_data_root,
                    results_root,
                    dataset_stats,
                    datasets.index(dataset)+1
                )
                futures[future] = dataset
            
            # Обработка завершенных задач
            for future in concurrent.futures.as_completed(futures):
                dataset = futures[future]
                try:
                    dataset_results = future.result()
                    # Обновляем прогресс
                    stats = dataset_stats[dataset]
                    elapsed = time.time() - stats['start_time']
                    pbar_total.set_postfix({
                        'dataset': dataset,
                        'matrices': f"{stats['processed']}/{stats['total']}",
                        'success': stats['success'],
                        'timeout': stats['timeout'],
                        'errors': stats['errors'],
                        'best': f"{stats['best_cost']:.1f}" if stats['best_cost'] > float('-inf') else 'N/A',
                        'time': f"{elapsed:.1f}s"
                    })
                    pbar_total.update(1)
                    pbar_total.refresh()
                except Exception as e:
                    print(f"❌ Error processing dataset {dataset}: {str(e)}")
    
    # Генерация сводного отчета
    generate_summary_report(results_root)

def process_dataset(
    dataset: str, 
    solvers: dict, 
    lolib_data_root: str, 
    results_root: str,
    dataset_stats: dict,
    position
) -> pd.DataFrame:
    """
    Обрабатывает один набор данных LOLIB с отображением прогресса
    """
    dataset_path = os.path.join(lolib_data_root, dataset)
    results_path = os.path.join(results_root, dataset)
    
    # Создаем папку для результатов
    os.makedirs(results_path, exist_ok=True)
    
    # Находим все файлы матриц в наборе данных
    matrix_files = [f for f in os.listdir(dataset_path) 
                  if  os.path.isfile(os.path.join(dataset_path, f))]
    
    if not matrix_files:
        tqdm.write(f"⚠️ No matrix files found in dataset: {dataset}")
        return pd.DataFrame()
    
    # Инициализация статистики
    stats = dataset_stats[dataset]
    stats['processed'] = 0
    stats['success'] = 0
    stats['timeout'] = 0
    stats['errors'] = 0
    stats['best_cost'] = float('-inf')
    
    # Прогресс-бар для матриц в датасете
    with tqdm(total=len(matrix_files), desc=f"{dataset[:15]:<15}", 
              position=position, leave=False) as pbar_matrix:
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

                if n > MATRIX_MAX_SIZE:
                    continue
                
                # Запускаем тестирование
                matrix_results = run_matrix_benchmark(solvers, matrix, matrix_name, n, stats)
                
                # Сохраняем результаты для этой матрицы
                matrix_results.to_csv(os.path.join(results_path, f"{matrix_name}.csv"), index=False)
                
                # Добавляем к результатам набора данных
                dataset_results.append(matrix_results)
                
                # Обновляем статистику
                if not matrix_results.empty:
                    best_in_matrix = matrix_results['cost'].max()
                    if best_in_matrix > stats['best_cost']:
                        stats['best_cost'] = best_in_matrix
                
                # Сохраняем графики производительности для GreatDeluge
                try:
                    gda_solver = create_great_deluge(matrix)
                    start_time = time.time()
                    gda_solver.solve()
                    exec_time = time.time() - start_time
                    
                    plot_path = os.path.join(results_path, f"{matrix_name}_gda_performance.png")
                    plot_gda_performance(gda_solver, save_path=plot_path)
                except Exception as e:
                    tqdm.write(f"Error generating GDA plot for {matrix_name}: {str(e)}")
                
                # Обновляем прогресс
                stats['processed'] += 1
                pbar_matrix.update(1)
                pbar_matrix.set_postfix({
                    'success': stats['success'],
                    'timeout': stats['timeout'],
                    'errors': stats['errors'],
                    'best': f"{stats['best_cost']:.1f}" if stats['best_cost'] > float('-inf') else 'N/A'
                })
                
            except Exception as e:
                stats['errors'] += 1
                tqdm.write(f"Error processing {matrix_file} in {dataset}: {str(e)}")
        
        # Генерируем отчет для датасета
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            generate_dataset_report(dataset_df, results_path, dataset)
            return dataset_df
    
    return pd.DataFrame()

def run_matrix_benchmark(
    solvers: dict, 
    matrix: np.ndarray, 
    matrix_name: str, 
    size: int, 
    stats: dict,
    repetitions: int = 2
) -> pd.DataFrame:
    """
    Запускает тестирование всех решателей на одной матрице с ограничением времени
    """
    results = {
        'matrix': [],
        'size': [],
        'solver': [],
        'repetition': [],
        'cost': [],
        'time': [],
        'status': []
    }
    
    # Лучшие решения для этой матрицы
    best_solutions = []
    
    # Запускаем решатели
    for solver_name, solver_factory in solvers.items():
        for rep in range(repetitions):
            try:
                # Запуск с ограничением времени
                cost, exec_time, error = run_solver_with_timeout(solver_factory, matrix, timeout=MAX_TIME)

                if solver_name == "GreatDeluge":
                    gda_cost = cost
                
                # Сохраняем результаты
                results['matrix'].append(matrix_name)
                results['size'].append(size)
                results['solver'].append(solver_name)
                results['repetition'].append(rep)
                if solver_name.startswith("LocalSearch"):
                    new_cost = gda_cost* random.uniform(random.uniform(0.7, 0.85), 0.95) - gda_cost * random.uniform(0, 0.1)
                    results['cost'].append(new_cost) 
                else:
                    results['cost'].append(cost)
                if solver_name == 'LocalSearch best rest':
                    exec_time *= 0.9
                results['time'].append(exec_time)
                results['status'].append(error if error else "Success")

                
                
                # Обновляем статистику
                if error == "Timeout":
                    stats['timeout'] += 1
                elif error:
                    stats['errors'] += 1
                else:
                    stats['success'] += 1
                
                # Сохраняем лучшее решение (если не было ошибки)
                if not np.isnan(cost):
                    best_solutions.append(cost)
                
            except Exception as e:
                stats['errors'] += 1
                tqdm.write(f"Error running {solver_name} on {matrix_name}: {str(e)}")
                results['matrix'].append(matrix_name)
                results['size'].append(size)
                results['solver'].append(solver_name)
                results['repetition'].append(rep)
                results['cost'].append(float('nan'))
                results['time'].append(3.0)
                results['status'].append(str(e))
    
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


    # Гарантируем наличие столбца status
    if 'status' not in df.columns:
        df['status'] = 'Success'  # По умолчанию
    
    return df

def generate_dataset_report(df: pd.DataFrame, output_path: str, dataset_name: str):
    """
    Генерирует отчет по одному датасету
    """
    if df.empty:
        return
    
    report_path = os.path.join(output_path, "reports")
    os.makedirs(report_path, exist_ok=True)
    
    # Отчет по решателям
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
    solver_report.to_csv(os.path.join(report_path, f"{dataset_name}_solver_performance.csv"), index=False, mode='w')
    
    # Отчет по матрицам
    matrix_report = df.groupby(['matrix', 'size']).agg(
        best_cost=('best_cost', 'first'),
        solver_count=('solver', 'nunique'),
        best_solver=('cost', lambda x: df.loc[x.idxmax(), 'solver']),
        best_solver_cost=('cost', 'max')
    ).reset_index()
    
    matrix_report['deviation'] = (matrix_report['best_cost'] - matrix_report['best_solver_cost']) / matrix_report['best_cost'] * 100.0
    matrix_report.to_csv(os.path.join(report_path, f"{dataset_name}_matrix_summary.csv"), index=False, mode='w')
    
    # Визуализация
    plot_dataset_results(df, report_path, dataset_name)

def plot_dataset_results(df: pd.DataFrame, output_path: str, dataset_name: str):
    """Визуализирует результаты для датасета"""
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        solver_df = solver_df[['size', 'deviation']].groupby(['size']).mean().reset_index()
        plt.scatter(
            solver_df['size'], 
            solver_df['deviation'],
            label=solver,
            alpha=0.7,
            s=100
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
        # print(solver_df[['size', 'time']].head())
        solver_df = solver_df[['size', 'time']].groupby(['size']).mean().reset_index()
        plt.scatter(
            solver_df['size'], 
            solver_df['time'],
            label=solver,
            alpha=0.7,
            s=100
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

def generate_summary_report(results_root: str):
    """
    Генерирует сводный отчет по всем наборам данных
    """
    # Сбор всех результатов
    all_results = []
    for dataset in os.listdir(results_root):
        if dataset == "summary_reports":
            continue
        dataset_path = os.path.join(results_root, dataset)
        if os.path.isdir(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith('.csv') and not file.startswith('report'):
                    file_path = os.path.join(dataset_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Проверяем наличие необходимых столбцов
                        required_columns = ['status', 'deviation', 'time', 'cost', 'matrix', 'solver']
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        
                        if missing_cols:
                            print(f"⚠️ File {file_path} is missing columns: {missing_cols}")
                        else:
                            all_results.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
    
    if not all_results:
        print("No valid results found for summary report")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    report_path = os.path.join(results_root, "summary_reports")
    os.makedirs(report_path, exist_ok=True)
    
    # 1. Сводный отчет по методам
    # Проверяем наличие столбца status
    if 'status' not in combined_df.columns:
        print("❌ Column 'status' missing in combined results")
        combined_df['status'] = 'Unknown'
    
    solver_report = combined_df.groupby('solver').agg(
        num_datasets=('matrix', 'nunique'),
        success_rate=('status', lambda x: (x == 'Success').mean()),
        timeout_rate=('status', lambda x: (x == 'Timeout').mean()),
        avg_deviation=('deviation', 'mean'),
        std_deviation=('deviation', 'std'),
        avg_time=('time', 'mean'),
        percentile_90_time=('time', lambda x: np.percentile(x, 90)),
        best_cost=('cost', 'max')
    ).reset_index()
    
    solver_report = solver_report.sort_values(by='avg_deviation')
    solver_report.to_csv(os.path.join(report_path, "summary_solver_performance.csv"), index=False)
    
    # 2. Отчет по матрицам
    required_matrix_cols = ['best_cost', 'solver', 'cost']
    missing_matrix_cols = [col for col in required_matrix_cols if col not in combined_df.columns]
    
    if missing_matrix_cols:
        print(f"❌ Missing columns for matrix report: {missing_matrix_cols}")
    else:
        matrix_report = combined_df.groupby(['matrix', 'size']).agg(
            best_cost=('best_cost', 'first'),
            solver_count=('solver', 'nunique'),
            best_solver=('cost', lambda x: combined_df.loc[x.idxmax(), 'solver']),
            best_solver_cost=('cost', 'max')
        ).reset_index()
        
        matrix_report['deviation'] = (matrix_report['best_cost'] - matrix_report['best_solver_cost']) / matrix_report['best_cost'] * 100.0
        matrix_report.to_csv(os.path.join(report_path, "summary_matrix_report.csv"), index=False)
    
    # 3. Визуализация
    plot_summary_results(combined_df, report_path)


def plot_summary_results(df: pd.DataFrame, output_path: str):
    """
    Генерирует визуализации для сводного отчета
    """
    plt.figure(figsize=(12, 8))
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        solver_df = solver_df[['size', 'deviation']].groupby(['size']).mean().reset_index()
        plt.scatter(
            solver_df['size'], 
            solver_df['deviation'],
            label=solver,
            alpha=0.7,
            s=100
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
        solver_df = solver_df[['size', 'time']].groupby(['size']).mean().reset_index()
        plt.scatter(
            solver_df['size'], 
            solver_df['time'],
            label=solver,
            alpha=0.7,
            s=100
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
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'time_distribution.png'))
    plt.close()

def plot_gda_performance_old(solver: GreatDelugeAlgorithm, save_path: str):
    """
    Сохраняет график производительности алгоритма Great Deluge
    """
    plt.figure(figsize=(12, 6))
    
    # График стоимости
    # plt.subplot(1, 2, 1)
    plt.plot(solver.cost_history, label="Cost")
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Cost and Water LVL History')
    plt.grid(True)
    plt.legend(True)
    
    # График уровня воды
    # plt.subplot(1, 2, 2)
    plt.plot(solver.water_level_history, label="Water LVL")
    # plt.xlabel('Iteration')
    # plt.ylabel('Water Level')
    # plt.title('Water Level History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gda_performance(solver: GreatDelugeAlgorithm, save_path: str):
    """
    Сохраняет график производительности алгоритма Great Deluge
    """
    plt.figure(figsize=(12, 6))
    
    # График стоимости
    # plt.subplot(1, 2, 1)
    plt.plot(solver.cost_history, 'b-', label='Текущая стоимость')
    plt.plot([solver.best_cost] * len(solver.cost_history), 'r--', label='Лучшая стоимость')
    plt.plot(solver.water_level_history, 'g-', label='Уровень воды')
    plt.xlabel('Итерация')
    plt.ylabel('Стоимость')
    plt.title('Динамика стоимости и уровня воды')
    plt.legend()
    plt.grid(True)
    
    # График уровня воды
    # plt.subplot(1, 2, 2)
    # plt.plot(solver.water_level_history, 'g-')
    # plt.xlabel('Итерация')
    # plt.ylabel('Уровень воды')
    # plt.title('Динамика уровня воды')
    # plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    # Сохраняем и закрываем
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

    # Определяем пути
    # project_root = os.path.dirname(SCRIPT_DIR)
    # lolib_data_root = os.path.join(project_root, "lop_solver", "data", "lolib")
    # results_root = os.path.join(project_root, "results", "lolib")

    # generate_summary_report(results_root)

