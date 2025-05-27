import numpy as np


def validate_lolib_matrix(matrix):
    """Проверяет свойства матрицы LOP"""
    n = matrix.shape[0]
    assert matrix.shape == (n, n), "Матрица должна быть квадратной"
    assert (np.diag(matrix) == 0).all(), "Диагональ должна быть нулевой"
    print(f"Матрица {n}x{n} успешно валидирована")
    return True

def analyze_matrix(matrix):
    """Возвращает основные характеристики матрицы"""
    n = matrix.shape[0]
    density = np.count_nonzero(matrix) / (n*n - n) * 100  # Исключаем диагональ
    
    return {
        "size": n,
        "density (%)": round(density, 2),
        "min_value": int(np.min(matrix)),
        "max_value": int(np.max(matrix)),
        "avg_value": round(float(np.mean(matrix)), 2),
        "symmetry": check_symmetry(matrix)
    }

def check_symmetry(matrix):
    """Проверяет степень симметрии матрицы"""
    diff = matrix - matrix.T
    return {
        "is_symmetric": np.allclose(matrix, matrix.T),
        "asymmetry_ratio": round(np.sum(np.abs(diff)) / np.sum(matrix), 4)
    }


if __name__ == "__main__":
    import sys
    import os

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from benchmarks.lolib import load_lolib_matrix
    # Загрузка и анализ матрицы
    project_root = os.path.dirname(SCRIPT_DIR)
    file_path = os.path.join(project_root, "data", "lolib", "IO", "N-t59f11xx")
    print(file_path)
    matrix = load_lolib_matrix(file_path)
    validate_lolib_matrix(matrix)
    stats = analyze_matrix(matrix)

    print("Характеристики матрицы:")
    for k, v in stats.items():
        print(f"{k:>15}: {v}")