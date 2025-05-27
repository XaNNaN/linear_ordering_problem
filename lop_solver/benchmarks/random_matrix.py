import numpy as np

def generate_random_lop_instance(n: int, seed: int = 42) -> np.ndarray:
    """Генерирует случайную матрицу для LOP."""
    np.random.seed(seed)
    matrix = np.random.randint(0, 100, size=(n, n))
    np.fill_diagonal(matrix, 0)  # Диагональ = 0
    return matrix