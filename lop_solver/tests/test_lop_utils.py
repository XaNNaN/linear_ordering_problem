# pytest tests/test_lop_utils.py -v
import numpy as np
import pytest
from utils.lop_utils import calculate_lop_cost, generate_neighborhood, is_valid_permutation, calculate_lop_cost_fast

def test_calculate_lop_cost_2x2():
    """Тест для матрицы 2x2."""
    matrix = np.array([
        [0, 5],
        [2, 0]
    ])
    ordering = [0, 1]
    assert calculate_lop_cost(matrix, ordering) == 5  # matrix[0][1]
    assert calculate_lop_cost_fast(matrix, ordering) == 5  # matrix[0][1]

    ordering = [1, 0]
    assert calculate_lop_cost(matrix, ordering) == 2  # matrix[1][0]
    assert calculate_lop_cost_fast(matrix, ordering) == 2  # matrix[1][0]

def test_calculate_lop_cost_3x3():
    """Тест для матрицы 3x3 с известным результатом."""
    matrix = np.array([
        [0, 3, 1],
        [2, 0, 4],
        [5, 6, 0]
    ])
    ordering = [0, 1, 2]
    # Сумма: matrix[0][1] + matrix[0][2] + matrix[1][2] = 3 + 1 + 4 = 8
    assert calculate_lop_cost(matrix, ordering) == 8
    assert calculate_lop_cost_fast(matrix, ordering) == 8

    ordering = [2, 1, 0]
    # Сумма: matrix[2][1] + matrix[2][0] + matrix[1][0] = 6 + 5 + 2 = 13
    assert calculate_lop_cost(matrix, ordering) == 13
    assert calculate_lop_cost_fast(matrix, ordering) == 13

def test_calculate_lop_cost_zero_matrix():
    """Тест для нулевой матрицы."""
    matrix = np.zeros((4, 4))
    ordering = [0, 1, 2, 3]
    assert calculate_lop_cost(matrix, ordering) == 0
    assert calculate_lop_cost_fast(matrix, ordering) == 0

@pytest.mark.parametrize("ordering, expected", [
    ([0, 1, 2], 8),
    ([2, 1, 0], 13),
    ([1, 0, 2], 2 + 1 + 4)  # matrix[1][0] + matrix[1][2] + matrix[0][2]
])
def test_calculate_lop_cost_parametrized(ordering, expected):
    """Параметризованный тест для разных перестановок."""
    matrix = np.array([
        [0, 3, 1],
        [2, 0, 4],
        [5, 6, 0]
    ])
    assert calculate_lop_cost(matrix, ordering) == expected
    assert calculate_lop_cost_fast(matrix, ordering) == expected


def test_calculate_lop_cost_performance(benchmark):
    matrix = np.random.rand(100, 100)
    ordering = list(range(100))
    benchmark(calculate_lop_cost, matrix, ordering)


def test_generate_neighborhood_size():
    """Тест на количество соседей для swap-окрестности."""
    ordering = [0, 1, 2]
    neighborhood = generate_neighborhood(ordering)
    assert len(neighborhood) == 3  # C(3,2) = 3 swap-соседа

def test_generate_neighborhood_content():
    """Тест на содержание окрестности."""
    ordering = [0, 1, 2]
    neighborhood = generate_neighborhood(ordering)
    expected = [
        [1, 0, 2],  # swap(0,1)
        [2, 1, 0],  # swap(0,2)
        [0, 2, 1]   # swap(1,2)
    ]
    assert all(n in neighborhood for n in expected)


@pytest.mark.parametrize("ordering, n, expected", [
    ([0, 1, 2], 3, True),
    ([2, 1, 0], 3, True),
    ([0, 1], 3, False),  # Не хватает элемента
    ([0, 1, 1], 3, False)  # Дубликат
])
def test_is_valid_permutation(ordering, n, expected):
    assert is_valid_permutation(ordering, n) == expected
    

def test_calculate_lop_cost_fast_performance(benchmark):
    matrix = np.random.rand(100, 100)
    ordering = list(range(100))
    benchmark(calculate_lop_cost_fast, matrix, ordering)

