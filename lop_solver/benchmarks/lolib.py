import numpy as np

def load_lolib_matrix(filepath):
    """
    Загружает матрицу из файла LOLIB в формате:
    Первая строка - размерность N
    Последующие N строк - строки матрицы
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    n = int(lines[0])
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(1, n+1):
        if i >= len(lines):
            raise ValueError(f"Ожидалось {n} строк матрицы, но найдено только {len(lines)-1}")
        
        row = list(map(int, lines[i].split()))
        if len(row) != n:
            raise ValueError(f"Строка {i} содержит {len(row)} элементов, ожидалось {n}")
        
        matrix[i-1] = row

    np.fill_diagonal(matrix, 0)
    
    return matrix


def load_lolib_dataset(directory):
    """Загружает все матрицы из директории"""
    import glob
    datasets = {}
    for filepath in glob.glob(f"{directory}/*.txt"):
        try:
            name = filepath.split('/')[-1].split('.')[0]
            datasets[name] = load_lolib_matrix(filepath)
        except Exception as e:
            print(f"Ошибка загрузки {filepath}: {str(e)}")
    return datasets