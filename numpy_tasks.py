"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = matrices.shape[1]
    result = np.zeros((n, 1))
    for i in range(len(matrices)):
        result += matrices[i] @ vectors[i]
    return result

# тест для sum_prod
A1 = np.array([[1, 2], [3, 4]])
A2 = np.array([[0, 1], [1, 0]])
x1 = np.array([[3], [4]])
x2 = np.array([[1], [2]])   
matrices = np.array([A1, A2])    
vectors  = np.array([x1, x2])        
data = MatrixVectorBatchInput(matrices, vectors)
print(sum_prod(data))
    # Ожидаем:
    # [[13]
    #  [26]]

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    mask = matrix > threshold
    result = mask.astype(int)
    return result

#тест для binarize
M = np.array([[1, 5, 3], [4, 2, 6]])
result = binarize(BinarizeInput(M, 3))
print(result)
# [[0 1 0]
# [1 0 1]]

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for row in matrix:
        unique_values = list(set(row))
        result.append(unique_values)
    return result

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        unique_values = list(set(column))
        result.append(unique_values)
    return result

#тест для unique_rows
M = np.array([[1, 2, 1, 3], [5, 5, 5, 5]])
result = unique_rows(MatrixInput(M))
print(result) #[[1, 2, 3], [5]]

#тест для unique_columns
M = np.array([[1, 2, 1],
              [5, 5, 5],
              [1, 2, 3]])
result = unique_columns(MatrixInput(M))
print(result) #[[1, 5], [2, 5], [1, 5, 3]]

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_variances = matrix.var(axis=1)
    column_variances = matrix.var(axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )

#тест для matrix_statistics
data = RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=42)
r = matrix_statistics(data)
print(r.matrix.shape)           # (3, 4)
print(r.row_means.shape)         # (3,)
print(r.column_means.shape)      # (4,)
print(r.row_variances.shape)     # (3,)
print(r.column_variances.shape)  # (4,)


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.zeros((rows, columns))    
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                matrix[i, j] = first
            else:
                matrix[i, j] = second
    
    return matrix

#тест для chess
result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
print(result)
result = chess(ChessInput(rows=2, columns=3, first=7, second=9))
print(result)

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8) 
    cy = image_height // 2 
    cx = image_width // 2

    y_start = cy - height // 2
    y_end = y_start + height
    x_start = cx - width // 2
    x_end = x_start + width

    image[y_start:y_end, x_start:x_end] = shape_color   
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    cy = image_height // 2
    cx = image_width // 2   
    # Сетка координат
    yy, xx = np.indices((image_height, image_width))

    mask = ((xx - cx) ** 2 / semi_axis_x ** 2) + ((yy - cy) ** 2 / semi_axis_y ** 2) <= 1

    image[mask] = shape_color   
    return image

# Тест для draw_rectangle
img = draw_rectangle(RectangleInput(
    width=4, height=2,
    image_height=6, image_width=8,
    shape_color=(255, 0, 0),
    background_color=(0, 0, 0),
))
print(img.shape)           # (6, 8, 3)

# Тест для draw_ellipse
img = draw_ellipse(EllipseInput(
    semi_axis_x=2, semi_axis_y=1,
    image_height=5, image_width=7,
    shape_color=(0, 255, 0),
    background_color=(0, 0, 0),
))
print(img.shape)           # (5, 7, 3)

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    mean = float(values.mean())
    variance = float(values.var())
    std = float(values.std())

    left  = values[:-2]     
    mid   = values[1:-1]    
    right = values[2:]      
    
    maxima_mask = (mid > left) & (mid > right)
    minima_mask = (mid < left) & (mid < right) 
    #mid начинается с индекса 1
    local_maxima = np.where(maxima_mask)[0] + 1
    local_minima = np.where(minima_mask)[0] + 1
    #cumsum: сумма окна = S[i+p] - S[i], где S — кумулятивная сумма
    cumsum = np.cumsum(np.insert(values, 0, 0))
    moving_average = (cumsum[window:] - cumsum[:-window]) / window  
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )

#тесты для analyze_time_series 
values = np.array([1, 3, 2, 5, 1, 4], dtype=float)
result = analyze_time_series(TimeSeriesInput(values=values, window=3))
print(result.mean)                    # 16/6 ≈ 2.666...
print(result.variance)                # 
print(result.std)                     # 
print(result.local_maxima_indices)    # [1, 3]
print(result.local_minima_indices)    # [2, 4]
print(result.moving_average)          # [2.0, 3.33, 2.67, 3.33]

def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1
    n = len(labels)
    result = np.zeros((n, class_count), dtype=int)
    result[np.arange(n), labels] = 1 
    return result

#тесты для one_hot
labels = np.array([0, 2, 3, 0])
result = one_hot(OneHotInput(labels=labels, class_count=4))
print(result)
# [[1 0 0 0]
# [0 0 1 0]
# [0 0 0 1]
# [1 0 0 0]]