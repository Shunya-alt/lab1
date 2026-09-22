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

# тесты для sum_prod
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

#тесты для binarize


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

#тесты для chess
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
    raise NotImplementedError  # TODO


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    raise NotImplementedError  # TODO
