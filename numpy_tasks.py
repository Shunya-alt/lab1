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


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    mask = matrix > threshold
    result = mask.astype(int)
    return result


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


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1
    n = len(labels)
    result = np.zeros((n, class_count), dtype=int)
    result[np.arange(n), labels] = 1 
    return result