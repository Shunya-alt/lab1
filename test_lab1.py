"""Тесты для лабораторной работы №1."""
import numpy as np

from grader_contracts.python_basics import (
    PositiveIntegerInput, TextInput, VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput,
    RectangleInput, TimeSeriesInput,
)
from python_basics import (
    count_vowels, has_unique_characters, count_one_bits,
    multiplicative_persistence, mse, prime_factorization,
    pyramid, is_balanced_number,
)
from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns,
    matrix_statistics, chess, draw_rectangle, draw_ellipse,
    analyze_time_series, one_hot,
)


def test_count_vowels():
    assert count_vowels(TextInput("hello")) == 2
    assert count_vowels(TextInput("HELLO")) == 2
    assert count_vowels(TextInput("")) == 0
    print("+ count_vowels")


def test_has_unique_characters():
    assert has_unique_characters(TextInput("world")) == True
    assert has_unique_characters(TextInput("hello")) == False
    assert has_unique_characters(TextInput("")) == True
    print("+ has_unique_characters")


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(13)) == 3
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(255)) == 8
    print("+ count_one_bits")


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    print("+ multiplicative_persistence")


def test_mse():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2, 3], [2, 3, 4])) == 1.0
    assert mse(VectorPairInput([0, 0], [3, 4])) == 12.5
    print("+ mse")


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(7)) == "(7)"
    print("+ prime_factorization")


def test_pyramid():
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(6)) == "It is impossible"
    print("+ pyramid")


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) == True
    assert is_balanced_number(PositiveIntegerInput(123456)) == False
    assert is_balanced_number(PositiveIntegerInput(5)) == True
    print("+ is_balanced_number")


def test_sum_prod():
    A1 = np.array([[1, 2], [3, 4]])
    A2 = np.array([[0, 1], [1, 0]])
    x1 = np.array([[3], [4]])
    x2 = np.array([[1], [2]])
    matrices = np.array([A1, A2])
    vectors = np.array([x1, x2])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert np.array_equal(result, np.array([[13], [26]]))
    print("+ sum_prod")


def test_binarize():
    M = np.array([[1, 5, 3], [4, 2, 6]])
    result = binarize(BinarizeInput(M, 3))
    expected = np.array([[0, 1, 0], [1, 0, 1]])
    assert np.array_equal(result, expected)
    print("+ binarize")


def test_unique_rows():
    M = np.array([[1, 2, 1, 3], [5, 5, 5, 5]])
    result = unique_rows(MatrixInput(M))
    assert set(result[0]) == {1, 2, 3}
    assert set(result[1]) == {5}
    print("+ unique_rows")


def test_unique_columns():
    M = np.array([[1, 2, 1], [5, 5, 5], [1, 2, 3]])
    result = unique_columns(MatrixInput(M))
    assert set(result[0]) == {1, 5}
    assert set(result[1]) == {2, 5}
    assert set(result[2]) == {1, 5, 3}
    print("+ unique_columns")


def test_matrix_statistics():
    data = RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=42)
    r = matrix_statistics(data)
    assert r.matrix.shape == (3, 4)
    assert r.row_means.shape == (3,)
    assert r.column_means.shape == (4,)
    assert r.row_variances.shape == (3,)
    assert r.column_variances.shape == (4,)
    print("+ matrix_statistics")


def test_chess():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(result, expected)
    print("+ chess")


def test_draw_rectangle():
    img = draw_rectangle(RectangleInput(
        width=4, height=2,
        image_height=6, image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    ))
    assert img.shape == (6, 8, 3)
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert tuple(img[3, 3]) == (255, 0, 0)
    print("+ draw_rectangle")


def test_draw_ellipse():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=1,
        image_height=5, image_width=7,
        shape_color=(0, 255, 0),
        background_color=(0, 0, 0),
    ))
    assert img.shape == (5, 7, 3)
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert tuple(img[2, 3]) == (0, 255, 0)
    print("+ draw_ellipse")


def test_analyze_time_series():
    values = np.array([1, 3, 2, 5, 1, 4], dtype=float)
    r = analyze_time_series(TimeSeriesInput(values=values, window=3))
    assert list(r.local_maxima_indices) == [1, 3]
    assert list(r.local_minima_indices) == [2, 4]
    assert len(r.moving_average) == 4
    print("+ analyze_time_series")


def test_one_hot():
    labels = np.array([0, 2, 3, 0])
    result = one_hot(OneHotInput(labels=labels, class_count=4))
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert np.array_equal(result, expected)
    print("+ one_hot")


if __name__ == "__main__":
    test_count_vowels()
    test_has_unique_characters()
    test_count_one_bits()
    test_multiplicative_persistence()
    test_mse()
    test_prime_factorization()
    test_pyramid()
    test_is_balanced_number()
    test_sum_prod()
    test_binarize()
    test_unique_rows()
    test_unique_columns()
    test_matrix_statistics()
    test_chess()
    test_draw_rectangle()
    test_draw_ellipse()
    test_analyze_time_series()
    test_one_hot()
    print("\nВсе тесты пройдены!")