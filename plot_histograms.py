"""Построение гистограмм для задачи matrix_statistics."""
import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(result, bins=15):
    """Рисует 4 гистограммы: средние и дисперсии по строкам и столбцам."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(result.row_means, bins=bins, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Средние по строкам")
    axes[0, 0].set_xlabel("Значение")
    axes[0, 0].set_ylabel("Частота")
    
    axes[0, 1].hist(result.column_means, bins=bins, color="seagreen", edgecolor="black")
    axes[0, 1].set_title("Средние по столбцам")
    axes[0, 1].set_xlabel("Значение")
    axes[0, 1].set_ylabel("Частота")
    
    axes[1, 0].hist(result.row_variances, bins=bins, color="coral", edgecolor="black")
    axes[1, 0].set_title("Дисперсии по строкам")
    axes[1, 0].set_xlabel("Значение")
    axes[1, 0].set_ylabel("Частота")
    
    axes[1, 1].hist(result.column_variances, bins=bins, color="gold", edgecolor="black")
    axes[1, 1].set_title("Дисперсии по столбцам")
    axes[1, 1].set_xlabel("Значение")
    axes[1, 1].set_ylabel("Частота")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = RandomMatrixInput(rows=50, columns=50, mean=0.0, std=1.0, seed=42)
    result = matrix_statistics(data)
    plot_histograms(result)