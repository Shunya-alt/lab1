"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = 'aeiou'
    k = 0
    for i in text.lower():
        if i in vowels:
            k += 1
    return k


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    k = 0
    while number >= 10:
        i = 1
        for j in str(number):
            i *= int(j)
        number = i
        k += 1
    return k


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = 0
    for x, y in zip(predicted, expected):
        total += (x - y) ** 2
    return total / n



def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = []
    divisor = 2

    while divisor * divisor <= number:
        if number % divisor == 0:
            k = 0
            while number % divisor == 0:
                number //= divisor
                k += 1
            if k == 1:
                result.append(f"({divisor})")
            else:
                result.append(f"({divisor}**{k})")
        divisor += 1

    # если number > 1 — остался простой множитель
    if number > 1:
        result.append(f"({number})")

    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0

    while total < cube_count:
        k += 1
        total += k * k

    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = str(data.value)
    n = len(number)
    half = n // 2

    left = number[:half]              # левая половина
    right = number[-half:] if half else ""  # правая половина

    left_sum = sum(int(d) for d in left)
    right_sum = sum(int(d) for d in right)

    return left_sum == right_sum
