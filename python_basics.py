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

# тесты для count_vowels
print(count_vowels(TextInput("hello")))   # 2
print(count_vowels(TextInput("")))        # 0

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))

# тесты для has_unique_characters
print(has_unique_characters(TextInput("world")))  # True
print(has_unique_characters(TextInput("hello")))  # False

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")

# тесты для count_one_bits
print(count_one_bits(PositiveIntegerInput(13)))  # 3 (1101)
print(count_one_bits(PositiveIntegerInput(0)))   # 0 (0)
print(count_one_bits(PositiveIntegerInput(255))) # 8 (11111111)

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

# тесты для multiplicative_persistence
print(multiplicative_persistence(PositiveIntegerInput(4)))    # 0
print(multiplicative_persistence(PositiveIntegerInput(39)))   # 3
print(multiplicative_persistence(PositiveIntegerInput(10)))   # 1 (1*0 = 0)
print(multiplicative_persistence(PositiveIntegerInput(25)))   # 2 (2*5=10, 1*0=0)

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = 0
    for x, y in zip(predicted, expected):
        total += (x - y) ** 2
    return total / n

# тесты для mse
print(mse(VectorPairInput([1, 2, 3], [1, 2, 3])))  # 0.0
print(mse(VectorPairInput([1, 2, 3], [2, 3, 4])))  # 1.0
print(mse(VectorPairInput([0, 0], [3, 4])))        # 12.5  ((9+16)/2)


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

# тесты для prime_factorization
print(prime_factorization(PositiveIntegerInput(86240)))  # (2**5)(5)(7**2)(11)
print(prime_factorization(PositiveIntegerInput(1)))      # "" (пустая строка)
print(prime_factorization(PositiveIntegerInput(7)))      # (7)


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

# тесты для pyramid
print(pyramid(PositiveIntegerInput(6)))    # It is impossible
print(pyramid(PositiveIntegerInput(14)))   # 3

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = str(data.value)
    n = len(number)
    half = n // 2

    left = number[:half]              # левая половина
    right = number[-half:] if half else ""  # правая половина

    left_sum = sum(int(d) for d in left)
    right_sum = sum(int(d) for d in right)

    return left_sum == right_sum

#тесты для is_balanced_number
print(is_balanced_number(PositiveIntegerInput(1234006)))  # True
print(is_balanced_number(PositiveIntegerInput(12)))       # False (1 != 2)
print(is_balanced_number(PositiveIntegerInput(5)))        # True  (одна цифра)