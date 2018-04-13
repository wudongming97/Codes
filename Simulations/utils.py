def is_prime(n):
    """
    :param n: 整数
    :return: 是否素数
    """
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

