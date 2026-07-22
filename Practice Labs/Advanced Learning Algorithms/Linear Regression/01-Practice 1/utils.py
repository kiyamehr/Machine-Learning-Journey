import numpy as np # type: ignore

def gen_data():
    np.random.seed(42)

    sizes = np.random.randint(40, 220, 80)

    prices = 900 * sizes + np.random.normal(0, 20_000, len(sizes))
    prices = np.round(prices, 0).astype(int)

    x = sizes.reshape(-1, 1)
    y = prices

    return x, y