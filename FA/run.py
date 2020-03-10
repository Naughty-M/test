import numpy as np
from FA import FireflyAlgorithm


if __name__ == "__main__":
    bound = np.tile([[-600], [600]], 25)
    fa = FireflyAlgorithm(60, 2, bound, 200, [1.0, 0.000001, 0.6])
    fa.solve()