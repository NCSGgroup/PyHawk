import numpy as np


def rms(x: np.ndarray):
    """
    :param x: 1-dim
    :return:
    """
    # 确认公式
    return np.linalg.norm(x) / np.sqrt(np.shape(x)[0])


matrix = np.array([[1,1,1],
          [2,2,2]])

index3 = list(np.ones(10).astype(bool))
index4 = [True, True, False, True, False]
print(index4)