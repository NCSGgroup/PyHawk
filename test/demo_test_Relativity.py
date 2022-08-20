from src.ForceModel.Relativity import Relativity
import numpy as np


def demo1():

    re = Relativity()
    re.setConstant().setKind().setSatState(pos=np.array([100000, 100000, 200000]) + 6378136.3e0, vel=np.array([3000, 5000, 6000]))

    print(re.getAcceleration())


if __name__ == '__main__':
    demo1()