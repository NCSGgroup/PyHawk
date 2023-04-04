import sys
sys.path.append('../')
import multiprocessing
from src.Solver.NEQ import GravityNEQperArc
from src.Preference.EnumType import SatID
import src.Preference.Pre_NEQ as Pre_NEQ
import src.Preference.Pre_Interface as Pre_Interface
import pathlib
import h5py
import numpy as np
from tqdm import tqdm
from src.Auxilary.SortCS import SortCS
from src.Auxilary.GeoMathKit import GeoMathKit
import src.Preference.Pre_Parameterization as Pre_Parameterization
from src.Interface.LoadSH import LoadSH, LoadGif48
from src.Preference.Pre_ForceModel import RefGravModelConfig, PathOfFiles


def demo1(start, end):
    for i in range(start, end):
        neq = GravityNEQperArc(i)
        neq.configure().get_neq()


def demo():

    res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath\
        (Pre_Interface.date_span[0] + '_' + Pre_Interface.date_span[1])
    # res_filename = res_dir.joinpath(str(0) + '.hdf5')
    # h5 = h5py.File(res_filename, 'r')
    NMatrix = np.zeros((3717, 3717))
    LMatrix = np.zeros((3717, 1))
    for i in tqdm(range(0, 243), desc='Solve normal equations: '):
        res_filename = res_dir.joinpath(str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        NMatrix = NMatrix + (h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:]) * 1E-12 + h5['RangeRate_N'][:]
        LMatrix = LMatrix + (h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:]) * 1E-12 + h5['RangeRate_l'][:].reshape((3717, 1))

    x = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]
    print(x)
    np.savez('CS-3.24.npz', cs=x)


def draw():
    data = np.load('CS.npz')['cs']
    c = np.zeros((int(91), int(91)))
    s = np.zeros((int(91), int(91)))
    listC = []
    listS = []
    index = 0
    for m in range(0, 2):
        for n in range(2, 91):
            c[n][m] = data[index]
            index += 1

    for i in range(2, 91):
        s[i][1] = data[index]
        index += 1

    for m in range(2, 91):
        len = 91 - m
        for n in range(m, 91):
            c[n][m] = data[index]
            s[n][m] = data[index + len]
            index += 1
        index += len

    for m in range(0, 91):
        for l in range(m, 91):
            listC.append(c[l][m])
            listS.append(s[l][m])
    print(listC)
    print(listS)
    np.save('CS.npz', c=c, s=s)
    # print(np.shape(data))


def csReshape():
    data = np.load('CS-3.24.npz')['cs']
    sort = SortCS(Pre_Parameterization.StokesCoefficients.SortMethod,
                  Pre_Parameterization.StokesCoefficients.MaxDegree,
                  Pre_Parameterization.StokesCoefficients.MinDegree)
    c, s = sort.invert(data)

    static_c,static_s = LoadGif48().load(PathOfFiles.EIGEN6_C4).getCS(90)

    # C, S = c + static_c, s + static_s

    C, S = GeoMathKit.CS_1dTo2d(c), GeoMathKit.CS_1dTo2d(s)

    print(C)
    np.savez('CS2d-3.24.npz', c=C, s=S)
    pass


if __name__ == '__main__':
    # demo1(36,37)
    # p1 = multiprocessing.Process(target=demo1, args=(0, 41))
    # p2 = multiprocessing.Process(target=demo1, args=(41, 82))
    # p3 = multiprocessing.Process(target=demo1, args=(82, 123))
    # p4 = multiprocessing.Process(target=demo1, args=(123, 164))
    # p5 = multiprocessing.Process(target=demo1, args=(164, 205))
    # p6 = multiprocessing.Process(target=demo1, args=(205, 243))
    #
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    # benchmark()
    # draw()
    csReshape()