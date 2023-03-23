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
from src.Auxilary.GeoMathKit import GeoMathKit
from src.Auxilary.SortCS import SortCS
import src.Preference.Pre_Parameterization as Pre_Parameterization
from src.Interface.LoadSH import LoadSH, LoadGif48
from src.Preference.Pre_ForceModel import RefGravModelConfig, PathOfFiles
from src.Preference.EnumType import StaticGravModel


def demo1(start, end):
    for i in tqdm(range(start, end), desc='arcNo.'):
        neq = GravityNEQperArc(i)
        neq.configure().get_neq()


def demo():

    res_dir = pathlib.Path(Pre_NEQ.PathOfFiles.NormalEqTemp).joinpath\
        (Pre_Interface.date_span[0] + '_' + Pre_Interface.date_span[1])
    # res_filename = res_dir.joinpath(str(0) + '.hdf5')
    # h5 = h5py.File(res_filename, 'r')
    NMatrix = np.shape((8277, 8277))
    LMatrix = np.shape((8277, 1))
    for i in tqdm(range(0, 108), desc='Solve normal equations: '):
        res_filename = res_dir.joinpath(str(i) + '.hdf5')
        h5 = h5py.File(res_filename, 'r')
        NMatrix = NMatrix + (h5['Orbit_N_A'][:] + h5['Orbit_N_B'][:]) * 1E-12 + h5['RangeRate_N'][:]
        LMatrix = LMatrix + (h5['Orbit_l_A'][:] + h5['Orbit_l_B'][:]) * 1E-12 + h5['RangeRate_l'][:].reshape((8277, 1))
        h5.close()

    x = np.linalg.lstsq(NMatrix, LMatrix[:, 0], rcond=None)[0]
    print(x)
    np.savez('CS.npz', cs=x)


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



def cs():

    c = np.load('CS.npz')['c1d']
    s = np.load('CS.npz')['s1d']

    C = GeoMathKit.CS_1dTo2d(c)
    S = GeoMathKit.CS_1dTo2d(s)

    np.savez('CS2d.npz', c=C, s=S)
    pass


if __name__ == '__main__':
    # demo1(0, 108)
    # p1 = multiprocessing.Process(target=demo1, args=(0, 18))
    # p2 = multiprocessing.Process(target=demo1, args=(18, 36))
    # p3 = multiprocessing.Process(target=demo1, args=(36, 54))
    # p4 = multiprocessing.Process(target=demo1, args=(54, 72))
    # p5 = multiprocessing.Process(target=demo1, args=(72, 90))
    # p6 = multiprocessing.Process(target=demo1, args=(90, 108))
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
    # demo()
    # draw()

    cs()