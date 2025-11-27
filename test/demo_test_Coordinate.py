from src.Frame.EOP import EOP
from src.Frame.Frame import Frame
import src.Frame.SOFA as sf
import time as Ti
import numpy as np
from src.Preference.EnumType import TimeFormat
from src.Preference.Pre_Frame import FrameConfig
import os
import json


def demo():
    frame_config = FrameConfig()
    frame = json.load(open('../setting/Calibrate/FrameConfig.json', 'r'))
    frame_config.__dict__ = frame

    eop = EOP().configure(frameConfig=frame_config).load()
    fr = Frame(eop).configure(frameConfig=frame_config)

    data = []
    with open("D:\张进\GFOC_RL02_20001.txt", "r") as f, open("D:\张进\GFOC_RL02_20001-GCRS.txt", "w") as fout:
        for line in f:
            parts = line.split()
            time = float(parts[4])
            x = float(parts[5])
            y = float(parts[6])
            z = float(parts[7])

            tf = fr.setTime(time, TimeFormat.GPS_second)

            pos_ITRS = np.array([x, y, z])
    # vel_GCRS = np.array([100., 200., 300.])

    # pos_ITRS = fr.PosGCRS2ITRS(pos_GCRS, tf.getRotationMatrix)
            pos_GCRS = fr.PosITRS2GCRS(pos_ITRS, tf.getRotationMatrix)
            fout.write(f"{pos_GCRS[0]:.6f}  {pos_GCRS[1]:.6f} {pos_GCRS[2]:.6f}\n")
            print(1)
    # vel_ITRS = fr.VelGCRS2ITRS(pos_GCRS, vel_GCRS, tf.getRotationMatrix)
    # vel_GCRS = fr.VelITRS2GCRS(pos_ITRS, vel_ITRS, tf.getRotationMatrix)
    print(pos_ITRS)
    # print(vel_ITRS)

    return True


if __name__ == '__main__':
    demo()