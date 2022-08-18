"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/8/8
@Description:
"""

import pathlib as pathlib
from src.Preference.EnumType import CoordinateTransform


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the EOP===============================
    EOP = root_dir.joinpath('data', 'eop', 'eopc04.62-now.txt')
    EOPgrav = root_dir.joinpath('data', 'eop', 'EOPgrav.txt')
    EOPocean = root_dir.joinpath('data', 'eop', 'EOPocean.txt')


isCorrection_EOP = True
isRadian_EOP = True

CoorTrans_method = CoordinateTransform.CIOxy
OmegaOption = 1
tdbOption = 1


