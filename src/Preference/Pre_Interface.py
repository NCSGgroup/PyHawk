"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/8/8
@Description:
"""

from src.Preference.EnumType import Mission, Payload, SatID
import pathlib as pathlib

mission = Mission.GRACE_RL02
date_span = ['2008-01-01', '2008-01-31']
arcNo = 1
sat = {SatID.A: True,
       SatID.B: True}
date_delete = []

include_payloads = {Payload.ACC: True,
                    Payload.SCA: True,
                    Payload.KBR: True,
                    Payload.GNV: True,
                    Payload.KinematicOrbit: False,
                    Payload.LRI: False}

is_orbit_GCRS = {Payload.GNV: False,
                 Payload.KinematicOrbit: True}

sr_payload = {Payload.ACC: 1,
              Payload.SCA: 5,
              Payload.KBR: 5,
              Payload.GNV: 5,
              Payload.KinematicOrbit: 1,
              Payload.LRI: 5}

sr_target = 5

gapfix_linear_mis_max = 5
gapfix_quadra_mis_max = 21
gapfix_cubic_mis_max = 80
gapfix_cubic_int = 150

arc_length = 3
min_arcLen = 3


class PathOfFiles:
    root_dir = pathlib.Path(__file__).parent.parent.parent

    # ======================= For the data interface===============================
    payload_ACC = root_dir.joinpath('data', mission.name)
    payload_SCA = root_dir.joinpath('data', mission.name)
    payload_KBR = root_dir.joinpath('data', mission.name)
    payload_GNV = root_dir.joinpath('data', mission.name)
    payload_Kinematic = root_dir.joinpath('data', 'KinematicOrbit')
    payload_LRI = root_dir.joinpath('data', mission.name)

    # ======================== For the data preprocess===============================
    report_gap = root_dir.joinpath('report', 'gap')
    report_arc = root_dir.joinpath('report', 'arc')
    temp_raw_data = root_dir.joinpath('temp', 'RawData')
