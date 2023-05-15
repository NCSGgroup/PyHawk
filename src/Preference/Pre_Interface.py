from src.Preference.EnumType import Mission, Payload, SatID
import json


class InterfaceConfig:
    def __init__(self):
        self.mission = Mission.GRACE_RL02.name
        self.date_span = ['2008-01-01', '2008-01-31']
        self.arcNo = 0
        self.sat = {SatID.A.name: True,
                    SatID.B.name: True}
        self.date_delete = []

        self.include_payloads = {Payload.ACC.name: True,
                                 Payload.SCA.name: True,
                                 Payload.KBR.name: True,
                                 Payload.GNV.name: True,
                                 Payload.KinematicOrbit.name: False,
                                 Payload.LRI.name: False}

        self.is_orbit_GCRS = {Payload.GNV.name: False,
                              Payload.KinematicOrbit.name: True}

        self.sr_payload = {Payload.ACC.name: 1,
                           Payload.SCA.name: 5,
                           Payload.KBR.name: 5,
                           Payload.GNV.name: 5,
                           Payload.KinematicOrbit.name: 1,
                           Payload.LRI.name: 5}
        self.sr_target = 5

        self.gapfix_linear_mis_max = 5
        self.gapfix_quadra_mis_max = 21
        self.gapfix_cubic_mis_max = 80
        self.gapfix_cubic_int = 150

        self.arc_length = 3
        self.min_arcLen = 2

        self.PathOfFilesConfig = self.PathOfFiles(self).__dict__

    class PathOfFiles:
        def __init__(self, interface):
            # ======================= For the data interface===============================
            self.payload_ACC = "../data/" + interface.mission
            self.payload_SCA = "../data/" + interface.mission
            self.payload_KBR = "../data/" + interface.mission
            self.payload_GNV = "../data/" + interface.mission
            self.payload_Kinematic = "../data/KinematicOrbit"
            self.payload_LRI = "../data" + interface.mission

            # ======================== For the data preprocess===============================
            self.report_gap = "../report/gap"
            self.report_arc = "../report/arc"
            self.temp_raw_data = "../temp/RawData"


def class_to_json():
    Obj1 = InterfaceConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/InterfaceConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()
