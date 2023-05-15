import json


class ConstantsConfig:
    def __init__(self):
        # 儒略日
        self.J2000_MJD = 51544.5
        self.Diff_Tai_GPS = 19
        self.Diff_Tai_TT = 32.184
        self.SecForOneDay = 86400
        self.DJMJD0 = 2400000.5

        self.num_tolerance = 1e-5


def class_to_json():
    Obj1 = ConstantsConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/ConstantsConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

