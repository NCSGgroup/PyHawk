from src.Preference.EnumType import CoordinateTransform
import json


class FrameConfig:
    def __init__(self):
        self.isCorrection_EOP = True
        self.isRadian_EOP = True

        self.CoorTrans_method = CoordinateTransform.CIOxy.name
        self.OmegaOption = 1
        self.tdbOption = 1

        self.EphemeridesConfig = self.Ephemerides().__dict__
        self.PathOfFilesConfig = self.PathOfFiles().__dict__

    class Ephemerides:
        def __init__(self):
            self.isMeter = True
            self.include_planets = {
                'Sun': True,
                'Moon': True,
                'Mercury': False,
                'Venus': False,
                'Earth': False,
                'Mars': False,
                'Jupiter': False,
                'Saturn': False,
                'Uranus': False,
                'Neptune': False,
                'Pluto': False}

    class PathOfFiles:
        def __init__(self):
            self.EOP = "../data/eop/eopc04_14_IAU2000.62-now"
            self.EOPgrav = "../data/eop/EOPgrav.txt"
            self.EOPocean = "../data/eop/EOPocean.txt"


def class_to_json():
    Obj1 = FrameConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/FrameConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    class_to_json()

