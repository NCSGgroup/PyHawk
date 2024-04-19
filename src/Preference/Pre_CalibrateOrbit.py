import json
from src.Preference.EnumType import Mission


class CalibrateOrbitConfig:
    def __init__(self):

        self.MissionConfig = self.Mission().__dict__
        self.StepControlConfig = self.StepControl().__dict__
        self.ParallelControlConfig = self.ParallelControl().__dict__

    class Mission:
        def __init__(self):
            self.mission = Mission.GRACE_FO_RL04.name

    class StepControl:
        def __init__(self):
            self.isPreprocess = True
            self.isAdjustOrbit = True
            self.isAdjustKBRR = True
            self.isGetGravityDesignMat = True
            self.isGetNEQ = True
            self.isGetSH = True

    class ParallelControl:
        def __init__(self):
            self.AdjustOrbitProcess = 50
            self.AdjustKBRRProcess = 50
            self.GetGravityDesignMatProcess = 15
            self.GetNEQProcess = 50


def demo2():
    Obj1 = CalibrateOrbitConfig()
    tc_dict = Obj1.__dict__
    with open('../../setting/Calibrate/CalibrateOrbitConfig.json', 'w') as f:
        json.dump(tc_dict, f, indent=4)


if __name__ == '__main__':
    # res = ParseObjToJson(ForceModelConfig())
    # data = json.loads(res)
    # demo1()
    demo2()