import json


class CalibrateOrbitConfig:
    def __init__(self):

        self.isPreprocess = True
        self.isAdjustOrbit = True
        self.isAdjustKBRR = True
        self.isGetGravityDesignMat = True
        self.isGetNEQ = True
        self.isGetSH = True

        self.AdjustOrbitNodes = 50
        self.AdjustKBRRNodes = 50
        self.GetGravityDesignMatNodes = 15
        self.GetNEQNodes = 50


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