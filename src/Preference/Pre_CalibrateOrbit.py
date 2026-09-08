import json
from src.Preference.EnumType import Mission


class CalibrateOrbitConfig:
    def __init__(self):

        self.MissionConfig = self.Mission().__dict__
        self.StepControlConfig = self.StepControl().__dict__
        self.ParallelControlConfig = self.ParallelControl().__dict__
        self.GLSIterationConfig = self.GLSIteration().__dict__
        self.LGDConfig = self.LGD().__dict__
        self.ExperimentAuditConfig = self.ExperimentAudit().__dict__

    class Mission:
        def __init__(self):
            self.mission = Mission.GRACE_FO_RL04.name

    class StepControl:
        def __init__(self):
            self.isPreprocess = True
            self.isAdjustOrbit = True
            self.isAdjustKBRR = True
            self.isGetLGD = True
            self.isGetGravityDesignMat = True
            self.isGetNEQ = True
            self.isGetSH = True
            self.isGLS = True

    class ParallelControl:
        def __init__(self):
            self.AdjustOrbitProcess = 50
            self.AdjustKBRRProcess = 50
            self.GetGravityDesignMatProcess = 15
            self.GetNEQProcess = 50

    class GLSIteration:
        def __init__(self):
            self.MaxIteration = 5
            self.MinIteration = 2
            self.RelativeTolerance = 1e-3
            self.AbsoluteTolerance = 1e-15
            self.RandomSeed = 20240724
            self.OrbitCovarianceMode = 'FixedDiagonal'
            self.OrbitFixedDiagonal = 0.004
            # Shared state-vector HDF5 files are read-only inputs for child
            # covariance cases.  Production keeps the historical write-on
            # behavior; isolated child cases may disable it to avoid file
            # locks when several cases run concurrently.
            self.WriteOrbitArcQuality = True

    class LGD:
        def __init__(self):
            self.Method = "RangeRateDerivative"
            self.ReferenceKBRAPayload = "KBR"
            self.ReferenceKBRAAllowVelocityFallback = False
            self.FLowHz = 1.0e-3
            self.FHighHz = None
            self.FilterOrder = 6
            self.GapThreshold = 20.0
            self.MinSegmentLength = 300
            self.KBRRPreprocessEnable = True
            self.KBRRDespikeEnable = True
            self.KBRRDespikeWindow = 121
            self.KBRRDespikeSigma = 6.0
            self.KBRRRemoveLowFrequency = True
            self.KBRRTrendDegree = 2
            self.KBRRCPRPeriod = 5400.0
            self.KBRRCPRHarmonics = 2
            self.SegmentEdgeTrimSeconds = 60.0

    class ExperimentAudit:
        """Optional, write-only audit outputs for controlled experiments.

        The defaults are deliberately disabled so the production numerical
        path and its storage footprint remain unchanged unless a case opts
        in explicitly.
        """
        def __init__(self):
            self.Enabled = False
            self.OutputDirectory = None
            self.SaveCovarianceStages = False
            self.SaveObservationNormalEquations = False
            self.SaveQRDiagnostics = False
            self.SaveSolutionDiagnostics = False
            self.Compression = "gzip"

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
