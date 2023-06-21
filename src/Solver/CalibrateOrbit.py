from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Interface.L1b import GRACE_OLD_RL02
from src.Interface.KinematicOrbit import KinematicOrbit
from src.Interface.GapFix import GapFix
from src.Interface.ArcOperation import ArcSelect


"""step 1: Read raw data"""
fr = Frame(EOP().load())
s = GRACE_OLD_RL02().configure().setDate()
s.read_double_sat()
"""step 2: Convert orbit from ITRS frame into GCRS frame"""
kine = KinematicOrbit(L1b=s).configure()
# kine.read_double_sat()
kine.GNVandKiOrbitItrs2Gcrs(fr)
"""step 3: Gap fix"""
gf = GapFix(s).configure()
gf.fix_all().makeReport()
arc = ArcSelect(gf).configure()
arc.unpackArcs().makeReport()