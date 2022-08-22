from src.ForceModel.BaseForceCS import BaseForceCS
from src.ForceModel.LoadSH import LoadSH


class NonTide(BaseForceCS, LoadSH):

    def __init__(self):
        super(NonTide, self).__init__()
        LoadSH.__init__(self)



