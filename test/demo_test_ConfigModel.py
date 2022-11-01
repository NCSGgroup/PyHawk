from src.SecDerivative.ConfigGravHar import ConfigGravHar
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    config_model = ConfigGravHar()
    fr = Frame(EOP(True).load())

    time = fr.setTime(505051205, TimeFormat.GPS_second)
    config_model = config_model.setTime(time)
    print(config_model.getTotalCS())


if __name__ == '__main__':
    demo1()

