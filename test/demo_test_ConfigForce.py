from src.SecDerivative.ConfigForce import ConfigForce
from src.Frame.Frame import Frame
from src.Frame.EOP import EOP
from src.Preference.EnumType import TimeFormat


def demo1():
    config_model = ConfigForce()
    fr = Frame(EOP(True).load())

    time = fr.setTime(505051205, TimeFormat.GPS_second)
    # config_model = config_model.setTime(time)
    print(config_model.__ForceType)


if __name__ == '__main__':
    demo1()

