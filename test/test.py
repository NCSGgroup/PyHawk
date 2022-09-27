import numpy as np
import os
import src.Preference.EnumType as EnumType
import src.Preference.Pre_ForceModel as Pre_ForceModel

test1 = {'Sun': True,
         'Moon': True}

test2 = {
    'Mercury': True,
    'Venus': True,
    'Earth': False,
    'Mars': True,
    'Jupiter': True,
    'Saturn': True,
    'Uranus': True,
    'Neptune': True,
    'Pluto': True
}

lisdt = []
# test3 = [key for key, value in test2.items() if value]
print(len(lisdt))
# print(Pre_ForceModel.ThreeBodyConfig.include_planets)
# print(test2)
# print(test1)
# print(dict(test1.items() | test2.items()))
