import numpy as np
import os

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
test3 = [key for key, value in test2.items() if value]
print(test3)
# print(test1)
# print(dict(test1.items() | test2.items()))
