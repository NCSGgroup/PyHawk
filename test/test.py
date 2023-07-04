import requests
import jsoneditor
import json

data2 = json.load(open('../setting/Calibrate/AccelerometerConfig_A.json', 'r'))

editor = jsoneditor.editjson(data2)

print(editor.data)
