import pathlib as pathlib


cali_A = {'Scale': [False, False, False,
                    False, False, False,
                    False, False, False],
          'Bias_constant': [True, True, True],
          'Bias_trend': [False, False, False],
          'Bias_quadratic': [False, False, False]
          }

cali_B = {'Scale': [False, False, False,
                    False, False, False,
                    False, False, False],
          'Bias_constant': [True, True, True],
          'Bias_trend': [False, False, False],
          'Bias_quadratic': [False, False, False]
          }

cali_value_A = {'Scale': [0.9595, 0, 0,
                          0, 0.9797, 0,
                          0, 0, 0.9485],
              'Bias_constant': [0, 0, 0],
              'Bias_trend': [0, 0, 0],
              'Bias_quadratic': [0, 0, 0]
                }

cali_value_B = {'Scale': [0.9595, 0, 0,
                          0, 0.9797, 0,
                          0, 0, 0.9485],
              'Bias_constant': [0, 0, 0],
              'Bias_trend': [0, 0, 0],
              'Bias_quadratic': [0, 0, 0]
                }



root_dir = pathlib.Path(__file__).parent.parent.parent
temp_non_conservative_data = root_dir.joinpath('temp', 'NonConservativeForce')