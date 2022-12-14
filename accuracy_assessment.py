import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import numpy as np
import scipy.signal

from scipy import stats
    key_str = ['327', '267', '268', '269']
    cal_result_1 = calculate_plot_image(file_name_input1)
    cal_result_2 = calculate_plot_image(file_name_input2)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Each grid all hours: ConvLSTM')
    plt.boxplot([cal_result_1[0], cal_result_1[1]],showfliers=False)
    plt.ylim(-0.5, 2)
    plt.xticks([1, 2,], ['MAPE', 'MAE', ])

    plt.subplot(122)
    plt.title('Each grid all hours: STRN')
    plt.boxplot([cal_result_2[0], cal_result_2[1]], showfliers=False)
    plt.ylim(-0.5, 2)
    plt.xticks([1, 2, ], ['MAPE', 'MAE', ])
    plt.show()

def calculate_accuracy(file_name):
    f_in = open(file_name, 'r')
    grid_dict = {}
    grid_dict_value = {}
    grid_accuracy_hourly = {}

    for line in f_in:
        line_element = line.strip().split(':')
        grid_dict[line_element[0]] = line_element[1].split(';')
    global_sum_mape, global_sum_mae, global_sum_mse, global_sum_rmse = 0., 0., 0., 0.
    global_num_mape, global_num_mae, global_num_mse, global_num_rmse= 0, 0, 0, 0
  
    for key in grid_dict:
        num_mape, num_mae, num_mse = 0, 0, 0
        sum_mape, sum_mae, sum_mse = 0., 0., 0.
        for item in grid_dict[key]:
            if item != '':
                pred = float(item.split(',')[0])
                gt = float(item.split(',')[1])
                if key not in grid_dict_value:
                    grid_dict_value[key] = [[pred, gt]]
                else:
                    grid_dict_value[key].append([pred, gt])
                if gt != 0: # select the cells with bicycle usage to calculate the accuracy
                    mape = abs(pred - gt) / gt
                    global_sum_mape += mape
                    global_num_mape += 1

                    mae = abs(pred - gt)
                    global_sum_mae += mae
                    global_num_mae += 1

                    rmse = (pred - gt) ** 2
                    global_sum_rmse += rmse
                    global_num_rmse += 1
    global_rmse = round((global_sum_rmse /global_num_rmse)**0.5,6)
    global_mape =  round(global_sum_mape / global_num_mape,6)
    global_mae =  round(global_sum_mae / global_num_mae,6)
    print(file_name_input,':','global_mape:', global_mape, 'global_mae', global_mae, 'global_rmse', global_rmse)



if __name__ == '__main__':
    calculate_accuracy('model_test_NYC.csv') # prediction accuracy calculation 