import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

class hourly_bike_1h_1km_NYC(Dataset):
    def __init__(self, closeness, period, trend, start_date='2012-01-01', end_date='2015-12-31'):

        self.closeness = closeness
        self.period = period
        self.trend = trend

        self.start_date = start_date
        if len(start_date) == 0:
            print("No start date was specified")
            return
        self.end_date = end_date
        if len(end_date) == 0:
            print("No end date was specified")
            return

        self.zeros_grids = np.load("NYC/nyc_raw_data.npy")

        index_to_date_dict, date_to_index_dict = date_to_index('2019-06-01 00:00:00', '2019-10-25 23:00:00')
        self.zeros_grids_new = {}
        for index in range(len(date_to_index_dict)):
            self.zeros_grids_new[index_to_date_dict[index]] = self.zeros_grids[index]

        self.raw_data_collection = []
        date_index = {}
        count = 0
        for index in range(date_to_index_dict[start_date] - 14 * 24, date_to_index_dict[self.end_date] + 1):
            self.raw_data_collection.append(self.zeros_grids_new[index_to_date_dict[index]])
            date_index[index_to_date_dict[index]] = count
            count += 1

        self.raw_data_collection = np.asarray(self.raw_data_collection)
        self.scaler_array = maxminscaler_3d(self.raw_data_collection)
        self.zeros_grids = self.scaler_array[0]
        self.closeness_array = []
        self.period_array = []
        self.trend_array = []

        self.target = []
        for index in range(date_index[self.start_date], date_index[self.end_date]):
            self.target.append(self.zeros_grids[index, :, 0])
            for c_index in sorted(range(1, self.closeness + 1), reverse=True):
                self.closeness_array.append([self.zeros_grids[index - c_index]])

            for p_index in sorted(range(1, self.period + 1), reverse=True):
                self.period_array.append([self.zeros_grids[index - p_index * 24]])

            for t_index in sorted(range(1, self.trend + 1), reverse=True):
                self.trend_array.append([self.zeros_grids[index - t_index * 168]])

        self.chunks_closeness = torch.FloatTensor(np.array(self.closeness_array)).squeeze(1).unfold(0, self.closeness, self.closeness).permute(0, 3, 1, 2)
        self.chunks_period = torch.FloatTensor(np.array(self.period_array)).squeeze(1).unfold(0, self.period, self.period).permute(0, 3, 1, 2)
        self.chunks_trend = torch.FloatTensor(np.array(self.trend_array)).squeeze(1).unfold(0, self.trend, self.trend).permute(0, 3, 1, 2)
        self.chunks_target = torch.FloatTensor(np.array(self.target)).unsqueeze(1).unsqueeze(3)

    def __getitem__(self, index):
        x1 = self.chunks_closeness[index, :, :, :9]
        x2 = self.chunks_period[index, :, :, :9]
        x3 = self.chunks_trend[index, :, :, :9]
        y = self.chunks_target[index, -1:, :, :]
        return x1, x2, x3, y

    def __len__(self):
        return self.chunks_target.size(0)

    def get_max_min(self):
        return self.scaler_array[1], self.scaler_array[2]

def date_to_index(start_time, end_time):
    dates_time = pd.date_range(start_time, end_time, freq='1h')
    date_hourly_interval = list(dates_time)
    date_to_index, index_to_date = {}, {}

    for index,item in enumerate(date_hourly_interval):
        key = str(item).split(' ')[0].split('-')[1] + str(item).split(' ')[0].split('-')[2] + str(item).split(' ')[1].split(':')[0]
        index_to_date[index] = key
        date_to_index[key] = index
    return index_to_date,date_to_index


def get_files(rootDir):
    list_dirs = os.walk(rootDir)
    file_container = {}
    for root, dirs, files in list_dirs:
        for f in files:
            file_container[f]=0
    return file_container

def get_files_list(rootDir):
    list_dirs = os.walk(rootDir)
    file_container = []
    for root, dirs, files in list_dirs:
        for f in files:
            file_container.append(f)
    return file_container

def maxminscaler_3d(tensor_3d, range = (0,1)):
    scaler_max = tensor_3d.max()
    scaler_min = tensor_3d.min()
    X_std = (tensor_3d - scaler_min)/(scaler_max - scaler_min)
    X_scaled = X_std * (range[1] - range[0]) + range[0]
    return X_scaled, scaler_max, scaler_min
