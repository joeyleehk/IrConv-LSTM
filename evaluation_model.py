import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from data import hourly_bike_1h_1km_NYC

def maxminscaler_3d_reverse(tensor_3d, max_x, min_x):
    a = tensor_3d * ((max_x - min_x) + min_x)
    return a
def data_saving_2km(gts_in, predcitions_in):
    node_list, correlation_table = [], {}
    file_map = open('NYC/DTW_Similarity_Table.csv', 'r') # loading the cell id
    for line in file_map:
        line_element = line.strip().split(':')
        node_id = line_element[0]
        value_list = line_element[1].split(',')
        node_list.append(node_id)
        correlation_table[node_id] = value_list
    f_out = open('model_test_NYC.csv', 'w') # output the prediction results
    for index,line_no in enumerate(node_list):
        f_out.writelines([str(line_no), ':'])
        for day in range(len(gts_in)):
            f_out.writelines([str(predcitions_in[day][index][0]), ',', str(gts_in[day][index][0]), ';'])
        f_out.write('\n')


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    # TEST
    hourly = 24
    daily = 7
    weekly = 2
    model = torch.load('irregular_convolution_LSTM_42_1661428838.pkl')
    model.eval()
    batch_size_pred = 1
    start_date = '100100'
    end_date = '102523'
    # Create test data set
    dtest = hourly_bike_1h_1km_NYC(hourly=hourly, daily=daily, weekly=weekly,
                           start_date=start_date,
                           end_date=end_date)
    test_loader = DataLoader(dtest,
                             batch_size=batch_size_pred,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True  # CUDA only
                             )
    # Create list of n_stocks lists for storing predictions and GT
    predictions = []
    gts = []
    k = 0
    losses = []
    losses_MAE = []
    gts_array = np.zeros([1, 1, 1])
    predictions_array = np.zeros([1, 1, 1])
    for batch_idx, (data1, data2, data3, target) in enumerate(test_loader):
        loss_MAE_ = 0.
        data1 = Variable(data1).contiguous().cuda()
        data2 = Variable(data2).contiguous().cuda()
        data3 = Variable(data3).contiguous().cuda()
        target = Variable(target).double().cuda()
        if target.data.size()[0] == batch_size_pred:
            output = model(data1,data2,data3)[0][0].double()
            output = output.cpu().detach().numpy()
            k += 1
            gt = target[0][0].cpu().detach().numpy()
            predictions.append(output)
            gts.append(gt)
            gts_array = np.array(gts)
            predictions_array = np.array(predictions)
    gts_array_reverse = maxminscaler_3d_reverse(gts_array, dtest.get_max_min()[0],dtest.get_max_min()[1])
    predictions_array_reverse = maxminscaler_3d_reverse(predictions_array, dtest.get_max_min()[0], dtest.get_max_min()[1])
    data_saving_2km(gts_array_reverse, predictions_array_reverse)
