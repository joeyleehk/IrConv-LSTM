import torch
import torch.nn as nn
from torch.autograd import Variable

class irregular_convolution(nn.Module):
    def __init__(self,in_channel, out_channel, kernel_length=9, batch_size=150, num_node=125, bias=True):
        """
        :param in_channel: int, The No. of channel for input matrix
        :param out_channel: int, The No. of channel for output matrix
        :param kernel_length: int, The size of irregular convolution kernel size (default: 9)
        :param batch_size: int, The size of batch (default: 150)
        :param num_node: int, The number of nodes (cells involved in this study)(default: 125 cells in New York)
        :param bias: bool, add bias in linear layer or not
        This is the core for irregular convolution: covert the convolution operation in the discrete domain
        to linear layers(weighted addition).
        """
        super(irregular_convolution, self).__init__()
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.kernel_length = kernel_length
        self.kernel_fun_bias = nn.Linear(self.in_channel * self.kernel_length, self.out_channel).cuda()
        self.kernel_fun = nn.Linear(self.in_channel * self.kernel_length, self.out_channel, bias=False).cuda()
        self.bias = bias

    def forward(self, x_input):
        reshape_size = self.in_channel * self.kernel_length
        x_size = x_input.size()
        x_input = x_input.permute(0,1,3,2).reshape([x_size[0], x_size[2], reshape_size])
        if self.bias:
            output = self.kernel_fun_bias(x_input).permute(0,2,1).unsqueeze(-1)
        else:
            output = self.kernel_fun(x_input).permute(0,2,1).unsqueeze(-1)
        return output


class Extraction_spatial_features(nn.Module):
    def __init__(self, kernel_size=9, batch_size=150,seq_len=24,total_nodes=125):
        """
        This module is to capture spatial dependency of bicycle usage for each interval in a sequence using irregular
        convolution operations.

        The built-in function, self.prepare_data(), is to select the semantic neighbors for each central cell using
        a Pytorch function torch.masked_select().

        The function, reconstruction_file(), is to read the look-up table for mapping relationship between central cells
        and their corresponding semantic neighbors. The return of this function is the input for torch.masked_select().
        :param kernel_size: int, The size of irregular convolution kernel size (default: 9)
        :param batch_size: int, The size of batch (default: 150)
        :param seq_len: int, The length of sequence (default:24)
        :param total_nodes:The number of nodes (cells involved in this study)(default: 125 cells in New York)
        """
        super(Extraction_spatial_features, self).__init__()
        self.mask = reconstruction_file(kernel_size=kernel_size)
        self.batch_size=batch_size
        self.seq_len=seq_len
        self.total_nodes=total_nodes
        self.kernel_size = kernel_size
        self.relu = nn.ReLU(inplace=True)
        self.first_no_layer = 32
        self.second_no_layer = 16
        self.irregular_layer1 = irregular_convolution(1, self.first_no_layer, kernel_length=self.kernel_size,
                                                      batch_size=self.batch_size)
        self.irregular_layer2 = irregular_convolution(self.first_no_layer, self.second_no_layer, kernel_length=self.kernel_size, batch_size=self.batch_size)
        self.batchnormal = nn.BatchNorm2d(self.second_no_layer)
        self.temporal_out = torch.ones([self.total_nodes,1]).cuda()
        self.reduce_dimension = irregular_convolution(self.second_no_layer, 1, kernel_length=self.kernel_size,
                                                      batch_size=self.batch_size) # reduce the dimension of the high-level features to 1
        self.mask = torch.tensor(self.mask, dtype=torch.bool).cuda()

    def prepare_data(self, input_x):
        """
        This function is to prepare the tensor based on the map between central cells and their corresponding neighbors.
        :param input_x: A tensor needs to re-structure following the look-up table.
        :return:
        """
        shape = input_x.size()
        temporal_out = torch.matmul(self.temporal_out,input_x)
        temporal_out = torch.masked_select(temporal_out, self.mask).reshape(
            [shape[0], shape[1], shape[3], self.kernel_size]).cuda()
        return temporal_out

    def forward(self, input):
        """
        This forward function captures spatial dependency of bicycle usage for each interval in a sequence.
        :param input: The sequence of historical bicycle usage.
        :return: A tensor with captured spatial dependency, which will be input into the LSTM module.
        """
        output = torch.empty(input.shape[0], self.seq_len, self.total_nodes).cuda()
        for index in range(self.seq_len):
            x_input = torch.unsqueeze(input[:,index,:,:], 1)
            cnn_output = self.relu(self.irregular_layer1(x_input)).permute(0,1,3,2)
            cnn_input = self.prepare_data(input_x=cnn_output)
            cnn_output = self.relu(self.batchnormal(self.irregular_layer2(cnn_input))).permute(0,1,3,2)
            cnn_input = self.prepare_data(input_x=cnn_output)
            cnn_output = self.reduce_dimension(cnn_input)
            cnn_output = self.relu(cnn_output)
            cnn_output = cnn_output.squeeze(-1).squeeze(1)
            output[:, index, :] = cnn_output
        return output

class Convolution_LSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layer=2, batch_size=50,):
        """
        This is a LSTM module to capture the temporal dependency of bicycle usage from a historical sequence.
        :param input_channels: int, The No. of features in the input tensor.
        :param hidden_channels: int, The No. of features in the hidden state.
        :param num_layer: int, Number of recurrent layers.
        :param batch_size: int, Number of batch size.
        """
        super(Convolution_LSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_channels, hidden_size=self.hidden_channels,
                            num_layers=self.num_layer, batch_first=True)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.hidden2out_1 = nn.Linear(self.hidden_channels, 64)
        self.hidden2out_2 = nn.Linear(64, self.input_channels)
    def initalize_parameters(self,batch_size):
        return (Variable(torch.zeros(self.num_layer, batch_size, self.hidden_channels).cuda(), requires_grad=True),
                Variable(torch.zeros(self.num_layer, batch_size, self.hidden_channels).cuda(), requires_grad=True),
                )
    def forward(self, input):
        h0, c0 = self.initalize_parameters(input.shape[0])
        outputs, (ht, ct) = self.lstm(input, (h0, c0))
        output = ht[-1]
        output = self.hidden2out_1(output)
        output = self.hidden2out_2(output)
        output = self.tanh(output)
        output = output.unsqueeze(1).unsqueeze(-1)
        return output

class Irregular_Convolution_LSTM(nn.Module):
    def __init__(self, input_size_closeness, input_size_period, input_size_trend, kernel_size, bsize=50):
        """
        This class assembles the module capturing spatial dependency by irregular convolution and the module capturing
        temporal dependency by LSTM module. As mentioned in the paper, there are three historical periods to capture
        spatial-temporal features, respectively. The outputs of the three periods are fused by weighted addition.
        Please refer to the pre-print paper in Arxiv: https://arxiv.org/abs/2202.04376
        :param input_size_closeness: int, The size of Closeness.
        :param input_size_period: int, The size of Period.
        :param input_size_trend: int, The size of Trend.
        :param kernel_size: int, The size of convolution kernel.
        :param bsize: The size of batch.
        """
        super(Irregular_Convolution_LSTM, self).__init__()
        self.kernel_size = kernel_size
        self.bsize = bsize
        self.input_size_closeness = input_size_closeness
        self.input_size_period = input_size_period
        self.input_size_trend = input_size_trend

        self.conv_module_closeness = Extraction_spatial_features(kernel_size=self.kernel_size,batch_size=self.bsize,
                                                             seq_len=self.input_size_closeness,total_nodes=125).cuda()
        self.convlstm_closeness = Convolution_LSTM(input_channels=125, hidden_channels=125,
                                                num_layer=2, batch_size=self.bsize,).cuda()

        self.conv_module_period = Extraction_spatial_features(kernel_size=self.kernel_size, batch_size=self.bsize,
                                                             seq_len=self.input_size_period, total_nodes=125).cuda()
        self.convlstm_period = Convolution_LSTM(input_channels=125, hidden_channels=125,
                                                num_layer=2, batch_size=self.bsize,).cuda()

        self.conv_module_trend = Extraction_spatial_features(kernel_size=self.kernel_size, batch_size=self.bsize,
                                                             seq_len=self.input_size_trend, total_nodes=125).cuda()
        self.convlstm_trend = Convolution_LSTM(input_channels=125, hidden_channels=125,
                                                num_layer=2, batch_size=self.bsize).cuda()

        self.tanh = nn.Tanh()
        self.W_closeness, self.W_period, self.W_trend = self.init_hidden([125, 1])
        self.W_closeness = torch.nn.init.xavier_normal_(self.W_closeness)
        self.W_period = torch.nn.init.xavier_normal_(self.W_period)
        self.W_trend = torch.nn.init.xavier_normal_(self.W_trend)


    def forward(self, x_closeness, x_period, x_trend):
        output_closeness = self.conv_module_closeness(x_closeness)
        output_closeness = self.convlstm_closeness(output_closeness)

        output_period = self.conv_module_period(x_period)
        output_period = self.convlstm_period(output_period)

        output_trend = self.conv_module_trend(x_trend)
        output_trend = self.convlstm_trend(output_trend)

        output = self.W_closeness * output_closeness + self.W_period * output_period + self.W_trend * output_trend
        output = self.tanh(output)
        return output

    def init_hidden(self,shape):
        return (torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
                torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
                torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
                )

def reconstruction_file(kernel_size):
    """
    This function returns the look-up table for central cells to check their corresponding semantic neighbors.
    In this instance, we use the metric calculated by DTW algorithm in New York.
    The look-up table is named as DTW_Similarity_Table.csv in NYC folder.
    :param kernel_size: int, the size of convolution kernel.
    :return: look-up table for mapping central cells with their corresponding semantic neighbors.
    """
    node_list, correlation_table = [], {}
    file_map = open('NYC/DTW_Similarity_Table.csv', 'r')
    for line in file_map:
        line_element = line.strip().split(':')
        node_id = line_element[0]
        value_list = line_element[1].split(',')
        node_list.append(node_id)
        correlation_table[node_id] = value_list
    node_index_dict = {}
    for index, item in enumerate(node_list):
        node_index_dict[item] = index
    correlation_table_index = {}
    for key in correlation_table:
        for index, item in enumerate(correlation_table[key]):
            if index < kernel_size:
                if node_index_dict[key] not in correlation_table_index:
                    correlation_table_index[node_index_dict[key]] = [node_index_dict[item]]
                else:
                    correlation_table_index[node_index_dict[key]].append(node_index_dict[item])

    zero_matrix = []
    for key in correlation_table_index:
        temp_zero = []
        for i in range(125):
            if i in correlation_table_index[key]:
                temp_zero.append(1)
            else:
                temp_zero.append(0)
        zero_matrix.append(temp_zero)

    return zero_matrix

