import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math
from model.irregular_convolution_LSTM import Irregular_Convolution_LSTM as irconvlstm
from data import hourly_bike_1h_1km_NYC
import time

def train_model(size_of_kernel):
    """
    This function is to train the IrConv+LSTM model, taking NEW York dataset as an example.
    Please refer to the pre-print paper in Arxiv: https://arxiv.org/abs/2202.04376
    :param
    size_of_kernel: The size of irregular convolution kernel size;
    start_date & end_date: The starting and ending date for training model;
    valid_start_date & valid_end_date: The starting and ending date for validating model;
    closeness_size & period_size & trend_size: Three historical periods for training model.
    """

    use_cuda = torch.cuda.is_available()
    # Keep track of loss in tensorboard
    writer = SummaryWriter()
    # Parameters
    learning_rate = 0.00028
    batch_size = 50
    max_epochs = 100
    kernel_size = size_of_kernel
    start_date = '061500'
    end_date = '093023'
    closeness_size = 24
    period_size = 7
    trend_size = 2
    torch.cuda.manual_seed(50)
    # Training data loading
    dset = hourly_bike_1h_1km_NYC(closeness=closeness_size, period=period_size, trend=trend_size, start_date=start_date, end_date=end_date)
    train_loader = DataLoader(dset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True  # CUDA only
                              )
    print('Training data loading complete.')
    # Validating data loading
    valid_start_date = '100900'
    valid_end_date = '101523'
    dset_vaild = hourly_bike_1h_1km_NYC(closeness=closeness_size, period=period_size, trend=trend_size, start_date=valid_start_date,
                                    end_date=valid_end_date)
    vaild_loader = DataLoader(dset_vaild,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True  # CUDA only
                              )
    print('Validating data loading complete.')
    # Model Definition + Optimizer + Scheduler
    model = irconvlstm(input_size_closeness=closeness_size, input_size_period=period_size, input_size_trend=trend_size,
                       kernel_size=kernel_size, bsize=batch_size).cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

    # Loss function
    criterion = nn.MSELoss(size_average=True).cuda()
    # Store successive losses
    losses = []
    losses_valid = []
    losses_epochs_valid = []
    # min_delta & patience are the parameters for early stop training model
    min_delta = 0.0000005
    patience = 10
    # Training model
    for i in range(max_epochs):
        loss_ = 0.
        # Store current predictions
        valid_dataloader_iter = iter(vaild_loader)
        losses_epoch_valid = []
        # Go through training data set
        for batch_idx, (data1, data2, data3, target) in enumerate(train_loader):
            data1 = Variable(data1).contiguous().cuda()
            data2 = Variable(data2).contiguous().cuda()
            data3 = Variable(data3).contiguous().cuda()
            target = Variable(target).double().cuda()
            if target.data.size()[0] == batch_size:
                # Set optimizer gradient to 0
                optimizer.zero_grad()
                # Compute predictions
                output = model(data1,data2,data3).double()
                loss = criterion(output, target)
                loss_ += loss.item()
                # Backpropagation
                loss.backward()
                # Gradient descent step
                optimizer.step()
        # Validation
        try:
            data1_val, data2_val, data3_val, labels_val = next(valid_dataloader_iter)
        except StopIteration:
            valid_dataloader_iter = iter(vaild_loader)
            data1_val, data2_val, data3_val, labels_val = next(valid_dataloader_iter)

        if use_cuda:
            data1_val, data2_val, data3_val, labels_val = Variable(data1_val.cuda()), Variable(
                data2_val.cuda()), Variable(data3_val.cuda()), Variable(labels_val.cuda())
        else:
            data1_val, data2_val, data3_val, labels_val = Variable(data1_val), Variable(data2_val), Variable(
                data3_val), Variable(labels_val),
        outputs_val = model(data1_val, data2_val, data3_val)
        loss_valid = criterion(outputs_val, labels_val)
        losses_valid.append(loss_valid.data)
        losses_epoch_valid.append(loss_valid.data)
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        losses_epochs_valid.append(avg_losses_epoch_valid)
        # Early Stopping
        if i == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta or i < 20:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0

            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    max_epochs = i
                    print('Early Stopped at Epoch:', i)
                    print("Epoch = ", i)
                    print("Loss = ", loss_,  ' ', 'validation_MAE = ', avg_losses_epoch_valid)
                    break
        print("Epoch = ", i)
        print("Loss = ", loss_,' ', 'validation_MAE = ', avg_losses_epoch_valid.cpu().detach().numpy())
        losses.append(loss_)
        # Store loss for display in Tensorboard
        writer.add_scalar("loss_epoch", loss_, i)
        # Schkeduler step for decrease of learning rate
        scheduler_model.step()

    model_name ="irregular_convolution_LSTM_" + str(max_epochs) + '_'+str(round(time.time()))
    print('MSE:', losses[-1], ' ', 'RMSE:', round(math.sqrt(losses[-1]), 2), ' ',)
    # Save trained model
    torch.save(model, model_name + '.pkl')
    # Plot training loss
    # plt.figure()
    # x = range(len(losses))
    # plt.plot(x, np.array(losses), label="loss")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    train_model(size_of_kernel=9)

