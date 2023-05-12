"""
calculate the functional similarity and acc, fgt for CIFAR10 2 class
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models.resnet import resnet50
import csv

from metrics import compute_acc_fgt
from util import trainES, get_Cifar10, test

# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
basic_task = 0  # count from 0
experience = 5
train_bs = 64
test_bs = 64
lr_init = 0.001
max_epoch = 2
run_times = 1
patience = 1

accuracy_list1 = [] # multiple run
accuracy_list2 = []
accuracy_list3 = []
accuracy_list4 = []

# use GPU?
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {"num_workers":1, "pin_memory":True} if use_cuda else {}

fun_score = np.zeros((run_times, 4))
for run in range(run_times):
    print("run time: {}".format(run+1))
# ------------------------------------ step 1/5 : load data------------------------------------
    train_stream, test_stream = get_Cifar10()
# ------------------------------------ step 2/5 : define network-------------------------------
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # ------------------------------------ step 3/5 : define loss function and optimization ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # ------------------------------------ step 4/5 : training --------------------------------------------------
    # training basic task
    basic_task_data = train_stream[basic_task]
    basic_task_test_data = test_stream[basic_task]
    model, _, _, avg_valid_losses, _, _ = trainES(basic_task_data,basic_task_test_data,model,criterion, optimizer,max_epoch,device,patience = patience, func_sim = False)
    basic_loss = avg_valid_losses[-1]
    print("basic loss:{:.4}".format(basic_loss))
    # setting stage 1 matrix
    acc_array1 = np.zeros((4, 2))
    # testing basic task
    _, acc_array1[:, 0] = test(test_stream[basic_task], model, criterion, device)
    # pop the src data from train_stream and test_stream
    train_stream.pop(basic_task)
    test_stream.pop(basic_task)
    # test other tasks except basic task
    for i, probe_data in enumerate(test_stream):
        with torch.no_grad():
            _, acc_array1[i, 1] = test(probe_data, model, criterion, device)
    # save task 1
    PATH = "./"
    trained_model_path = os.path.join(PATH, 'basic_model.pth')
    torch.save(model.state_dict(), trained_model_path)

    # setting stage 2 matrix
    acc_array2 = np.zeros((4, 2))
    for j, (train_data, test_data) in enumerate(zip(train_stream, test_stream)):
        print("task {} starting...".format(j))
        # load old task's model
        trained_model = resnet50()
        trained_model.fc = nn.Linear(trained_model.fc.in_features, 2)  # final output dim = 2
        trained_model.load_state_dict(torch.load(trained_model_path))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(trained_model.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
        # training other tasks
        trained_model, _, _, _, _, new_task_first_loss = trainES(train_data, test_data, trained_model, criterion, optimizer, max_epoch,
                                            device, patience, func_sim=True)
        # record func_sim of current new task
        print("for task {}, the new_task_first_loss is {:.4}.".format(j, new_task_first_loss))
        fun_score[run, j] = (1 - (new_task_first_loss / basic_loss.item()))

        # test model on basic task and task j
        with torch.no_grad():
            _, acc_array2[j, 0] = test(basic_task_test_data, trained_model, criterion, device)
            _, acc_array2[j, 1] = test(test_stream[j], trained_model, criterion, device)
        # computing avg_acc and CF
    accuracy_list1.append([acc_array1[0, :], acc_array2[0, :]])
    accuracy_list2.append([acc_array1[1, :], acc_array2[1, :]])
    accuracy_list3.append([acc_array1[2, :], acc_array2[2, :]])
    accuracy_list4.append([acc_array1[3, :], acc_array2[3, :]])

accuracy_array1 = np.array(accuracy_list1)
accuracy_array2 = np.array(accuracy_list2)
accuracy_array3 = np.array(accuracy_list3)
accuracy_array4 = np.array(accuracy_list4)

fun_score_mean = np.mean(fun_score, axis=0)
fun_score_std = np.std(fun_score, axis=0)
print(fun_score_mean)
print(fun_score_std)


avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array1)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array2)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array3)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array4)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))

# save func_sim and metrics
np.savetxt("func_score.csv", fun_score, delimiter=',')




