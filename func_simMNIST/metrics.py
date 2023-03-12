import numpy as np
from scipy.stats import sem
import scipy.stats as stats
import torch


def transferability_array(acc_transfer, acc_directly):
    transfer_ability_array = []
    for i in range(acc_transfer.shape[0]):
        single_acc_transfer = acc_transfer[i]
        single_acc_directly = acc_directly[i]

        errorT = np.ones(single_acc_directly.shape)-single_acc_directly
        errorS2T= np.ones(single_acc_transfer.shape)-single_acc_transfer

        transfer_ability = np.zeros(single_acc_transfer.shape)

        # 计算本次的迁移力
        for j in range(single_acc_transfer.shape[1]): # 逐列计算
            transfer_ability[:,j] = (( errorT.reshape(-1,) - errorS2T[:,j]) / errorT.reshape(-1,))*100

        transfer_ability_array.append(transfer_ability)

    return transfer_ability_array

def transferability(acc_transfer, acc_directly):
    mean_acc_transfer = np.mean(acc_transfer,axis=0)
    mean_acc_directly = np.mean(acc_directly,axis=0)

    errorT = np.ones(mean_acc_directly.shape)-mean_acc_directly
    errorS2T= np.ones(mean_acc_transfer.shape)-mean_acc_transfer

    transfer_ability = np.zeros(mean_acc_transfer.shape)
    # 计算迁移力
    for i in range(mean_acc_transfer.shape[1]): # 逐列计算
        transfer_ability[:,i] = (( errorT.reshape(-1,) - errorS2T[:,i]) / errorT.reshape(-1,))*100

    return transfer_ability

def accuracy(y_pred, y):
  y_pred = torch.argmax(y_pred, dim=1)  #将预测对应的概率的最大值的位置找到
  return (y_pred == y).float().mean()        #对比匹配一致的数量，均值

def compute_acc_fgt(end_task_acc_arr):
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1 + 0.95) / 2, n_run - 1)  # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]  # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)  # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), np.std(avg_acc_per_run),t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), np.std(avg_fgt),t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), np.std(acc_per_run), t_coef * sem(acc_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc

def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt




def single_run_avg_end_fgt(acc_array):
    best_acc = np.max(acc_array, axis=1)
    end_acc = acc_array[-1]
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets)
    return avg_fgt
