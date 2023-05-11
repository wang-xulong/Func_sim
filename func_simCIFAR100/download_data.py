from avalanche.benchmarks.classic import SplitCIFAR10,SplitCIFAR100
from torchvision import utils
import os

base_dir = "./" #修改为当前Data
# data_dir = os.path.join(base_dir, "Data", "SplitMNIST")
# train_o_dir = os.path.join( base_dir, "Data", "SplitMNIST", "train")
# test_o_dir = os.path.join( base_dir, "Data", "SplitMNIST", "test")
# data_dir = os.path.join(base_dir, "Data", "SplitCifar10")
# train_o_dir = os.path.join( base_dir, "Data", "SplitCifar10", "train")
# test_o_dir = os.path.join( base_dir, "Data", "SplitCifar10", "test")
data_dir = os.path.join(base_dir, "Data", "SplitCifar100")
train_o_dir = os.path.join( base_dir, "Data", "SplitCifar100", "train")
test_o_dir = os.path.join( base_dir, "Data", "SplitCifar100", "test")

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':

    # SplitMNIST, SplitCIFAR10 是 5，  SplitCIFAR100 是 50
    n_experiences = 50
    # SplitMNIST, SplitCIFAR10 是 10，  SplitCIFAR100 是 100
    fixed_class_order = [
        4,30,55,72,95,
        1,32,67,73,91,
        54,62,70,82,92,
        9,10,16,28,61,
        0,51,53,57,83,
        22,39,40,86,87,
        5,20,25,84,94,
        6,7,14,18,24,
        3,42,43,88,97,
        12,17,37,68,76,
        23,33,49,60,71,
        15,19,21,31,38,
        34,63,64,66,75,
        26,45,77,79,99,
        2,11,35,46,98,
        27,29,44,78,93,
        36,50,65,74,80,
        47,52,56,59,96,
        8,13,48,58,90,
        41,69,81,85,89
    ]

    split_Cifar100 = SplitCIFAR100(n_experiences=n_experiences,fixed_class_order=fixed_class_order)

    train_stream = split_Cifar100.train_stream
    test_stream = split_Cifar100.test_stream

    for e, experience in enumerate(train_stream):
        current_training_set = experience.dataset
        for i in range(len(current_training_set)):
            img = current_training_set[i][0]  # 取出一张图片
            label = str(current_training_set[i][1])  # 得到对应的标签
            context = str(e)

            o_dir = os.path.join(train_o_dir,context) # 确定保存的文件夹
            my_mkdir(o_dir)

            img_name = label + '_' + str(i) + '.png'
            img_path = os.path.join(o_dir, img_name)
            utils.save_image(img,img_path)

    for e, experience in enumerate(test_stream): # respective test stream
        current_test_set = experience.dataset
        for i in range(len(current_test_set)):
            img = current_test_set[i][0]
            label = str(current_test_set[i][1])
            context = str(e)

            o_dir = os.path.join(test_o_dir,context)
            my_mkdir(o_dir)

            img_name = label + '_' + str(i) + '.png'
            img_path = os.path.join(o_dir, img_name)
            utils.save_image(img,img_path)




