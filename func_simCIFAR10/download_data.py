from avalanche.benchmarks.classic import SplitCIFAR10
from torchvision import utils
import os

base_dir = "/home" #修改为当前Data 目录所在的绝对路径
# data_dir = os.path.join(base_dir, "Data", "SplitMNIST")
# train_o_dir = os.path.join( base_dir, "Data", "SplitMNIST", "train")
# test_o_dir = os.path.join( base_dir, "Data", "SplitMNIST", "test")
data_dir = os.path.join(base_dir, "Data", "SplitCifar10")
train_o_dir = os.path.join( base_dir, "Data", "SplitCifar10", "train")
test_o_dir = os.path.join( base_dir, "Data", "SplitCifar10", "test")

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':

    n_experiences = 5
    fixed_class_order = list(range(10))

    split_Cifar10 = SplitCIFAR10(n_experiences=n_experiences, fixed_class_order=fixed_class_order)

    train_stream = split_Cifar10.train_stream
    test_stream = split_Cifar10.test_stream

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




