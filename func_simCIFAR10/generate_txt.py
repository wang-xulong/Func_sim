import os

# train_dir = os.path.join( "Data",  "SplitMNIST","train")
# test_dir = os.path.join("Data",  "SplitMNIST", "test")
train_dir = os.path.join( "Data",  "SplitCifar10","train")
test_dir = os.path.join("Data",  "SplitCifar10", "test")
label_map = {
    '0':'0','1':'1',
    '2':'0','3':'1',
    '4':'0','5':'1',
    '6':'0','7':'1',
    '8':'0','9':'1',
}


def gen_txt(img_dir):


    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各context的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取context的文件夹下所有png图片的路径
            context_txt_path = img_dir + ' ' + sub_dir + '.txt'
            f = open(context_txt_path, 'w')
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label = label_map[img_list[i].split('_')[0]]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
            f.close()


if __name__ == '__main__':

    gen_txt(train_dir)
    gen_txt(test_dir)

