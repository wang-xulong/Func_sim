import os

# train_dir = os.path.join( "Data",  "SplitMNIST","train")
# test_dir = os.path.join("Data",  "SplitMNIST", "test")
# train_dir = os.path.join( "Data",  "SplitCifar10","train")
# test_dir = os.path.join("Data",  "SplitCifar10", "test")
train_dir = os.path.join( "Data",  "SplitCifar100_2class","train")
test_dir = os.path.join("Data",  "SplitCifar100_2class", "test")
#
# label_map = {# For SplitMNIST and SplitCifar10
#     '0':'0','1':'1',
#     '2':'0','3':'1',
#     '4':'0','5':'1',
#     '6':'0','7':'1',
#     '8':'0','9':'1',
# }
#

# label_map = {# For SplitCifar100
#     '0':'0','1':'1',
#     '2':'0','3':'1',
#     '4':'0','5':'1',
#     '6':'0','7':'1',
#     '8':'0','9':'1',
#     '10':'0','11':'1',
#     '12':'0','13':'1',
#     '14':'0','15':'1',
#     '16':'0','17':'1',
#     '18':'0','19':'1',
#     '20':'0','21':'1',
#     '22':'0','23':'1',
#     '24':'0','25':'1',
#     '26':'0','27':'1',
#     '28':'0','29':'1',
#     '30':'0','31':'1',
#     '32':'0','33':'1',
#     '34':'0','35':'1',
#     '36':'0','37':'1',
#     '38':'0','39':'1',
#     '40':'0','41':'1',
#     '42':'0','43':'1',
#     '44':'0','45':'1',
#     '46':'0','47':'1',
#     '48':'0','49':'1',
#     '50': '0', '51': '1',
#     '52': '0', '53': '1',
#     '54': '0', '55': '1',
#     '56': '0', '57': '1',
#     '58': '0', '59': '1',
#     '60': '0', '61': '1',
#     '62': '0', '63': '1',
#     '64': '0', '65': '1',
#     '66': '0', '67': '1',
#     '68': '0', '69': '1',
#     '70': '0', '71': '1',
#     '72': '0', '73': '1',
#     '74': '0', '75': '1',
#     '76': '0', '77': '1',
#     '78': '0', '79': '1',
#     '80': '0', '81': '1',
#     '82': '0', '83': '1',
#     '84': '0', '85': '1',
#     '86': '0', '87': '1',
#     '88': '0', '89': '1',
#     '90': '0', '91': '1',
#     '92': '0', '93': '1',
#     '94': '0', '95': '1',
#     '96': '0', '97': '1',
#     '98': '0', '99': '1',
# }
label_map_2class = {
    '4':'0', '30':'1',
    '55':'0', '72':'1',
    '95':'0', '1':'1',
    '32':'0', '67':'1',
    '69':'0', '81':'1',
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
                label = label_map_2class[img_list[i].split('_')[0]]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
            f.close()


if __name__ == '__main__':

    gen_txt(train_dir)
    gen_txt(test_dir)

