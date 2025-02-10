import csv
import glob
import os
import random
import shutil
# 找出所有csv文件


def concat_csv(csv_list):
    file_with_label = {}
    for csv_path in csv_list:
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for line in reader:
                if (line[1] not in file_with_label.keys()):
                    file_with_label[line[1]] = [line[0]]
                else:
                    file_with_label[line[1]].append(line[0])
    return file_with_label


def split_dataset(file_with_label, root, ratio):
    for label in file_with_label.keys():
        if not os.path.exists(os.path.join(root, "train", label)) and not os.path.exists(os.path.join(root, "val", label)):
            os.makedirs(os.path.join(root, "train", label))
            os.makedirs(os.path.join(root, "val", label))
        for file_name in file_with_label[label]:
            shutil.move(os.path.join(root, "images", file_name),
                        os.path.join(root, "train", label))
    for label in os.listdir(os.path.join(root, "train")):
        samples = random.sample(os.listdir(
            os.path.join(root, "train", label)), int(len(os.listdir(os.path.join(root, "train", label))) * ratio))
        for files in samples:
            shutil.move(os.path.join(root, "train", label, files),
                        os.path.join(root, "val", label))
    print("数据集划分完成！")

def main():
    root = "C:/Users/AUSU/Desktop/OwFSCIL1/local_datasets/miniimagenet/"  # 修改imagenet根目录
    csv_list = glob.glob(os.path.join(root, "*.csv"))  # 获取三个csv文件
    file_with_label = concat_csv(csv_list)  # 整合csv文件
    split_dataset(file_with_label, root, 1/6)  # 分成训练集和验证集,修改验证集比例，默认0.2

if __name__ == "__main__":
    main()

