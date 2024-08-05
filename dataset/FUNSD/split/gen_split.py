import os
import random
import json

label_name = set(("header", "question", "answer", "other"))

shots = [1,3,5,7]
splits = [0,1,2,3,4]
for shot in shots:
    for split in splits:
        # 训练集
        dir_data = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/training_data/images"
        prefix = "/".join(dir_data.split("/")[-2:])
        #dir_save = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/split/1_shot_split_1"
        dir_save = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/split/{}_shot_split_{}".format(str(shot), str(split))
        save_name = "train.txt" if "training" in dir_data else "validate.txt"
        few_shot_num = shot

        os.makedirs(dir_save, exist_ok=True)

        sample_list = os.listdir(dir_data)
        sample_list.sort()
        sample_list = [prefix + '/' + i for i in sample_list]

        if few_shot_num:
            random.shuffle(sample_list)

        # 保证所有的sample都包含了所有的value
        candidates = []
        i = 0
        while len(candidates) < shot:
            dir_json = os.path.join(dir_data, sample_list[i].split("/")[-1])
            dir_json = dir_json.replace("images", "annotations")
            dir_json = dir_json.split(".")[0] + ".json"
            label_type_in_sample = set()
            with open(dir_json) as f:
                label = json.load(f)['form']
            for item in label:
                label_type_in_sample.add(item["label"])
            if label_type_in_sample == label_name:
                candidates.append(sample_list[i])
            i += 1

        with open(os.path.join(dir_save, save_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(candidates))
        
        # 验证集
        dir_data = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/testing_data/images"
        prefix = "/".join(dir_data.split("/")[-2:])
        dir_save = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/split/{}_shot_split_{}".format(str(shot), str(split))
        save_name = "train.txt" if "training" in dir_data else "validate.txt"

        sample_list = os.listdir(dir_data)
        sample_list.sort()
        sample_list = [prefix + '/' + i for i in sample_list]

        with open(os.path.join(dir_save, save_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(sample_list))

# 这个是保存全集的
# 训练集
dir_data = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/training_data/images"
prefix = "/".join(dir_data.split("/")[-2:])
dir_save = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/split/full_set"
save_name = "train.txt" if "training" in dir_data else "validate.txt"
few_shot_num = None

os.makedirs(dir_save, exist_ok=True)

sample_list = os.listdir(dir_data)
sample_list.sort()
sample_list = [prefix + '/' + i for i in sample_list]

with open(os.path.join(dir_save, save_name), 'w', encoding='utf-8') as f:
    f.write("\n".join(sample_list))

# 验证集
dir_data = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/testing_data/images"
prefix = "/".join(dir_data.split("/")[-2:])
dir_save = "/home/neo/4TB/neo/code/layoutlmv3/datasets/FUNSD/dataset/split/full_set"
save_name = "train.txt" if "training" in dir_data else "validate.txt"

os.makedirs(dir_save, exist_ok=True)

sample_list = os.listdir(dir_data)
sample_list.sort()
sample_list = [prefix + '/' + i for i in sample_list]

with open(os.path.join(dir_save, save_name), 'w', encoding='utf-8') as f:
    f.write("\n".join(sample_list))