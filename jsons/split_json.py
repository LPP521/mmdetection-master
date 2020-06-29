import json
import pdb
import numpy as np
import os.path as osp
import os
import shutil
import random

# 竞赛
CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '结节',
           '肿块', '肺气肿', '钙化', '肺不张', '骨折']

en_classes = {'肺实变': 'Consolidation', '纤维化表现': 'Fibrosis',
            '胸腔积液': 'Effusion', '结节': 'Nodule', '肿块': 'Mass',
            '肺气肿':'Emphysema', '钙化': 'Calcification',
            '肺不张': 'Atelectasis', '骨折': 'Fracture',
            '气胸': 'Pneumothorax'}

def First():
    json_file = 'test_x-ray.json'
    all_anno = json.load(open(json_file))
    random.shuffle(all_anno)

    with open('test_x-rayA.json', 'w') as f:
        json_file = json.dumps(all_anno[:260], ensure_ascii=False)
        f.write(json_file)

    with open('test_x-rayB.json', 'w') as f:
        json_file = json.dumps(all_anno[260:], ensure_ascii=False)
        f.write(json_file)

def count_cls(json_file):
    cls_dic = dict.fromkeys(CLASSES)
    anno = json.load(open(json_file))
    for entry in anno:
        syms = entry['syms']
        for _, sym in enumerate(syms):
            if sym in CLASSES:
                if cls_dic[sym] == None:
                    cls_dic[sym] = 1
                else:
                    cls_dic[sym] += 1

    return  cls_dic

def calc_ratio(dic1, dic2):
    ratio_dic = dict.fromkeys(CLASSES)
    for i in range(len(ratio_dic)):
        s = CLASSES[i]
        ratio_dic[s] = round((dic1[s] / dic2[s]), 2)

    return ratio_dic

def second():
    dic1 = count_cls('test_x-rayA.json')
    dic2 = count_cls('test_x-rayB.json')
    ratio_dic = calc_ratio(dic1, dic2)
    print(ratio_dic)

def gen_new(dir, json_file):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)

    anno = json.load(open(json_file))
    new_anno = []
    for entry in anno:
        new_entry = {}
        file_name = entry['file_name']
        file_path = osp.join('/data1/liujingyu/DR/fold_all', file_name)
        copy_path = osp.join(dir, file_name)
        shutil.copyfile(file_path, copy_path)

        new_entry['file_name'] = entry['file_name']
        new_syms = []
        for sym in entry['syms']:
            new_syms.append(en_classes[sym])
        new_entry['syms'] = new_syms
        new_entry['boxes'] = entry['boxes']
        new_anno.append(new_entry)

    new_json = '/home/lianjie/deepwise_x-ray/jsons/' + json_file
    with open(new_json, 'w') as f:
        # json_file = json.dumps(anno, indent=4, ensure_ascii=False)
        json_file = json.dumps(new_anno, ensure_ascii=False)
        f.write(json_file)


def gen_new_1(json_file):
    anno = json.load(open(json_file))
    new_anno = []
    for entry in anno:
        new_entry = {}
        file_name = entry['file_name']

        new_entry['file_name'] = entry['file_name']
        new_syms = []
        for sym in entry['syms']:
            new_syms.append(en_classes[sym])
        new_entry['syms'] = new_syms
        new_entry['boxes'] = entry['boxes']
        new_anno.append(new_entry)

    new_json = '/home/lianjie/deepwise_x-ray/jsons/' + json_file
    with open(new_json, 'w') as f:
        # json_file = json.dumps(anno, indent=4, ensure_ascii=False)
        json_file = json.dumps(new_anno, ensure_ascii=False)
        f.write(json_file)

def third():
    # dir = '/home/lianjie/deepwise_x-ray/img/test/'
    # gen_new(dir + 'A', 'test_x-rayA.json')
    # gen_new(dir + 'B', 'test_x-rayB.json')
    gen_new_1('test_x-ray10.json')
    gen_new_1('train_x-ray10.json')


if __name__ == '__main__':
    # First()
    # second()
    third()