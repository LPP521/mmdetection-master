import json
import pdb
import numpy as np
import os.path as osp
import os
import shutil

# 生成适用于mmdet框架数据接口的json文件
class mmdet_jsons():
    # eccv
    # CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '结节',
    #     '肿块', '气胸', '肺气肿', '钙化', '弥漫性结节', '肺不张', '心影增大', '骨折']

    # 竞赛
    # CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '结节',
    #                 '肿块', '肺气肿', '钙化', '肺不张', '骨折']

    # 产品
    # CLASSES = [
    #     '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',  # 1 ~ 7
    #     '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节',  # 8 ~ 17
    #     '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',  # 18 ~ 23
    #     '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常', '肺纹理增多'  # 24 ~ 30
    # ]  # 30

    # 竞赛 + 气胸
    CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '结节',
                    '肿块', '肺气肿', '钙化', '肺不张', '骨折', '气胸']


    def __init__(self):
        self.train_json = '/home/lianjie/cvpr_code/part_seg/found_yy_jsons/train_gk_yy.json'
        self.test_json = '/home/lianjie/cvpr_code/part_seg/found_yy_jsons/test_gk_yy.json'
        self.new_train_json = 'train_x-ray10.json'
        self.new_test_json = 'test_x-ray10.json'

    def valid_polygon(self, polygon):
        if len(polygon) < 3:
            return False
        for point in polygon:
            for xy in point:
                if xy < 0:
                    return False
        return True

    def clip_xyxy_to_image(self, x1, y1, x2, y2, height, width):
        """Clip coordinates to an image with the given height and width."""
        x1 = np.minimum(width - 1., np.maximum(0., x1))
        y1 = np.minimum(height - 1., np.maximum(0., y1))
        x2 = np.minimum(width - 1., np.maximum(0., x2))
        y2 = np.minimum(height - 1., np.maximum(0., y2))
        return x1, y1, x2, y2

    def gen_jsons(self, *jsons_info):
        total_sample = 0
        negative_sample = 0
        old_json = jsons_info[0]
        new_json = jsons_info[1]
        annos = json.load(open(old_json))
        new_annos = []
        for entry in annos:
            syms, polygons = entry['syms'], entry['polygon']

            if 'rows' in entry:
                rows = entry['rows']
                cols = entry['cols']
            elif 'bottom_row' in entry:
                 rows = entry['bottom_row'] - entry['top_row']
                 cols = entry['right_col'] - entry['left_col']

            new_syms = []
            new_polygons = []
            boxes = []
            new_entry = {}

            # 竞赛数据
            file_name = entry['file_name']
            # 处理一些特殊情况
            remove_flag = False
            # 弥漫性结节
            if file_name in ['58695.png', '57569.png', '45795.png', '59788.png', '60191.png',
                             '60795.png', '69838.png', '70454.png', '69845.png']:
                continue

            for idx, (sym, polygon) in enumerate(zip(syms, polygons)):  # for each region
                # ForkedPdb().set_trace()
                if not self.valid_polygon(polygon):  # exclude non-valid polygons
                    continue
                if '膈面异常' in sym and entry['doc_name'] == 'fj6311':
                    continue
                if '胸腔积液' in sym and entry['file_name'] == '39420.png':
                    continue

                if '主动脉异常' in sym and '钙化' in sym:
                    sym = ['主动脉钙化', '主动脉异常']
                if '结节' in sym and '乳头影' in sym:
                    sym = ['乳头影']

                if '结节' in sym and '弥漫性结节' in sym:
                    sym.remove('结节')
                if '结节' in sym and '多发结节' in sym:
                    sym.remove('结节')
                if '结核结节' in sym and '弥漫性结节' in sym:
                    sym.remove('结核结节')
                if '结核结节' in sym and '多发结节' in sym:
                    sym.remove('结核结节')
                if '结核球' in sym and '弥漫性结节' in sym:
                    sym.remove('结核球')
                if '结核球' in sym and '多发结节' in sym:
                    sym.remove('结核球')

                for s in sym:  # for each sub-sym
                    if s == '膈面膨隆' or s == '膈面抬高':  # awkward ...
                        s = '膈面异常'
                    if s == '肺门影浓' or s == '肺门影大':
                        s = '肺门增浓'
                    if s == '主动脉异常':
                        s = '主动脉结增宽'

                    # 以下是肺结核的征象
                    if s == '三均匀粟粒样结节' or s == '非三均匀粟粒样结节':
                        s = '弥漫性结节'
                    if s == '结核球' or s == '结核结节':
                        s = '结节'
                    if s == '索条影':
                        s = '纤维化表现'

                    # eccv 13类处理
                    if s == '骨折' or s == '肋骨缺失':
                        s = '骨折'
                    if s == '弥漫性结节' or s == '多发结节':
                        s = '弥漫性结节'

                    if s in self.CLASSES:
                        tmp = [tuple(point) for point in polygon]
                        polygon_np = np.array(tmp)
                        x1, y1, x2, y2 = polygon_np[:, 0].min(), polygon_np[:, 1].min(), \
                                         polygon_np[:, 0].max(), polygon_np[:, 1].max(),
                        if 'rows' in entry or 'bottom_row' in entry:
                            x1, y1, x2, y2 = self.clip_xyxy_to_image(
                                x1, y1, x2, y2, rows, cols)

                        # if nodule is too large, then assign it to diffusive nodules
                        if s == '结节' and (x2 - x1 > 300 or y2 - y1 > 300):
                            s = '弥漫性结节'
                            # 竞赛数据集特殊处理
                            remove_flag = True
                            break

                        # eccv
                        # expand too-small boxes (width or height < 20)
                        # if x2 - x1 < 20:
                        #     cx = (x1 + x2) * 0.5
                        #     x1 = cx - 10
                        #     x2 = cx + 10
                        # if y2 - y1 < 20:
                        #     cy = (y1 + y2) * 0.5
                        #     y1 = cy - 10
                        #     y2 = cy + 10

                        # print('fold ', fold, 'evaId ', evaId, idx + 1, '多发结节')
                        box = [int(x1), int(y1), int(x2), int(y2)]
                        new_syms.append(s)
                        new_polygons.append(polygon)
                        boxes.append(box)

            if not remove_flag:
                new_entry['data_path'] = entry['data_path']
                new_entry['file_name'] = entry['file_name']
                new_entry['doc_name'] = entry['doc_name']
                new_entry['eva_id'] = entry['eva_id']
                new_entry['fold'] = entry['fold']
                new_entry['boxes'] = boxes
                new_entry['syms'] = new_syms
                new_entry['polygons'] = new_polygons
                new_entry['rows'] = entry['rows']
                new_entry['cols'] = entry['cols']
                new_annos.append(new_entry)

                total_sample += 1

                if len(boxes) == 0:
                    negative_sample += 1

        with open(new_json, 'w') as f:
            # json_file = json.dumps(anno, indent=4, ensure_ascii=False)
            json_file = json.dumps(new_annos, ensure_ascii=False)
            f.write(json_file)
            print(total_sample, negative_sample)

if __name__ == '__main__':
    my_jsons = mmdet_jsons()
    jsons_info = [my_jsons.train_json, my_jsons.new_train_json]
    my_jsons.gen_jsons(*jsons_info)
    jsons_info = [my_jsons.test_json, my_jsons.new_test_json]
    my_jsons.gen_jsons(*jsons_info)