import os.path as osp
import numpy as np
import json
from .custom import CustomDataset
import mmcv
import pdb
import sys
import itertools
from .registry import DATASETS
import os
import shutil
import glob
# sys.path.append('/home/lianjie/mmdetection-master/mAP/')

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@DATASETS.register_module
class ChestDataset(CustomDataset):
    CLASSES = None

    def load_annotations(self, ann_file):
        self.chest = 'chest'
        self.img_ids = []
        self.ann_file = ann_file
        img_infos = json.load(open(self.ann_file))
        for i in range(len(img_infos)):
            info = img_infos[i]
            im = mmcv.imread(osp.join('/data1/liujingyu/DR/fold_all', info['file_name']))
            # im = mmcv.imread(osp.join(mycfg.data.train.img_prefix, entry['file_name']))
            info['height'], info['width'] = im.shape[0], im.shape[1]
            info['filename'] = info['file_name']
            self.img_ids.append(info['filename'])

        return img_infos

    def get_ann_info(self, idx):
        ann = self.img_infos[idx]
        return self._parse_ann_info(ann)

    def _filter_imgs(self, min_size = 32):
        # pdb.set_trace()
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and len(img_info['polygons']) < 1:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)

        return valid_inds

    def _valid_polygon(self, polygon):
        if len(polygon) < 3:
            return False
        for point in polygon:
            for xy in point:
                if xy < 0:
                    return False
        return True

    def _clip_xyxy_to_image(self, x1, y1, x2, y2, height, width):
        """Clip coordinates to an image with the given height and width."""
        x1 = np.minimum(width - 1., np.maximum(0., x1))
        y1 = np.minimum(height - 1., np.maximum(0., y1))
        x2 = np.minimum(width - 1., np.maximum(0., x2))
        y2 = np.minimum(height - 1., np.maximum(0., y2))
        return x1, y1, x2, y2

    def _parse_ann_info(self, ann):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
        Returns:
            dict: A dict containing the following keys: bboxes,
                labels, masks. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        # gt_bboxes_ignore = []
        gt_masks_ann = []

        num_classes = len(self.CLASSES)
        _class_to_ind = dict(zip(self.CLASSES, range(num_classes)))
        cat2label = {
            cat_id: cat_id + 1 for cat_id in range(num_classes)
        }

        syms, polygons = ann['syms'], ann['polygons']
        boxes = ann['boxes']
        # rows, cols = ann['rows'], ann['cols']

        for idx, (sym, polygon) in enumerate(zip(syms, polygons)):  # for each region
            # ForkedPdb().set_trace()
            if not self._valid_polygon(polygon):  # exclude non-valid polygons
                continue
            if sym in self.CLASSES:
                cls = _class_to_ind[sym]
                gt_bboxes.append(boxes[idx])
                gt_labels.append(cat2label[cls])
                poly_ann = [list(itertools.chain(*polygon))]
                gt_masks_ann.append(poly_ann)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            masks=gt_masks_ann)

        return ann

    def xyxy2xyxy(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2], # - _bbox[0] + 1,
            _bbox[3], # - _bbox[1] + 1,
        ]

    def _det2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xyxy(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.CLASSES[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xyxy(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.CLASSES[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xyxy(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.CLASSES[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results):
        if isinstance(results[0], list):
            result_files = self._det2json(results)
        elif isinstance(results[0], tuple):
            result_files, _ = self._segm2json(results)
        else:
            raise TypeError('invalid type of results')

        return result_files

    def format_results(self, results):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list): Testing results of the dataset.

        Returns:
            result_files: result_files is a dict containing the test results.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        result_files = self.results2json(results)

        return result_files

    def get_gt_info(self, gt_path):
        if os.path.exists(gt_path):
            shutil.rmtree(gt_path)
            os.makedirs(gt_path)
        else:
            os.makedirs(gt_path)

        entries = json.load(open(self.ann_file))
        file_num = 0
        for idx in range(len(entries)):
            file_num += 1
            ann = entries[idx]
            polygons = ann['polygons']
            syms = ann['syms']
            boxes = ann['boxes']

            img_file_gt = open(osp.join(gt_path, ann['file_name'].split('.png')[0] + ".txt"), "w")

            if len(ann['syms']) == 0:
                img_file_gt.close()
                continue

            for idx, (sym, polygon) in enumerate(zip(syms, polygons)):
                if not self._valid_polygon(polygon):
                    continue
                if sym in self.CLASSES:
                    x1, y1, x2, y2 = boxes[idx]

                    img_file_gt.write(str(sym) + ' ')
                    img_file_gt.write(str(x1))
                    img_file_gt.write(' ')
                    img_file_gt.write(str(y1))
                    img_file_gt.write(' ')
                    img_file_gt.write(str(x2))
                    img_file_gt.write(' ')
                    img_file_gt.write(str(y2))
                    img_file_gt.write('\n')
            img_file_gt.close()
            if file_num % 100 == 0:
                print('%d files have been test!!!' % file_num)

    def get_pred_info(self, result_files, pred_path):
        if os.path.exists(pred_path):
            shutil.rmtree(pred_path)
            os.makedirs(pred_path)
        else:
            os.makedirs(pred_path)

        for pred in result_files:

            img_file = open(osp.join(pred_path, pred['image_id'].split('.png')[0] + ".txt"), "a")
            box = pred['bbox']
            img_file.write(pred['category_id'] + ' ')
            img_file.write(str(pred['score']))

            img_file.write(' ')
            img_file.write(str(box[0]))
            img_file.write(' ')
            img_file.write(str(box[1]))
            img_file.write(' ')
            img_file.write(str(box[2]))
            img_file.write(' ')
            img_file.write(str(box[3]))
            img_file.write('\n')
            img_file.close()

    def file_lines_to_list(self, path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]

        return content

    def txt2json(self, pred_files, out_dir):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        else:
            os.makedirs(out_dir)

        pred_files_list = glob.glob(pred_files)
        pred_files_list.sort()
        for txt_file in pred_files_list:
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = self.file_lines_to_list(txt_file)
            annos = []
            for line in lines:
                class_name, confidence, left, top, right, bottom = line.split()
                if class_name in self.CLASSES:
                    new_entry = {}
                    new_entry[class_name] = [float(left), float(top),
                                         float(right), float(bottom), float(confidence)]
                    annos.append(new_entry)
            with open(out_dir + "/" + file_id + ".json", 'w') as outfile:
                json.dump(annos, outfile)

    def evaluate(self,
                 results,
                 metric='bbox'):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
        """
        dir = '/home/lianjie/mmdetection-master/mAP/'
        gt_dir = dir + 'ground-truth/'
        pred_dir = dir + 'predicted/'
        pred_files = pred_dir + '*.txt'
        pred_json_dir = dir + 'predicted_jsons/'

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        result_files = self.format_results(results)
        self.get_gt_info(gt_dir)
        self.get_pred_info(result_files, pred_dir)
        self.txt2json(pred_files, pred_json_dir)

@DATASETS.register_module
class CompDataset(ChestDataset):
    # 竞赛
    CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '结节',
               '肿块', '肺气肿', '钙化', '肺不张', '骨折', '气胸'
    ]

@DATASETS.register_module
class ProdDataset(ChestDataset):
    # 产品
    CLASSES = [
        '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',  # 1 ~ 7
        '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节',  # 8 ~ 17
        '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',  # 18 ~ 23
        '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常', '肺纹理增多'  # 24 ~ 30
    ]  # 30

@DATASETS.register_module
class EccvDataset(ChestDataset):
    # eccv
    CLASSES = [
        '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '结节',
        '肿块', '气胸', '肺气肿', '钙化', '弥漫性结节', '肺不张', '心影增大', '骨折'
    ]