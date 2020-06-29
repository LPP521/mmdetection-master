import numpy as np
from scipy import interpolate
import json
import glob
import os
import pdb

def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps

def FROC(boxes_all, gts_all, iou_th):
    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    # img_idxs_ori : array([   0.,    0.,    0., ..., 4830., 4830., 4830.])
    img_idxs_ori = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    sorted_scores = scores[ord]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs_ori[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    no_lesion = 0
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if overlaps.shape[0] == 0:
            # pdb.set_trace()
            no_lesion += 1
            nMiss += 1
        elif overlaps.max() < iou_th:
            nMiss += 1
        else:
            max_index = np.argmax(overlaps)
            if overlaps[max_index] >= iou_th and not hits[img_idxs[i]][max_index]:
                hits[img_idxs[i]][max_index] = True
                nHits += 1
            elif overlaps[max_index] >= iou_th and hits[img_idxs[i]][max_index]:
                nMiss += 1

        tps.append(nHits)
        fps.append(nMiss)
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    if len(tps) != 0:
        tp_gt = [tps[-1], nGt]
    else:
        tp_gt = None

    return tp_gt, sens, fp_per_img, sorted_scores


def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
    # compute the sensitivity at avgFP (average FP per image)
    tp_gt, sens, fp_per_img, sorted_scores = FROC(boxes_all, gts_all, iou_th)

    # gt = tp and no fp
    if len(fp_per_img) != 0 and sens[-1] == 1 and fp_per_img[-1] == 0:
        res = np.ones(1)
        valid_avgFP = np.zeros(1)
        score_thresh = np.ones(1)

        return res, valid_avgFP, score_thresh
    if len(fp_per_img) != 0:
        max_fp = fp_per_img[-1]

        # tp = 0 and fp = 1
        if len(fp_per_img) == 1:
            pdb.set_trace()
            fp_per_img = np.insert(fp_per_img, 0, 0.0)
            sens = np.insert(sens, 0, 0.0)
            sorted_scores = np.insert(sorted_scores, 0, 0.0)

        if max_fp != 0:
            f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
        # s = interpolate.interp1d(sens, sorted_scores, fill_value='extrapolate')
        if (avgFP[-1] < max_fp):
            valid_avgFP_end_idx = len(avgFP)
        else:
            valid_avgFP_end_idx = np.argwhere(np.array(avgFP) >= max_fp)[0][0]
        valid_avgFP = np.hstack((avgFP[:valid_avgFP_end_idx], max_fp))
        if max_fp != 0:
            res = f(valid_avgFP)
        # no fp and tp < gt
        elif max_fp == 0 and tp_gt:
            res = np.zeros(len(valid_avgFP))
            res[:] = 1.0 * tp_gt[0] / tp_gt[1]
        # score_thresh = s(res)
        # fake score_thresh
        score_thresh = res[:] * 0.0
    # no tp
    else:
        res = np.zeros(1)
        valid_avgFP = np.zeros(1)
        score_thresh = np.zeros(1)

    return res, valid_avgFP, score_thresh

def eval_FROC(gt_boxes, all_boxes, CLASSES, avgFP=[0.05, 0.1, 0.2], iou_th=0.5):
    # all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score) 
    # all_boxes as same as gt_boxes 
    # all_boxes and gt_boxes [ [arr,       arr,       ...],  [arr,         arr,      ...],  ...]
    #                        img0_bbox   img1_bbox          img0_bbox    img1_bbox
    #                                    class0                          class1
    # classes(outside) and img_num(inside)

    # max_fp: using all fp to calculate fp_per_img
    # note that: max_fp = all_fp / nImg  Recall@max_fp = all_tp / nGt  
    mean_recall = [0.0 for i in range(len(avgFP) + 1)]

    for cls in range(len(all_boxes)):
        result, valid_avgFP, score_thresh = sens_at_FP(all_boxes[cls], gt_boxes[cls], avgFP, iou_th)
        print('='*20, 'cls: %s'%CLASSES[cls], '='*20)

        # fp_per_img 达不到avgFP里的值时 则采用max_fp的recall
        assert len(valid_avgFP) >= 1
        avgFP_ = avgFP.copy()
        avgFP_.append(valid_avgFP[-1])
        result_ = [0.0 for i in range(len(avgFP_))]
        valid_avgFP_ = [0.0 for i in range(len(avgFP_))]
        score_thresh_ = [0.0 for i in range(len(avgFP_))]
        for i in range(len(avgFP_)):
            if i < len(valid_avgFP):
                result_[i] = result[i]
                valid_avgFP_[i] = avgFP_[i]
                score_thresh_[i] = score_thresh[i]
            else:
                result_[i] = result[-1]
                valid_avgFP_[i] = avgFP_[i]
                score_thresh_[i] = score_thresh[-1]

        for idx, (recall, fp, score) in enumerate(zip(result_, valid_avgFP_, score_thresh_)):
        # for idx, (recall, fp, score) in enumerate(zip(result, valid_avgFP, score_thresh)):
            if idx < len(avgFP):
                 print('Recall@%.2f=%.2f%%, score_threshold=%.2f' % (fp, recall*100, score))
            else:
                 print('max fp_per_img=%.2f' % fp)
                 print('Recall@%.2f=%.2f%%, score_threshold=%.2f' % (fp, recall*100, score))
            mean_recall[idx] += float(recall)
        print()
        # TODO: when num of valid_avgFP < 6,is FROC correct?
        # print('Mean FROC is %.2f'% np.mean(np.array(result[:6])*100))

    print('='*20, 'recall', '='*20)
    avg_recall = 0.0
    for i in range(len(avgFP)):
        print('Recall@%.2f=%.2f%%' % (avgFP[i], mean_recall[i]/len(CLASSES)*100.0))
        avg_recall += mean_recall[i]/len(CLASSES)
    # print('Recall@max_fp=%.2f%%' % (mean_recall[-1]/len(CLASSES)*100))
    print('avg recall = %.2f%%' % (avg_recall/len(avgFP)*100.0))

def get_pred_info(pred_dir, file_id_list, CLASSES):
    class_num = len(CLASSES)
    all_boxes = [[] for _ in range(class_num)]

    for cls_idx, class_name in enumerate(CLASSES):
        for file_id in file_id_list[cls_idx]:
            file_id = file_id.replace('.png', '.json')
            json_file = os.path.join(pred_dir, file_id)
            all_box = []
            if os.path.exists(json_file):
                annos = json.load(open(json_file))
                for i in range(len(annos)):
                    if class_name in annos[i].keys():
                        all_box.append(annos[i][class_name])
            if all_box:
                all_box = np.array(all_box, dtype=np.float32)
            else:
                all_box = np.zeros((0, 5), dtype=np.float32)
            all_boxes[cls_idx].append(all_box)

    return all_boxes

def get_gt_info(test_json, CLASSES):
    class_num = len(CLASSES)
    gt_boxes = [[] for _ in range(class_num)]
    file_id_list = [[] for _ in range(class_num)]

    annos = json.load(open(test_json))
    for cls_idx, class_name in enumerate(CLASSES):
        for entry in annos:
            syms, boxes = entry['syms'], entry['boxes']
            file_name = entry['file_name']
            gt_box = []
            for idx, (sym, box) in enumerate(zip(syms, boxes)):
                if sym == class_name:
                    gt_box.append(box)
            if gt_box:
                gt_box = np.array(gt_box, dtype=np.float32)
            else:
                gt_box = np.zeros((0, 4), dtype=np.float32)
            gt_boxes[cls_idx].append(gt_box)
            file_id_list[cls_idx].append(file_name)

    return gt_boxes, file_id_list

def main():
    # CLASSES = [
    #     '肺实变', '纤维化表现', '胸腔积液', '胸膜增厚', '主动脉结增宽', '膈面异常', '结节',  # 1 ~ 7
    #     '肿块', '异物', '气胸', '肺气肿', '骨折', '钙化', '乳头影', '弥漫性结节', '肺不张', '多发结节',  # 8 ~ 17
    #     '心影增大', '脊柱侧弯', '纵隔变宽', '肺门增浓', '膈下游离气体', '肋骨异常',  # 18 ~ 23
    #     '皮下气肿', '主动脉钙化', '空洞', '液气胸', '肋骨缺失', '肩关节异常', '肺纹理增多'  # 24 ~ 30
    # ]  # 30

    CLASSES = ['肺实变', '纤维化表现', '胸腔积液', '结节',
               '肿块', '肺气肿', '钙化', '肺不张', '骨折', '气胸'
               ]

    test_json = '/home/lianjie/mmdetection-master/jsons/test_x-ray10.json'
    pred_dir = '/home/lianjie/mmdetection-master/mAP/predicted_jsons/'

    gt_boxes, file_id_list = get_gt_info(test_json, CLASSES)
    all_boxes = get_pred_info(pred_dir, file_id_list, CLASSES)
    eval_FROC(gt_boxes, all_boxes, CLASSES,
              avgFP=[0.05, 0.1, 0.2], iou_th=0.5)

if __name__ == '__main__':
    main()