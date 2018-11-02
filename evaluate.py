"""
  @Time    : 2018-7-26 05:42
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : evaluate.py
  @Function: evaluate our model
  
"""
import numpy as np
import os
import yaml
from PIL import Image


def get_mask(imgname, MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is column"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i] = 1
    return gt_mask

def get_class_ids(imgname, MASK_DIR):
    """Get class_id by specified single image name"""
    filestr = imgname.split(".")[0]
    labels = []
    class_folder = MASK_DIR
    class_path = class_folder + "/" + filestr + "_json/info.yaml"
    with open(class_path) as f:
        temp = yaml.load(f.read())
        labels = temp['label_names']
        del labels[0]
    labels_form = []
    for i in range(len(labels)):
        if labels[i].find("mirror")!=-1:
            labels_form.append("mirror")
    num = len(labels_form)
    class_ids = np.ones([num], dtype=int)
    return class_ids



def iou(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        IoU = 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def accuracy(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


def ber(predict_mask, gt_mask):
    """
    BER: balance error rate.
    :param predict_mask:
    :param gt_mask:
    :return:
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
