import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes
from pandas import DataFrame
import math
import statistics
import pandas as pd


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection

    # Compute union

    px1 = prediction_box[0]
    py1 = prediction_box[1]
    px2 = prediction_box[2]
    py2 = prediction_box[3]
    gx1 = gt_box[0]
    gy1 = gt_box[1]
    gx2 = gt_box[2]
    gy2 = gt_box[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(px1, gx1)
    y_top = max(py1, gy1)
    x_right = min(px2, gx2)
    y_bottom = min(py2, gy2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (px2 - px1) * (py2 - py1)
    bb2_area = (gx2 - gx1) * (gy2 - gy1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp+num_fp == 0:
        return 1
    else:
        return float(num_tp/(num_tp+num_fp))

def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if num_tp+num_fn != 0:
        return float(num_tp/(num_tp+num_fn))
    else:
        return 0


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """

    rows_list = []
    iouHigh = 0

    if len(prediction_boxes) > 0 and len(gt_boxes) > 0:
        # Find all possible matches with a IoU >= iou threshold
        for j in range(0, len(prediction_boxes)):
            for i in range(0, len(gt_boxes)):
                pred = prediction_boxes[j]
                gt = gt_boxes[i]
                iou = calculate_iou(pred, gt)
                if iou >= iou_threshold and iou > iouHigh:
                    iouHigh = iou
                    dict1 = {}
                    dict1.update({"pred": pred, "gt": gt, "iou": iou})

            if iouHigh:
                rows_list.append(dict1)
                iouHigh = 0


        # Sort all matches on IoU in descending order
        comb = DataFrame(rows_list)
        if len(comb)>0:
            comb.sort_values(by='iou', ascending=False)

        if len(comb) > 0:
            pred = np.stack(comb["pred"])
            gt = np.stack(comb["gt"])

        else:
            pred = np.array([])
            gt = np.array(())
        return pred, gt
    else:
        pred = np.array([])
        gt = np.array(())
        return pred, gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    predMatches, gtMatches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    tp = len(predMatches)
    fp = len(prediction_boxes)-tp
    fn = len(gt_boxes)- len(gtMatches)

    res = {"true_pos": tp, "false_pos": fp, "false_neg": fn}

    return res

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    i = 0
    prec = float(0)
    rec = float(0)
    for (pred, gt) in zip(all_prediction_boxes, all_gt_boxes):
        res = calculate_individual_image_result(pred, gt, iou_threshold)
        i += 1

        tp = res["true_pos"]
        fp = res["false_pos"]
        fn = res["false_neg"]

        prec += calculate_precision(tp, fp, fn)
        rec += calculate_recall(tp, fp, fn)

    return prec/i, rec/i




def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE


    pres = []
    rec = []

    for val in confidence_thresholds:
        pic = []
        gt = []
        for (pred, gts, confscores) in zip(all_prediction_boxes, all_gt_boxes, confidence_scores):
            test = []
            for (pre, conf) in zip(pred, confscores):
                if conf > val:
                    test.append(pre)
            pic.append(np.array(test))
            gt.append(np.array(gts))
        a, b = calculate_precision_recall_all_images(pic, gt, iou_threshold)
        pres.append(a)
        rec.append(b)

    return np.array(pres), np.array(rec)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
        """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    data = DataFrame({'prec': precisions, 'rec': recalls})
    a = []
    for val in recall_levels:
        x = data[data['rec'] >= val].prec.max()
        if math.isnan(x):
            a.append(0)
        else:
            a.append(x)

    average_precision = statistics.mean(a)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
