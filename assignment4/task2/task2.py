import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


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
    x11, y11, x12, y12 = np.split(prediction_box, 4)
    x21, y21, x22, y22 = np.split(gt_box, 4)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))        
    
    boxAArea = np.abs((x12 - x11) * (y12 - y11))
    boxBArea = np.abs((x22 - x21) * (y22 - y21))       
    # Compute intersection
    intersection = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # Compute union
    union = boxAArea + np.transpose(boxBArea) - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return float(iou)


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
    if num_tp == 0 and num_fp == 0:
        return 1 
    return num_tp / (num_tp + num_fp)


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
    if num_tp == 0 and num_fn == 0:
        return 0 
    return num_tp / (num_tp + num_fn)


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
    # Find all possible matches with a IoU >= iou threshold

    gt_boxes = np.array(list(set(list(map(tuple, gt_boxes)))))

    if(prediction_boxes.shape == (4,)):
        prediction_boxes = np.reshape(prediction_boxes, (1, 4))

    if(gt_boxes.shape == (4,)):
        gt_boxes = np.reshape(gt_boxes, (1, 4))

    result = np.zeros([1, 9])
    for i in range(len(prediction_boxes)):
        for j in range(len(gt_boxes)):
            iou = calculate_iou(prediction_boxes[i,:], gt_boxes[j, :])
            if (iou > iou_threshold):
                entry = np.hstack(np.array([iou, prediction_boxes[i,:], gt_boxes[j, :]]))
                entry = np.reshape(entry,(1,9))
                result = np.concatenate((result, entry), axis=0)
    result = result[1:,:]

    result[result[:,0].argsort()]
    return np.array(result[:,1:5]), np.array(result[:,5:9])


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
    pred, gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    true_positives = pred.shape[0]
    false_positives = prediction_boxes.shape[0] - pred.shape[0]
    false_negtatives = gt_boxes.shape[0] - gt.shape[0]

    result = {"true_pos": true_positives, "false_pos": false_positives, "false_neg": false_negtatives}
    return result


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
    total_false_positive = 0
    total_false_negative = 0
    total_true_positive = 0
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        result_img = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        total_false_positive += result_img.get("false_pos")
        total_false_negative += result_img.get("false_neg")
        total_true_positive += result_img.get("true_pos")

    precision = total_true_positive /  (total_true_positive + total_false_positive)
    recall = total_true_positive / (total_true_positive + total_false_negative) 
    return (precision, recall)



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

    all_prediction_boxes = np.reshape(np.vstack(all_prediction_boxes), (-1, 4))
    all_gt_boxes = np.reshape(np.vstack(all_gt_boxes), (-1, 4))
    confidence_scores = np.hstack(confidence_scores)

    # sort the boxes according to the corresponding confidence scores in ascending order
    all_prediction_boxes = all_prediction_boxes[confidence_scores.argsort()]
    all_gt_boxes = all_gt_boxes[confidence_scores.argsort()]
    confidence_scores = np.sort(confidence_scores)

    # reverse the order for descending order
    all_prediction_boxes = all_prediction_boxes[::-1]
    all_gt_boxes = all_gt_boxes[::-1]
    confidence_scores = confidence_scores[::-1]

    total_detections = calculate_individual_image_result(all_prediction_boxes, all_gt_boxes, iou_threshold)

    total_tp = total_detections.get("true_pos")
    total_fn = total_detections.get("false_neg")

    # first column is the confidence value, second the accumulated precision and third the accumulated recall
    pr_table = np.zeros([1, 3])

    for i in range(len(all_prediction_boxes)):
        predicition_box_accumulated = all_prediction_boxes[:i+1,:]
        gt_box_accumulated = all_gt_boxes[:i+1,:]
        
        if i == 0:
            predicition_box_accumulated = np.reshape(predicition_box_accumulated, (1, 4))
            gt_box_accumulated = np.reshape(gt_box_accumulated, (1, 4))
        acc_detections = calculate_individual_image_result(predicition_box_accumulated, gt_box_accumulated, iou_threshold)
        acc_tp = acc_detections.get("true_pos")
        acc_fp = acc_detections.get("false_pos")
        acc_recall = acc_tp / (total_tp + total_fn)
        acc_precision = acc_tp / (acc_tp + acc_fp)
        entry = np.array((confidence_scores[i], acc_precision, acc_recall))
        entry = np.reshape(entry, (1,3))
        pr_table = np.concatenate((pr_table, entry), axis=0)
    pr_table = pr_table[1:,:]
    # print(pr_table)
    # print(pr_table[:,0])
    unique_recall_values = np.array(list(set(list(pr_table[:,2]))))

    pr_table_unique_confidence = np.zeros((1,3))
    for unique_confidence in unique_recall_values:
        pr_table_val = pr_table[np.where(pr_table[:,2] == unique_confidence)]
        pr_table_max = pr_table_val[np.argmax(pr_table_val[:,1])]
        pr_table_max = np.reshape(pr_table_max, (1,3))
        pr_table_unique_confidence = np.concatenate((pr_table_unique_confidence, pr_table_max), axis=0)
        # pr_table_max_indices = [] 
    pr_table = pr_table_unique_confidence[1:, :]

    # sort the boxes according to the corresponding confidence scores in ascending order
    pr_table = pr_table[pr_table[:,0].argsort()]

    # reverse the order for descending order
    pr_table = pr_table[::-1]

    precisions = [] 
    recalls = []
    k = 0
    for recall in confidence_thresholds:
        row_candidates = np.where(pr_table[:,2] >= recall)
        if len(row_candidates[0]) == 0:
            precisions.append(0)
            k = k + 1
            recalls.append(recall)
        else: 
            # print(pr_table[row_candidates[0]])
            table_row = (pr_table[row_candidates[0]][0,:])
            precisions.append(table_row[1])
            recalls.append(recall)
            # precisions.append(pr_table[row_candidates(0)])

 
    return np.array(precisions), np.array(recalls)


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
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
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
    average_precision = 0
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
