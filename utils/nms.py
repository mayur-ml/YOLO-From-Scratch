import torch
from utils.iou import intersection_over_union

def nms(bboxes, iou_threshold,threshold ,box_format = "corners"):
    """
    Perform Non-Maximum Suppression (NMS) on a list of bounding boxes to filter out overlapping boxes.

    Parameters:
        bboxes (list): List of bounding boxes, where each bounding box is represented as 
                       [class_pred, prob_score, x1, y1, x2, y2]. Each element in the list
                       represents a bounding box with:
                       - class_pred (int): Predicted class.
                       - prob_score (float): Confidence score of the prediction.
                       - x1, y1, x2, y2 (float): Coordinates of the bounding box.
        iou_threshold (float): IoU threshold used to determine whether boxes overlap too much
                               with respect to the chosen box. Boxes with an IoU greater than
                               this threshold will be suppressed.
        score_threshold (float): Minimum score threshold. Bounding boxes with a score below
                                 this threshold will be discarded before applying NMS.
        box_format (str): Format of the bounding boxes. Can be either:
                          - "corners": Bounding box is represented by [x1, y1, x2, y2].
                          - "midpoint": Bounding box is represented by [x_mid, y_mid, width, height].

    Returns:
        list: A list of bounding boxes that remain after applying NMS. Each bounding box is
              represented as [class_pred, prob_score, x1, y1, x2, y2].

    Example:
        predictions = [
            [1, 0.9, 10, 10, 50, 50],
            [1, 0.8, 12, 12, 48, 48],
            [2, 0.85, 100, 100, 150, 150]
        ]
        result = nms(predictions, iou_threshold=0.5, score_threshold=0.6, box_format="corners")
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes_after_nms = []
    bboxes = sorted(bboxes , key = lambda x:x[1], reverse = True)

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box
                   for box in bboxes
                   if box[0] != chosen_box[0]
                   or intersection_over_union(
                       torch.tensor(chosen_box[2:]),
                       torch.tensor(box[2:]),
                       box_format = box_format,
                   ) < iou_threshold]
        
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms