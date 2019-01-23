import classification_mm as cmm

def easy_moderate_hard(bbox_height, occlusion_level, truncation):
    """Determines a difficulty of detection an object on an image using KITTI Benchmark rules (http://www.cvlibs.net/datasets/kitti/eval_object.php).   
    
    Args:
        bbox_height: A height of the object's ground truth bounding box in pixels, y_max - y_min.
        occlusion_level: A measure of the object occluded (overlapped) by other objects,
            0 is fully visible,
            1 is partly occluded,
            2 is largely occluded (difficult to see),
            3 is unknown.
        truncation: A measure refers to the object leaving image boundaries (part of the object is out of the image),
           float number from 0 (non-truncated) to 1 (the object is completely out of image).
    
    Returns:
        A detection difficulty level,
            0 is Easy       - Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 0.15,
            1 is Moderate   - Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 0.3,
            2 is Hard       - Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 0.5,
           -1 is Don't Care - examples with "Don't Care" label are ignored.
    
    """
    
    if bbox_height >= 40 and (occlusion_level == 0 or occlusion_level == 3) and truncation <= 0.15:
        return 0 # Easy
    if bbox_height >= 25 and (occlusion_level <= 1 or occlusion_level == 3) and truncation <= 0.3:
        return 1 # Moderate
    if bbox_height >= 25 and (occlusion_level <= 2 or occlusion_level == 3) and truncation <= 0.5:
        return 2 # Hard
    return -1 # Don't care

    
def bbox_overlap_area(bbox_a, bbox_b):
    """Computes an area of two bounding boxes overlap.
    
    Args:
        bbox_a: A first bounding box  [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max].
        bbox_b: A second bounding box [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max].
    
    Returns:
    
        An area of the overlap of two bounding boxes. 
    """
    
    return max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0])) * max(0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]))
    
def bbox_area(bbox):
    """Computes an area of a bounding box.
    
    Args:
        bbox: A bounding box [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max].
    
    Returns:
        An area of the bounding box.
        
    """
    
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
def bbox_iou(bbox_a, bbox_b):
    """Computes Intersection over Union (IoU) of two bounding box.
    IoU = (Area of Overlap / Area of Union)
    
    Args:
        bbox_a: A first bounding box [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max].
        bbox_b: A second bounding box [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max].
    
    Returns:
        An IoU value.
        
    """
    
    overlap_area = bbox_overlap_area(bbox_a, bbox_b)
    return float(overlap_area) / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area)

def compare_bbox(ground_truth_bboxes, classifier_output_bboxes,  dontcare_bboxes = [], iou_threshold = 0.5):
    """Compares list of ground truth class labels (ground truth bounding boxes) and list of classifier outputs (predicted bounding boxes).
    Bounding box (bbox) encoded as [x_min, y_min, x_max, y_max] or [y_min, x_min, y_max, x_max]
    
    Args:
        ground_truth_bboxes:      A list or array of ground truth bounding boxes, e.g. [[15, 20, 24, 40], [75, 80, 93, 89], ...] or [[0.15, 0.20, 0.24, 0.40], [0.75, 0.80, 0.93, 0.89], ...].
        classifier_output_bboxes: A list or array of predicted    bounding boxes, e.g. [[14, 21, 23, 41], [33, 5, 48, 22], ...]  or [[0.14, 0.21, 0.23, 0.41], [0.33, 0.05, 0.48, 0.22], ...].
        dontcare_bboxes:          A list or array of bounding boxes labels as "Don't Care". False positives on "Don't Care" bounding boxes are ignored.
        iou_threshold:            A minimum value of Intersection over Union (IoU) of a ground truth bbox and a predicted bbox to match them. 
    
    Returns: 
         An object of class MetricManager, to get true positive count, false positive count, false negative count, true negative count, accuracy, recall, precision, specificity.
    
    """
    
    # List of bbox flags. The flag indicates that a predicited bbox is matched with a ground truth bbox.
    checked_md_list = [False] * len(classifier_output_bboxes)
    
    # Phase 1. Search True Positive and True Negative.
    tp_count = 0
    
    for gt_bbox in ground_truth_bboxes: # for each ground truth bbox we select a predicted bbox with maximum IoU.
        max_iou = None
        max_iou_id = None
        for id_box, cl_bbox in enumerate(classifier_output_bboxes):
            if not checked_md_list[id_box]: # this predicted bbox has not been matched yet.
                iou = bbox_iou(gt_bbox, cl_bbox)
                if iou >= iou_threshold and (max_iou == None or iou > max_iou):
                    max_iou = iou
                    max_iou_id = id_box
        
        if not max_iou == None:
            # We found a predicted bbox match for this ground truth bbox. So it is true positive. 
            checked_md_list[max_iou_id] = True
            tp_count += 1
    
    # All unmatched ground truth bbox are false negatives.
    fn_count = len(ground_truth_bboxes) - tp_count
    
    # Phase 2. Search False Positives. 
    # Each predicted bbox unmatched with ground truth bbox is potential false positive. 
    # We need to check whether a potential false positive predicted bbox matches with a "Don't Care" bbox.
    for id_box, cl_bbox in enumerate(classifier_output_bboxes):
        if checked_md_list[id_box]: # this predicted bbox is matched, so it is not false positive.
            continue
        
        for dc_box in dontcare_bboxes:
            # False positives on a "Don't Care" bbox are ignored. Each "Don't Care" bbox can cover several false positives.
            # So we don't use IoU to match a potential false positive predicted bbox and a "Don't Care" bbox.
            # Instead of IoU, we compute a percentage of a area of predicted bbox covered by a "Don't Care" bbox.
            if float(bbox_overlap_area(cl_bbox, dc_box)) / bbox_area(cl_bbox) >= iou_threshold:
                # it is not false positive.
                checked_md_list[id_box] = True
                break
    
    fp_count = checked_md_list.count(False)
    
    return cmm.MetricManager(tp_count, fp_count, fn_count) # It's imbalanced classification, therefore False Negatives is't used.
