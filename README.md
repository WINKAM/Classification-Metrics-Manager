
# Classification Metrics Manager

Classification Metric Manager is  metrics calculator for  machine learning classification quality such as Precision, Recall, F-score, etc.

It can be used with both Python standard data structures and Numpy arrays. So you can apply Classification Metric Manager for machine learning framework as Keras, scikit-learn, Tensor Flow for Classification, Detection, Recognition  tasks.

*Contributions and feedback are very welcome!*


--------
## From 
[WINKAM R&D Lab](https://winkam.com)

--------
## Features
*  Binary classification with "Don't Care" class labels.
    * Compare ground truth labels and classifier results.
    * Recall (True Positive Rate, TPR).
    * Precision (Positive Predictive Values, PPV).
    * Specificity (True Negative Rate, TNF, 1.0 - False Positive Rate).
    * Accuracy (ACC).
    * F-score (F1-score and F&beta;-score).
    * Area Under Precision-Recall Curve (Precision-Recall AUC).
    * Average Precision (AP).
    * Area Under Receiver Operating Characteristics Curve (AUC ROC, AUROC).
* Computer Vision, Object detection.
    * Intersection over Union (IOU).
    * Compare ground truth bounding boxes and classifier output (predicted) bounding boxes.
    * Determination detection difficulty using [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) Benchmark rules.
    
--------
## Examples
### Simple example for binary classification.
```python
import classification_mm as cmm

# 1 is positive, 
# 0 is negative, 
# -1 is "Don't Care" class 
# (examples with "Don't Care" label are ignored, so it doesn't lead to true positive, false positive, true negative or false negative)
ground_truth_labels = [1, 1, 1, -1, 0, 0, 1, 1, 0, -1, 1, 1] 
classifier_output   = [1, 0, 1,  1, 1, 0, 1, 1, 0,  0, 0, 1]

metrics = cmm.compare_2_class(ground_truth_labels, classifier_output)
print('Metrics: ' + str(metrics) + '.')

# in case of numpy array 
# 
# import numpy as np
# ground_truth_labels = np.array(ground_truth_labels)
# classifier_output_labels = np.array(classifier_output)
# metrics = cmm.compare_2_class(ground_truth_labels, classifier_output)

print('Metrics: ' + str(metrics) + '.')

print('Recall: \t' + '{:0.1f}%'.format(100. * metrics.recall))
print('Precision: \t' + '{:0.1f}%'.format(100. * metrics.precision))
print('Specificity: \t' + '{:0.1f}%'.format(100. * metrics.specificity))
print('Accurancy: \t' + '{:0.1f}%'.format(100. * metrics.accuracy))
print("F1-score: \t" + '{:0.1f}%'.format(100. * metrics.f1_score))
print('F5-score: \t' + '{:0.1f}%'.format(100. * metrics.f_score(5)))
```

### Simple example for object detection
```python
import classification_mm as cmm
from classification_mm import cmm_cv as cmm_cv

img_1_ground_truth_labels = [(15, 20, 24, 40), (75, 80, 93, 89), (30, 5, 45, 20)]
img_1_model_output_labels = [(14, 21, 23, 41), (33, 5, 48, 22), (52, 60, 66, 75)] 
img_1_dont_care = []

# Image 1: the first ground truth bbox is detected by the first model output bbox (+1 true positive)
# , the second ground truth bbox isn't detected (+1 false negative)
# , the third ground truth bbox is detected by the second model output bbox (+1 true positive)
# , the third model output bbox is false positive (+1 false positive)


img_2_ground_truth_labels = [(18, 22, 27, 44), (70, 75, 87, 83)]
img_2_model_output_labels = [(17, 23, 25, 43), (52, 60, 66, 75), (95, 10, 105, 20)] # 1 true positive, 1 false negative, 2 false positive
img_2_dont_care = [(90, 5, 110, 25)]

# Image 2: the first ground truth bbox is detected by the first model output bbox (+1 true positive)
# , the second ground truth bbox isn't detected (+1 false negative)
# , the second model output bbox is ignored due to don't care bbox

img_1_metrics = cmm_cv.compare_bbox(img_1_ground_truth_labels, img_1_model_output_labels, img_1_dont_care)
img_2_metrics = cmm_cv.compare_bbox(img_2_ground_truth_labels, img_2_model_output_labels, img_2_dont_care)

print(img_1_metrics)
print(img_2_metrics)
print(img_1_metrics + img_2_metrics)
```

--------
## License
[MIT License](./LICENSE) 
