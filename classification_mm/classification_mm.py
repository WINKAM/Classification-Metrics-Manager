def precision(tp_count, fp_count):
    """Computes Precision (Positive Predictive Values, PPV).
    Precision is the fraction of examples labeled positive that are correctly classified among true positive and false positive classifier results.    
    Precision = tp_count / (tp_count + fp_count).
    If tp_count of fp_count are not positive numbers or sum of them is 0, then the function returns 0.
   
    Args:
        tp_count: A count of true  positive classifier results (ground truth class label is positive and classifier output is positive).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
   
    Returns:
        Precision (float).
   
    """
    
    return a_ab(tp_count, fp_count)

def recall(tp_count, fn_count):
    """Computes Recall (True Positive Rate, TPR, Sensitivity).
    Recall is the fraction of examples labeled positive that are correctly classified among the total number of positives examples.
    Recall = tp_count / (tp_count + fn_count).
    If tp_count of fn_count are not positive numbers or sum of them is 0, then the function returns 0.
   
    Args:
        tp_count: A count of true  positive classifier results (ground truth class label is positive and classifier output is positive).
        fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
   
    Returns:
        Recall (float).
   
    """
    
    return a_ab(tp_count, fn_count)


def specificity(tn_count, fp_count):
    """Computes Specificity (True Negative Rate, TNF, 1.0 - False Positive Rate).
    Recall is the fraction of examples labeled negative that are correctly classified among the total number of negative examples.
    Recall = tn_count / (tn_count + fp_count).
    If tn_count of fp_count are not positive numbers or sum of them is 0, then the function returns 0.
    
    Args:
        tn_count: A count of true  negative classifier results (ground truth class label is negative and classifier output is negative).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
    
    Returns:
        Specificity (float).
    
    """
    
    return a_ab(tn_count, fp_count)


def accuracy(tp_count, fp_count, fn_count, tn_count):
    """Computes Accuracy (ACC).
    Accuracy is the fraction of true classifier results (both true positives and true negatives) among the total number of examples (both positives and negatives).  
    Recall = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count).
    If the sum of the arguments less than zero, the function returns 0.
    
    Args:
        tp_count: A count of true positive  classifier results (ground truth class label is positive and classifier output is positive).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
        fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
        tn_count: A count of true negative  classifier results (ground truth class label is negative and classifier output is negative).
    
    Returns:
        Accuracy (float).
    
    """
    
    summ = (tp_count + tn_count + fp_count + fn_count)
    return (float(tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)) if summ > 0 else 0


def f_score(tp_count, fp_count, fn_count, beta):
    """Computes F-score (F-measure).
    F-score is a measure of classification quality taking into account both Precision and Recall.
    F-score = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall).
    
    Args:
        tp_count: A count of true positive  classifier results (ground truth class label is positive and classifier output is positive).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
        fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
        beta:    A Precision-Recall balance parameter, so Recall is beta times more important than Precision.
           beta --> 0, F-beta --> Precision; beta --> infinite, F-beta --> Recall.
    
    Returns:
        F-score (float) 
    
    """
    
    return a_ab((1 + beta**2) * float(tp_count), beta**2 * fn_count + fp_count)


def f1_score(tp_count, fp_count, fn_count):    
    """Computes F1-score (F1-measure, balanced F-score, F-score with beta = 1).
    F1-score is a measure of classification quality taking into account both Precision and Recall.
    F1-score = 2 * (Precision * Recall) / (Precision + Recall).
    
    Args:
        tp_count: A count of true positive classifier results (ground truth class label is positive and classifier output is positive).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
        fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
    
    Returns:
        F1-score (float). 
    
    """
    
    return f_score(tp_count, fp_count, fn_count, 1.)

def a_ab(a, b):
    """Computes a / (a + b) for positive numbers.
   
    Args:
        a: A dividend and the first added of a divider.
        b: The second added of the divider.
   
    Returns:
        The result of calculation of a and b are positive numbers; if a or b is None, it returns none; if a or b is not positive numbers or a + b == 0, it returns 0.
   
    """
    
    if a == None or b == None:
        return None
    
    return (float(a) / (a + b)) if (a >= 0 and b >= 0 and a + b > 0) else 0.


def compare_2_class(ground_truth, classifier_output):
    """Compares list of ground truth class labels and list of classifier outputs.
    Class labels:
                  1 is positive. 
                  0 is negative.
                 -1 is "Don't Care" class, examples with "Don't Care" label are ignored, so it doesn't lead to true positive, false positive, true negative or false negative.
    Args:
        ground_truth:      A list or array of ground truth class labels, e.g. [1, 1, -1, 0, 1, 0, -1, 0].
        classifier_output: A list or array of classifier outputs,        e.g. [0, 1,  1, 1, 0, 1,  1, 0].
    
    Returns: 
         Object of class MetricManager, to get true positive count, false positive count, false negative count, true negative count, accuracy, recall, precision, specificity.
    
    """

    # if input is numpy array, use numpy to optimize computing
    is_np_used = ('numpy' in str(type(ground_truth)))
    
    if is_np_used:
        diff = (ground_truth + 3 * classifier_output).tolist()
        tp_count = diff.count(4)
        tn_count = diff.count(0)
        fp_count = diff.count(3)
        fn_count = diff.count(1)
        
    else:        
        tp_count, tn_count, fp_count, fn_count = (0, 0, 0, 0)

        for gt, mo in zip(ground_truth, classifier_output):
            if not gt == -1: # is't Don't Care
                if gt == 1: # positive
                    if mo == gt:
                        tp_count += 1
                    else:
                        fn_count += 1
                else: # negative
                    if mo == gt:
                        tn_count += 1
                    else:
                        fp_count += 1
        
    return MetricManager(tp_count, fp_count, fn_count, tn_count)


class MetricManager:
    """A class to store basis classification quality metrics and to get metrics such as precision, recall, etc.
    
    Attributes:
        tp_count: A count of true positive  classifier results (ground truth class label is positive and classifier output is positive).
        fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
        fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
        tn_count: A count of true negative  classifier results (ground truth class label is negative and classifier output is negative).
    """
    
    def __init__(self, tp_count, fp_count, fn_count, tn_count = None):
        """Create a new MetricManager
        
        Args:
            tp_count: A count of true positive  classifier results (ground truth class label is positive and classifier output is positive).
            fp_count: A count of false positive classifier results (ground truth class label is negative and classifier output is positive).
            fn_count: A count of false negative classifier results (ground truth class label is positive and classifier output is negative).
            tn_count: A count of true negative  classifier results (ground truth class label is negative and classifier output is negative).
        
        Returns: 
             Object of class MetricManager.
        
        """
        
        self.tp_count = tp_count
        self.fp_count = fp_count
        self.fn_count = fn_count
        self.tn_count = tn_count
        
        
    @property
    def precision(self):
        """Computes Precision (Positive Predictive Values, PPV).
        Precision is the fraction of examples labeled positive that are correctly classified among true positive and false positive classifier results.    
        Precision = tp_count / (tp_count + fp_count).
        If tp_count of fp_count are not positive numbers or sum of them is 0, then the function returns 0.
       
        Returns:
            Precision (float).
       
        """
        
        return a_ab(self.tp_count, self.fp_count)
    @property
    def recall(self):
        """Computes Recall (True Positive Rate, TPR, Sensitivity).
        Recall is the fraction of examples labeled positive that are correctly classified among the total number of positives examples.
        Recall = tp_count / (tp_count + fn_count).
        If tp_count of fn_count are not positive numbers or sum of them is 0, then the function returns 0.
              
        Returns:
            Recall (float).
       
        """
        
        return a_ab(self.tp_count, self.fn_count)


    @property
    def specificity(self):
        """Computes Specificity (True Negative Rate, TNF, 1.0 - False Positive Rate).
        Recall is the fraction of examples labeled negative that are correctly classified among the total number of negative examples.
        Recall = tn_count / (tn_count + fp_count).
        If tn_count of fp_count are not positive numbers or sum of them is 0, then the function returns 0.
                
        Returns:
            Specificity (float).
        
        """
    
        return a_ab(self.tn_count, self.fp_count)

    @property
    def accuracy(self):
        """Computes Accuracy (ACC).
        Accuracy is the fraction of true classifier results (both true positives and true negatives) among the total number of examples (both positives and negatives).  
        Recall = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count).
        If the sum of the arguments less than zero, the function returns 0.
                
        Returns:
            Accuracy (float).
        
        """
        return accuracy(self.tp_count, self.fp_count, self.fn_count, self.tn_count) if not self.tn_count == None else None 


    @property
    def f1_score(self):
        """Computes F1-score (F1-measure, balanced F-score, F-score with beta = 1).
        F1-score is a measure of classification quality taking into account both Precision and Recall.
        F1-score = 2 * (Precision * Recall) / (Precision + Recall).
                
        Returns:
            F1-score (float). 
        
        """
        
        return f1_score(self.tp_count, self.fp_count, self.fn_count)


    def f_score(self, beta):
        """Computes F-score (F-measure).
        F-score is a measure of classification quality taking into account both Precision and Recall.
        F-score = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall).
        
        Args:
            beta:    A Precision-Recall balance parameter, so Recall is beta times more important than Precision.
               beta --> 0, F-beta --> Precision; beta --> infinite, F-beta --> Recall.
        
        Returns:
            F-score (float) 
        
        """
        
        return f_score(self.tp_count, self.fp_count, self.fn_count, beta)
    

    def __str__(self):
        return 'TP = ' + str(self.tp_count) + '; FP = ' + str(self.fp_count) + '; FN = ' + str(self.fn_count) + (('; TN = ' + str(self.tn_count)) if not self.tn_count == None else '')
    
    
    def __add__(self, other):
        m = MetricManager(self.tp_count, self.fp_count, self.fn_count, self.tn_count)
        m.tp_count += other.tp_count
        m.fp_count += other.fp_count
        m.fn_count += other.fn_count
        m.tn_count = (self.tn_count + other.tn_count) if not self.tn_count == None and not other.tn_count == None else None
        return m


def pr_auc(precision_recall_list):
    """Computes Area Under Precision-Recall Curve (Precision-Recall AUC).
    
    Args:
        precision_recall_list: A Precision-Recall Curve encoded as a list of pairs Precision value, Recall value.
    
    Returns:
        An AUC of a Precision-Recall Curve.
    
    """
    
    sorted_pr_list = sorted(precision_recall_list, key=lambda x: (x[1], -x[0]))
    sorted_pr_list = [(1., 0.)] + sorted_pr_list + [(0., 1.)]
    
    auc = 0
    for i in range(1, len(sorted_pr_list)):
        auc += (sorted_pr_list[i][0] + 0.5 * (sorted_pr_list[i][0] - sorted_pr_list[i -1][0])) * (sorted_pr_list[i][1] - sorted_pr_list[i - 1][1])
    
    return auc

def average_precision(precision_recall_list):
    """Computes Average Precision (AP).
    
    Args:
        precision_recall_list: A Precision-Recall Curve encoded as a list of pairs Precision value, Recall value.
    
    Returns:
        A Average Precision value
    
    """
    sorted_pr_list = sorted(precision_recall_list, key=lambda x: (x[1], -x[0]))
    sorted_pr_list = [(1., 0.)] + sorted_pr_list + [(0., 1.)]
    
    ap = 0
    for i in range(1, len(sorted_pr_list)):
        ap += sorted_pr_list[i][0] * (sorted_pr_list[i][1] - sorted_pr_list[i - 1][1])

    return ap

def roc_auc(fpr_tpr_list):
    """Computes Area Under Receiver Operating Characteristics Curve (AUC ROC, AUROC).
    
    Args:
        fpr_tpr_list: A True Positive Rate - False Positive Rate Curve encoded as a list of pairs TPR value, FPR value.    
    
    Returns:
        A AUC of a Receiver Operating Characteristics Curve.
    
    """
    
    sorted_roc_list = sorted(fpr_tpr_list, key=lambda x: (x[0], x[1]))
    sorted_roc_list = [(0., 0.)] + fpr_tpr_list + [(1., 1.)]

    auc = 0
    for i in range(1, len(sorted_roc_list)):
        print(sorted_roc_list[i - 1])
        print(sorted_roc_list[i])
        print((sorted_roc_list[i][0] - sorted_roc_list[i - 1][0]) * (sorted_roc_list[i][1] + 0.5 * (sorted_roc_list[i - 1][1] - sorted_roc_list[i][1])))
        print(' ')
        auc += (sorted_roc_list[i][0] - sorted_roc_list[i - 1][0]) * (sorted_roc_list[i][1] + 0.5 * (sorted_roc_list[i - 1][1] - sorted_roc_list[i][1]))
    
    return auc
