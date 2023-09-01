import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_curve, roc_auc_score

#----------------Evaluations------------------------------------------------

def calculate_iou(mask1, mask2):
    # Convert masks to numpy arrays
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    
    # Calculate intersection
    intersection = np.sum(np.logical_and(mask1, mask2))
    
    # Calculate union
    union = np.sum(np.logical_or(mask1, mask2))
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

def calculate_dice(mask1, mask2):
    # Convert masks to numpy arrays
    #mask1 = np.array(mask1)
    #mask2 = np.array(mask2)
    # Calculate intersection
    intersection = np.sum(np.logical_and(mask1, mask2))
    print("intersection:")
    print(intersection)
    print("sums:")
    print(np.sum(mask1))
    print(np.sum(mask2))
    
    # Calculate dice coefficient
    dice = 2 * intersection / (np.sum(mask2) + np.sum(mask2))
    
    return dice
    
def hausdorff_distance(mask1, mask2):
    # Find the coordinates of the non-zero pixels in each mask
    points1 = np.transpose(np.nonzero(mask1))
    points2 = np.transpose(np.nonzero(mask2))

    # Calculate the directed Hausdorff distances between the two sets of points
    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]

    # Return the maximum of the two directed distances
    return max(d1, d2)

def calculate_precision(mask1, mask2):
    # Ensure input masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate true positives
    true_positives = np.logical_and(mask1, mask2).sum()

    # Calculate false positives
    false_positives = np.logical_and(mask1, np.logical_not(mask2)).sum()

    # Calculate precision
    if true_positives + false_positives == 0:
        precision = 1.0
    else:
        precision = true_positives / (true_positives + false_positives)

    return precision

def calculate_precision2(mask, ground_truth):
    # Count the number of true positives and false positives
    true_positives = (mask & ground_truth).sum()
    false_positives = (mask & (~ground_truth)).sum()
    
    # Calculate precision
    if true_positives + false_positives == 0:
        precision = 1.0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    return precision

def calculate_recall(mask1, mask2):
    tp = np.sum(np.logical_and(mask1, mask2))
    fn = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    recall = tp / (tp + fn + 1e-6) # add epsilon to avoid division by zero
    return recall

def calculate_accuracy(mask1, mask2):
    # Convert masks to numpy arrays
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    
    # Calculate accuracy
    accuracy = np.sum(mask1 == mask2) / mask1.size
    
    return accuracy
    
def calculate_accuracy2(mask, ground_truth):
    # Count the number of true positives, true negatives, false positives, and false negatives
    true_positives = (mask & ground_truth).sum()
    true_negatives = ((~mask) & (~ground_truth)).sum()
    false_positives = (mask & (~ground_truth)).sum()
    false_negatives = ((~mask) & ground_truth).sum()
    
    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    return accuracy

def calculate_f1(precision, recall):

    
    # Calculate F1 score
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1
    
def calculate_mse(mask1, mask2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((mask1.astype("float") - mask2.astype("float")) ** 2)
    err /= float(mask1.shape[0] * mask2.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
    
def proportion_of_pixels_changed(gray):
    # Apply Gaussian blur with a 3x3 kernel
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Compute the histogram of the original image
    hist1, _ = np.histogram(gray, bins=256)
    # Compute the histogram of the blurred image
    hist2, _ = np.histogram(blurred, bins=256)
    # Count the number of bins that went from 0 to non-zero
    num_changed = np.count_nonzero(hist1) - np.count_nonzero(hist2)
    # Compute the proportion of pixels that changed
    prop_changed = num_changed / float(gray.size)
    return prop_changed
    

def calculate_roc_curve_auc(predicted_mask, ground_truth_mask):

    # Ensure that the masks have the same shape
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError("The predicted_mask and ground_truth_mask must have the same shape.")

    # Flatten the masks to 1D arrays
    predicted_mask = predicted_mask.ravel()
    ground_truth_mask = ground_truth_mask.ravel()

    # Compute the False Positive Rate (fpr), True Positive Rate (tpr), and Thresholds for ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth_mask, predicted_mask)

    # Compute the Area Under the Curve (AUC)
    auc_score = roc_auc_score(ground_truth_mask, predicted_mask)

    return fpr, tpr, auc_score
