import numpy as np
import matplotlib.pyplot as plt
def CRR(preds, labels):
    correctly_predicted = 0
    n = len(labels)
    
    for pred, label in zip (preds, labels):
        if pred == label:
            correctly_predicted +=1
    
    CRR = np.round((correctly_predicted/n)*100)
    return CRR
#Plots the CRR rate with respect to the number of dimensions 
def plot_CRR_curves(results):
    #L1
    plt.plot(results['dimensions'], results['crr_l1'])
    plt.title('CRR Curve (L1)')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()
    #L2
    plt.plot(results['dimensions'], results['crr_l2'])
    plt.title('CRR Curve (L2)')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()    
    #cosine
    plt.plot(results['dimensions'], results['crr_cosine'])
    plt.title('CRR Curve (Cosine)')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()
    
    
    
def false_rate(similarity, labels, threshold, preds):
    """
    Calculates False Match Rate (FMR) and False Non-Match Rate (FNMR) based on a threshold.

    Parameters:
    - similarity: list of float, similarity scores for each pair.
    - labels: list of int, actual labels.
    - threshold: float, threshold below which pairs are considered a match.
    - preds: list of int, predicted labels.

    Returns:
    - Tuple of calculated rates: (false_match_rate, false_non_match_rate)
    """
    # Initialize counts for False Positives, True Positives, True Negatives, and False Negatives
    FP, TP, TN, FN = 0, 0, 0, 0

    # Process each pair to classify as TP, FP, TN, or FN
    for i in range(len(similarity)):
        is_match = similarity[i] < threshold  # Determine if similarity score indicates a match
        actual = labels[i]                    # Actual label
        predicted = preds[i]                  # Predicted label

        if is_match:
            if predicted == actual:
                TP += 1  # True Positive: correct match
            else:
                FP += 1  # False Positive: incorrect match
        else:
            if predicted == actual:
                TN += 1  # True Negative: correct non-match
            else:
                FN += 1  # False Negative: incorrect non-match

    # Calculate False Match Rate and False Non-Match Rate
    false_match_rate = FP / (TP + FP) if (TP + FP) > 0 else 0
    false_non_match_rate = FN / (TN + FN) if (TN + FN) > 0 else 0

    return false_match_rate, false_non_match_rate

def plot_ROC(fmr, fnmr):
    plt.plot(fmr, fnmr, marker='o', linestyle='-', color='b', label='ROC Curve')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.show()