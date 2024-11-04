import numpy as np
import matplotlib.pyplot as plt


def CRR(preds, labels):
    true = np.array(labels)
    pred = np.array(preds)

    return ((true == pred).sum() / len(true)) * 100


#Plots the CRR rate with respect to the number of dimensions 
def plot_CRR_curves(results):
    #L1
    x = np.array(results.index)
    plt.plot(x, results['crr_l1'].values)
    plt.title('CRR Curve (L1)')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()
    #L2
    plt.plot(x, results['crr_l2'].values)
    plt.title('CRR Curve (L2)')
    plt.ylabel('Correct Recognition Rate')
    plt.xlabel('Number of dimensions')
    plt.show()    
    #cosine
    plt.plot(x, results['crr_cosine'].values)
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
    similarity = np.array(similarity)
    labels = np.array(labels)
    preds = np.array(preds)

    # Determine matches based on the threshold
    matches = similarity < threshold

    # Calculate FMR (false positives) where preds match but labels do not
    false_matches = (matches & (labels != preds)).sum()
    total_non_matches = (labels != preds).sum()
    false_match_rate = false_matches / total_non_matches if total_non_matches > 0 else 0

    # Calculate FNMR (false negatives) where labels match but preds do not
    false_non_matches = (~matches & (labels == preds)).sum()
    total_matches = (labels == preds).sum()
    false_non_match_rate = false_non_matches / total_matches if total_matches > 0 else 0

    return false_match_rate, false_non_match_rate

def plot_ROC(fmr, fnmr):
    plt.plot(fmr, fnmr, marker='o', linestyle='-', color='b', label='ROC Curve')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.show()
