import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def CRR(preds, labels):
    true = np.array(labels)
    pred = np.array(preds)

    return ((true == pred).sum() / len(true)) * 100


#Plots the CRR rate with respect to the number of dimensions 
def plot_CRR_curves(results):
    # Create a figure with 3 subplots, one for each similarity measure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Correct Recognition Rate (CRR) Curves for Different Similarity Measures')
    
    x = np.array(results.index)  # Assuming `results` DataFrame index represents the number of dimensions

    # Plot L1 similarity CRR curves
    axs[0].plot(x, results['crr_l1_reduced'].values, marker='x')
    axs[0].set_title('L1 Similarity')
    axs[0].set_xlabel('Number of Dimensions')
    axs[0].set_ylabel('Correct Recognition Rate (%)')

    # Plot L2 similarity CRR curves
    axs[1].plot(x, results['crr_l2_reduced'].values, marker='x')
    axs[1].set_title('L2 Similarity')
    axs[1].set_xlabel('Number of Dimensions')

    # Plot Cosine similarity CRR curves
    axs[2].plot(x, results['crr_cosine_reduced'].values, marker='x')
    axs[2].set_title('Cosine Similarity')
    axs[2].set_xlabel('Number of Dimensions')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    
    
def print_CRR_tables(CRR_RESULTS):
    for dimension in CRR_RESULTS.index.unique():
        # Filter the DataFrame for the current dimension
        current_dimension_data = CRR_RESULTS.loc[dimension]
        
        # Create a new DataFrame for display purposes
        table = pd.DataFrame({
            "Similarity Measure": ["L1", "L2", "Cosine"],
            "CRR Normal (%)": [
                current_dimension_data['crr_l1_normal'],
                current_dimension_data['crr_l2_normal'],
                current_dimension_data['crr_cosine_normal']
            ],
            "CRR Reduced (%)": [
                current_dimension_data['crr_l1_reduced'],
                current_dimension_data['crr_l2_reduced'],
                current_dimension_data['crr_cosine_reduced']
            ]
        })
        
        # Print the table with a title
        print(f"CRR Rates for Number of Dimensions: {dimension}")
        print(table)
        print("\n" + "="*50 + "\n")  # Separator between tables
        
        
def create_fmr_fnmr_table(thresholds, fmr_list, fnmr_list):
    # Create a DataFrame from the lists
    fmr_fnmr_df = pd.DataFrame({
        "Threshold": thresholds,
        "FMR": fmr_list,
        "FNMR": fnmr_list
    })
    fmr_fnmr_df.set_index("Threshold", inplace=True)
    print(fmr_fnmr_df)

