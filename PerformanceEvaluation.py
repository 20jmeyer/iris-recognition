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
    
    
def false_rate(score, labels, threshold, preds):
    pass