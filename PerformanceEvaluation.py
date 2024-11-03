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
    
    
    
def false_rate(score, labels, threshold, preds):
    