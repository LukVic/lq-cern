import sys
sys.path.append("/home/lucas/Documents/KYR/bc_thesis/thesis/project/utils/")
from ml_utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score
import logging
import pandas as pd


def calculate_significance_all_thresholds_new_method(predicted_probs_y: np.array, true_y: np.array,
                                                     weights: np.array, other_bgr: float, test_set_scale_factor: float) \
        -> Tuple[List[float], List[float], List[float], List[float], List[float], float]:
    """ Obtain characteristics for simplified significance maximization threshold tuning
    @param predicted_probs_y: Predicted probabilities
    @param true_y: Ground truth labels
    @param scale_factors: Weighting factors used for comparison with existing results
    @param N_events: Number of events in real data used for comparison
    @param other_bgr: Number of events in other background in real data used for comparison
    @return: Graph data, best threshold
    """
    x_values = list()
    y_S = list()
    y_B = list()
    y_signif = list()
    y_signif_simp = list()
    y_signif_imp = list()
    max_sig = 0
    best_th = 0
    threshold_start = 0
    logger = logging.getLogger()
    for th in np.round(np.arange(threshold_start, 1, 0.01), 2):
        th = np.round(th, 3)
        x_values.append(th)
        if th % 0.2 == 0:
            logger.info('Threshold: {}'.format(th))
        y_pred = calculate_class_predictions_basedon_decision_threshold(predicted_probs_y, th)
        response = calculate_significance_one_threshold_new_method(true_y, y_pred, weights, other_bgr,
                                                                   test_set_scale_factor)
        y_S.append(response['S'])
        y_B.append(response['B'])
        y_signif.append(response['significance'])
        y_signif_simp.append(response['significance_simple'])
        y_signif_imp.append(response['significance_improved'])

        if response['significance_improved'] > max_sig:
            max_sig = response['significance_improved']
            best_th = th
    return x_values, y_S, y_B, y_signif, y_signif_simp, y_signif_imp, best_th


def calculate_class_predictions_basedon_decision_threshold(predicted_probs_y: np.array, threshold: float):
    """ Outputs an array with predicted classes based on predicted class probabilities and decision threshold
    If the predicted probability for class 0 is greater than threshold, the predicted class is 0,
    otherwise it is max P(y|x) among the remaining classes
    @param predicted_probs_y: Predicted probabilities
    @param threshold: Decision threshold
    @return: Predicted classes

    """

    length = np.shape(predicted_probs_y)[0]
    width = np.shape(predicted_probs_y)[1]
    y_pred = list()
    for i in range(length):
        if predicted_probs_y[i, 0] >= threshold:
            y_pred.append(0)
        else:
            predicted_class = 0
            max_p = 0
            for j in range(1, width):
                if max_p < predicted_probs_y[i, j]:
                    max_p = predicted_probs_y[i, j]
                    predicted_class = j
            y_pred.append(predicted_class)
    return y_pred

def calculate_significance_one_threshold_new_method(true_y: np.array, predicted_y: np.array, weights: np.array,
                                                    other_background=0, \
                                                    test_set_scale_factor: float = 1):
    """ Approximate significance score for prediction.
        Used to compare with known results for given weights, number of events, and background compensation constant.
    @param true_y: Ground truth labels
    @param predicted_y: Predicted probabilities
    @param weights: Event weights
    @param other_background: Other backgrounds constants
    @param verbose: Turn on displaying information about computation
    @return: List containing efficiencies, S, B, and two significances (norml & simplified)
    """

    cm = confusion_matrix(true_y, predicted_y, sample_weight=weights)
    cm_len = np.shape(cm)[0]
    signal_total = cm[0, 0] #0,0
    background_total = 0
    for i in range(1, cm_len): # no -1
        background_total += cm[i, 0]


    S = signal_total * test_set_scale_factor
    B = background_total * test_set_scale_factor
    B += other_background
    if B < 1:
        S = 0
        B = 1

    signif = S / np.sqrt(S + B)
    signif_simple = S / np.sqrt(B)
    signif_improved = S / (np.sqrt(B) + 3/2)
    if signif_simple == float('+inf'):
        signif_simple = 0
    result = {'S': S,
              'B': B,
              'significance': signif,
              'significance_simple': signif_simple,
              'significance_improved': signif_improved}
    return result

def plot_threshold(x_values, y_values, optimums, title, ylabel, colors, labels, savepath: str,
                   force_threshold_value=None):
    """ Plots graphs for threshold characteristics.
    @param x_values: Values for x-axis (list)
    @param y_values: Values for y-axis (list of lists)
    @param optimums: Whether to search max or min values in y_values
    @param title: Title of plot
    @param colors: Colors for y_values
    @param labels: Labels for y_values
    @param force_threshold_value: Value of forced vertical line value (if None, vertical line is computed with usage of
    optimums parameter)
    """
    best_score_final = 0
    plt.figure()
    for values, optimum, color, label in zip(y_values, optimums, colors, labels):
        plt.plot(x_values, values, color=color, label=label, linewidth=2)

        if force_threshold_value is None:
            best_index = 0
            if optimum == 'max':
                best_score = 0
            else:
                best_score = max(values)
            for i, v in enumerate(values):
                if optimum == 'max':
                    if best_score < v:
                        best_score = v
                        best_index = i
                else:
                    if best_score > v:
                        best_score = v
                        best_index = i
        else:
            best_index = int(force_threshold_value * 100)
            best_score = values[best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        plt.plot([x_values[best_index], ] * 2 , [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        plt.annotate("%0.3f" % best_score,
                    (x_values[best_index], best_score + 0.005))
        if best_score > best_score_final:
            best_score_final = best_score

    plt.xticks(np.arange(0.0, 1.0, step=0.1))
    plt.xlabel('Threshold')
    plt.ylabel(ylabel)
    # if len(y_values) == 2:
    #     plt.yscale('log')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(savepath)
    # plt.show()
    # plt.close()
    return best_score_final
#----------------------OPTUNA-----------------------------------------------------------------------------------------------------------------
def threshold_value(x_values, y_values, optimums, force_threshold_value=None):
    """ Plots graphs for threshold characteristics.
    @param x_values: Values for x-axis (list)
    @param y_values: Values for y-axis (list of lists)
    @param optimums: Whether to search max or min values in y_values
    @param force_threshold_value: Value of forced vertical line value (if None, vertical line is computed with usage of
    optimums parameter)
    @param optimums: Whether to search max or min values in y_values
    @return: the highest significance value
    """
    for values, optimum in zip(y_values, optimums):

        if force_threshold_value is None:
            best_index = 0
            if optimum == 'max':
                best_score = 0
            else:
                best_score = max(values)
            for i, v in enumerate(values):
                if optimum == 'max':
                    if best_score < v:
                        best_score = v
                        best_index = i
                else:
                    if best_score > v:
                        best_score = v
                        best_index = i
        else:
            best_index = int(force_threshold_value * 100)
            best_score = values[best_index]
        
    return best_score
