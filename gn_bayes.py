# Author: Aaron Michael M. Ariedo
# CMSC 170 WX-5l
# Group: Kevin Quiambao
# Implementation from scratch of Gaussian Naive-Bayes Classifier

from preprocess import preprocess, FEATURES         # for fetching training data
from baseline import accuracy          # for testing weights
from decimal import Decimal
import math         # for log and pi

# params: training data (features, labels, classes)
# returns 2 class_num x feat_num tables model, and probabilities given class
def train_gnbayes(*params, f_vectors=None, features=None, d_labels=None, classes=['decrease', 'increase']):
    feat_num = len(features)
    class_num = len(classes)            # decrease / increase
    train_len = len(f_vectors)

    # probability of each class
    dec_count = d_labels.count('decrease')
    inc_count = d_labels.count('increase')
    P_decrease = dec_count / train_len
    P_increase = inc_count / train_len

    # get average per class per feature
    averages = [[0.0 for f in range(feat_num)] for c in range(class_num)]
    for vector_i in range(train_len):
        i = 1 if d_labels[vector_i] == 'increase' else 0        # 0: decrease, 1: increase

        for j in range(feat_num):
            averages[i][j] += f_vectors[vector_i][j]         # total sum

    for j in range(feat_num):
        averages[0][j] = averages[i][j] / dec_count
        averages[1][j] = averages[i][j] / inc_count
            
    # get variance per class per feature
    variances = [[0.0 for f in range(feat_num)] for c in range(class_num)]
    for vector_i in range(train_len):
        i = 1 if d_labels[vector_i] == 'increase' else 0        # 0: decrease, 1: increase

        for j in range(feat_num):
            variances[i][j] += (f_vectors[vector_i][j] - averages[i][j]) ** 2       # total sum

    for j in range(feat_num):
        variances[0][j] = variances[i][j] / dec_count
        variances[1][j] = variances[i][j] / inc_count

    return P_decrease, P_increase, averages, variances

# params: test data and model (P_dec, P_inc, means, vars)
# returns predictions
def test_gnbayes(*params, feat_len=0, P_dec=0.5, P_inc=0.5, means=None, variances=None, test=None):
    predictions = []

    # go through the whole testing data
    for t in test:
        # get score (log score for precision and speed)
        dec_score = math.log(P_dec)
        inc_score = math.log(P_inc)

        for j in range(feat_len):
            dec_score += -0.5 * math.log(2 * math.pi * variances[0][j])  -  ((t[j] - means[0][j])**2)/(2 * variances[0][j])
            inc_score += -0.5 * math.log(2 * math.pi * variances[1][j])  -  ((t[j] - means[1][j])**2)/(2 * variances[1][j])

        pred = 'increase' if inc_score > dec_score else 'decrease'
        predictions.append(pred)

    return predictions

# params: predicted and actual
# returns true positive, true negative, false positive, false negative
def count_metrics(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    tn = sum(p == a and a == 'decrease' for p, a in zip(predicted, actual))
    fp = sum(p != a and a == 'decrease' for p, a in zip(predicted, actual))
    fn = sum(p != a and a == 'increase' for p, a in zip(predicted, actual))

    return tp, tn, fp, fn  
 
# params: predicted and actual
# returns precision
def precision(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    fp = sum(p != a and a == 'decrease' for p, a in zip(predicted, actual))  
    return tp / (tp + fp)

# params: predicted and actual
# returns recall
def recall(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    fn = sum(p != a and a == 'increase' for p, a in zip(predicted, actual))  
    return tp / (tp + fn)

# fetch data
train, test, raw_test = preprocess("data/psa_cpi.csv")
X_train, y_train = train

# train and build model
P_dec, P_inc, means, vars = train_gnbayes(f_vectors=X_train, features=FEATURES, d_labels=y_train)

# test model
X_test, y_test = test
predictions = test_gnbayes(feat_len=len(FEATURES), P_dec=P_dec, P_inc=P_inc, means=means, variances=vars, test=X_test)


# print training parameters and evaluation metrics
tp, tn, fp, fn = count_metrics(predictions, y_test)
print('Gaussian Naive Bayes Metrics')
print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
print(f"Accuracy: {round(accuracy(predictions, y_test) * 100, 2)}%")
print(f"Recall: {round(recall(predictions, y_test) * 100, 2)}%")
print(f"Precision: {round(precision(predictions, y_test) * 100, 2)}%")