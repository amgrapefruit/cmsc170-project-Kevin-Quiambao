from gn_bayes import train_gnbayes, test_gnbayes, count_metrics, precision, recall
from preprocess import preprocess, FEATURES
from baseline import accuracy          # for testing weights

# fetch data
train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

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