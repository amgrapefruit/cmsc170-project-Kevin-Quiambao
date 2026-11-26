from gn_bayes import train_gnbayes, test_gnbayes
from preprocess import preprocess, FEATURES
from evaluation import count_metrics, print_metrics

# fetch data
train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

# train and build model
P_dec, P_inc, means, vars = train_gnbayes(f_vectors=X_train, features=FEATURES, d_labels=y_train)

# test model on validation data
X_test, y_test = test
predictions = test_gnbayes(feat_len=len(FEATURES), P_dec=P_dec, P_inc=P_inc, means=means, variances=vars, test=X_test)

# print training parameters and evaluation metrics
tp, tn, fp, fn = count_metrics(predictions, y_test)
print('Gaussian Naive Bayes Metrics')
print_metrics(tp, tn, fp, fn, predictions, y_test)


# test model on training data
predictions_train = test_gnbayes(feat_len=len(FEATURES), P_dec=P_dec, P_inc=P_inc, means=means, variances=vars, test=X_train)

# print training parameters and evaluation metrics
tp, tn, fp, fn = count_metrics(predictions_train, y_train)
print()
print('Gaussian Naive Bayes Metrics on Training Data')
print_metrics(tp, tn, fp, fn, predictions_train, y_train)