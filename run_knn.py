from knn import predict
from preprocess import preprocess
from evaluation import count_metrics, print_metrics

train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

for k in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 51):
    predictions = predict(X_train, y_train, X_test, k=k)

    print(f"k = {k}")
    tp, tn, fp, fn = count_metrics(predictions, y_test)
    print_metrics(tp, tn, fp, fn, predictions, y_test)
    print()