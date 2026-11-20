from knn import predict
from preprocess import preprocess

train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

for k in (3, 5, 7, 9):
    predictions = predict(X_train, y_train, X_test, k=k)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # count the results for increase as the positive class
    for i in range(len(y_test)):
        predicted = predictions[i]
        actual = y_test[i]

        if predicted == "increase" and actual == "increase":
            tp += 1
        elif predicted == "decrease" and actual == "decrease":
            tn += 1
        elif predicted == "increase" and actual == "decrease":
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = (tp + tn) / len(y_test)

    print(f"k = {k}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Accuracy: {accuracy:.2%}")
    print()