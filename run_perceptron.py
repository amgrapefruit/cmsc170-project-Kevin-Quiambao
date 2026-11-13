from perceptron import classificationCalc, perceptron
from preprocess import preprocess


train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

learningRate = 0.1
bias = 0.5
maxIterations = 1000
threshold = 0.0

# add x0 and convert the labels to 1 and -1
trainingData = []
labels = []
for i in range(len(X_train)):
    trainingData.append([1.0] + X_train[i])
    if y_train[i] == "increase":
        labels.append(1)
    else:
        labels.append(-1)

inputData = (learningRate, bias, trainingData, labels)
weights = perceptron(inputData, maxIterations, threshold)

# count the results for increase as the positive class
tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(X_test)):
    row = [1.0] + X_test[i]
    a, prediction = classificationCalc(row, weights, threshold)
    actual = y_test[i]

    if prediction == 1 and actual == "increase":
        tp += 1
    elif prediction == -1 and actual == "decrease":
        tn += 1
    elif prediction == 1 and actual == "decrease":
        fp += 1
    else:
        fn += 1

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
accuracy = (tp + tn) / len(y_test)

print(f"Learning rate: {learningRate}, iteration limit: {maxIterations}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"Accuracy: {accuracy:.2%}")
