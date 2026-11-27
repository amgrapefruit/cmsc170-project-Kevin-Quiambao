from perceptron import classificationCalc, perceptron
from preprocess import preprocess
from evaluation import count_metrics, print_metrics


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

print(f'Learning rate: {learningRate}, iteration limit: {maxIterations}, bias: {bias}')

# count the results for increase as the positive class
predictions = [
    "increase" if classificationCalc([1.0] + x, weights, threshold)[1] == 1 else "decrease"
    for x in X_test
]
tp, tn, fp, fn = count_metrics(predictions, y_test)
print_metrics(tp, tn, fp, fn, predictions, y_test)


# Evaluate the page's selected setting for the combined comparison graphs.
import csv
import os

os.makedirs("results", exist_ok=True)
plot_weights = perceptron((1.0, 1.0, trainingData, labels), 1000, threshold)
plot_predictions = []
for row in X_test:
    total, prediction = classificationCalc([1.0] + row, plot_weights, threshold)
    plot_predictions.append("increase" if prediction == 1 else "decrease")
plot_tp, plot_tn, plot_fp, plot_fn = count_metrics(plot_predictions, y_test)

training_predictions = []
for row in X_train:
    total, prediction = classificationCalc([1.0] + row, plot_weights, threshold)
    training_predictions.append("increase" if prediction == 1 else "decrease")
train_tp, train_tn, train_fp, train_fn = count_metrics(training_predictions, y_train)

with open("results/perceptron.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "train_correct", "train_total", "tp", "tn", "fp", "fn"])
    writer.writerow(["Perceptron", train_tp + train_tn, len(y_train), plot_tp, plot_tn, plot_fp, plot_fn])
