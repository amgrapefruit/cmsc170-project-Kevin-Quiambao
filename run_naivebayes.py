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

# Save the existing results, then draw the combined comparison graphs.
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("docs/assets/images", exist_ok=True)
os.makedirs("results", exist_ok=True)

test_tp, test_tn, test_fp, test_fn = count_metrics(predictions, y_test)
with open("results/naivebayes.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "train_correct", "train_total", "tp", "tn", "fp", "fn"])
    writer.writerow(["Naive Bayes", tp + tn, len(y_train), test_tp, test_tn, test_fp, test_fn])

# Run k-NN and perceptron first so their actual results are available.
if (os.path.exists("results/knn.csv") and os.path.exists("results/perceptron.csv")
        and os.path.exists("results/baseline.csv")):
    model_names = []
    model_results = []
    training_accuracy = []
    for filename in ["knn.csv", "naivebayes.csv", "perceptron.csv"]:
        with open("results/" + filename, newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the column headings.
            for row in reader:
                model_names.append(row[0].replace(" k = ", "\nk = "))
                training_accuracy.append(int(row[1]) / int(row[2]) * 100)
                plot_tp = int(row[3])
                plot_tn = int(row[4])
                plot_fp = int(row[5])
                plot_fn = int(row[6])
                accuracy_value = (plot_tp + plot_tn) / (plot_tp + plot_tn + plot_fp + plot_fn) * 100
                precision_value = plot_tp / (plot_tp + plot_fp) * 100
                recall_value = plot_tp / (plot_tp + plot_fn) * 100
                model_results.append([accuracy_value, precision_value, recall_value])

    with open("results/baseline.csv", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        row = next(reader)
        plot_tp, plot_tn, plot_fp, plot_fn = map(int, row)
    baseline_results = [
        (plot_tp + plot_tn) / (plot_tp + plot_tn + plot_fp + plot_fn) * 100,
        plot_tp / (plot_tp + plot_fp) * 100,
        plot_tp / (plot_tp + plot_fn) * 100
    ]

    # Accuracy, precision, and recall comparison graphs.
    metric_names = ["accuracy", "precision", "recall"]
    for metric in range(3):
        values = []
        for result in model_results:
            values.append(result[metric])
        plt.figure(figsize=(9, 4.5))
        plt.barh(model_names, values, color=["#176c5b", "#477ca8", "#a6672d"])
        plt.gca().invert_yaxis()
        plt.xlim(0, 100)
        plt.xlabel(metric_names[metric].capitalize() + " (%)")
        plt.axvline(baseline_results[metric], color="gray", linestyle="--", label="Persistence baseline")
        for model in range(len(values)):
            plt.text(values[model] + 1, model, f"{values[model]:.2f}%", va="center")
        plt.legend()
        plt.grid(axis="x", alpha=0.3)
        plt.box(False)
        plt.savefig("docs/assets/images/comparison-" + metric_names[metric] + ".png",
                    dpi=160, bbox_inches="tight")
        plt.close()

    # Training accuracy versus test accuracy.
    training_positions = []
    test_positions = []
    test_accuracy = []
    for model in range(len(model_names)):
        training_positions.append(model - 0.16)
        test_positions.append(model + 0.16)
        test_accuracy.append(model_results[model][0])

    plt.figure(figsize=(9, 5))
    plt.barh(training_positions, training_accuracy, height=0.3, color="#b8cec6", label="Training")
    plt.barh(test_positions, test_accuracy, height=0.3, color="#176c5b", label="Test")
    plt.yticks(range(len(model_names)), model_names)
    plt.gca().invert_yaxis()
    plt.xlim(0, 100)
    plt.xlabel("Accuracy (%)")
    for model in range(len(model_names)):
        plt.text(training_accuracy[model] + 1, training_positions[model],
                 f"{training_accuracy[model]:.2f}%", va="center", fontsize=10)
        plt.text(test_accuracy[model] + 1, test_positions[model],
                 f"{test_accuracy[model]:.2f}%", va="center", fontsize=10)
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.box(False)
    plt.savefig("docs/assets/images/train-test.png", dpi=160, bbox_inches="tight")
    plt.close()
else:
    print("Run run_knn.py and run_perceptron.py first, then run run_naivebayes.py for the combined graphs.")
