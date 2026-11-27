from knn import predict
from preprocess import preprocess
from evaluation import count_metrics, print_metrics

train, test, raw = preprocess("data/psa_cpi.csv")
X_train, y_train = train
X_test, y_test = test

# Store the results for the model comparison.
knn_results = []

for k in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 51):
    predictions = predict(X_train, y_train, X_test, k=k)

    print(f"k = {k}")
    tp, tn, fp, fn = count_metrics(predictions, y_test)
    print_metrics(tp, tn, fp, fn, predictions, y_test)
    print()
    knn_results.append([k, tp, tn, fp, fn])

# Save the selected results and draw the national inflation graph.
import csv
import os
from datetime import date
from baseline import persistence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("docs/assets/images", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Save the training and test results for the combined comparison graphs.
comparison_results = []
training_predictions = predict(X_train, y_train, X_train, k=11)
train_tp, train_tn, train_fp, train_fn = count_metrics(training_predictions, y_train)
for row in knn_results:
    if row[0] == 11:
        comparison_results.append([
            "k-NN k = 11", train_tp + train_tn, len(y_train),
            row[1], row[2], row[3], row[4]
        ])

with open("results/knn.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "train_correct", "train_total", "tp", "tn", "fp", "fn"])
    writer.writerows(comparison_results)

baseline_predictions = persistence(raw[0])
baseline_counts = count_metrics(baseline_predictions, y_test)
with open("results/baseline.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["tp", "tn", "fp", "fn"])
    writer.writerow(baseline_counts)

# Read national CPI and calculate YoY inflation for the overview graph.
with open("data/psa_cpi.csv", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.reader(file))
for row in rows:
    if len(row) > 1:
        if row[0] == "Geolocation":
            header = row
        if row[0] == "PHILIPPINES" and row[1] == "0 - ALL ITEMS":
            national_row = row

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
months = []
cpi_values = []
for column in range(2, header.index("2026 Sep") + 1):
    label = header[column]
    if not label.endswith(" Ave"):
        year = int(label[:4])
        month = month_names.index(label[5:]) + 1
        months.append(date(year, month, 1))
        cpi_values.append(float(national_row[column]))

dates = []
inflation_rates = []
for month in range(12, len(cpi_values)):
    dates.append(months[month])
    inflation_rates.append((cpi_values[month] / cpi_values[month - 12] - 1) * 100)

plt.figure(figsize=(9, 4))
plt.plot(dates, inflation_rates, color="#176c5b")
plt.axvspan(date(2025, 1, 1), dates[-1], color="#dcebe4", label="Test period")
plt.xlim(dates[0], dates[-1])
plt.xlabel("Month")
plt.ylabel("Year-on-year inflation (%)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.box(False)
plt.savefig("docs/assets/images/inflation-national.png", dpi=160, bbox_inches="tight")
plt.close()
