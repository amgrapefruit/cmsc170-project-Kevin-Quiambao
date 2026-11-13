from preprocess import preprocess


# params: predicted and actual labels
# returns the fraction of matching labels
def accuracy(predicted, actual):
    return sum(p == a for p, a in zip(predicted, actual)) / len(actual)


# params: raw (unscaled) feature rows
# returns the direction of the most recent change for each row
def persistence(X_raw):
    # row[1] and row[2] are rate_t2 and rate_t1
    return ["increase" if row[2] > row[1] else "decrease" for row in X_raw]


if __name__ == "__main__":
    train, validation, test, raw_test = preprocess("data/psa_cpi.csv")

    # sample counts and class balance per split
    for name, (X, y) in [("train", train), ("validation", validation), ("test", test)]:
        print(f"{name}: {len(y)} samples, {y.count('increase')} increase, {y.count('decrease')} decrease")

    X_raw, y_raw = raw_test
    print("persistence accuracy:", round(accuracy(persistence(X_raw), y_raw), 4))