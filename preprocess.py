import csv
import decimal


# column order of each feature row
FEATURES = [
    "rate_t3", "rate_t2", "rate_t1",
    "trend_1", "trend_2",
    "base", "recent", "gap", "national",
]


# params: one csv cell
# returns a Decimal, or None if the cell is blank or ".." (unreleased)
def to_decimal(value):
    try:
        return decimal.Decimal(value.strip())
    except (decimal.InvalidOperation, AttributeError):
        return None


# params: a cpi series and two month positions
# returns the percent change from month j to month k, or None if either is missing
def pct_change(cpi, k, j):
    if cpi[k] is None or cpi[j] is None:
        return None
    return 100 * (cpi[k] / cpi[j] - 1)


# params: one split and the training means and standard deviations
# returns the scaled feature rows and their labels
def scale(split, averages, std_devs):
    X, y = split
    scaled = []

    for row in X:
        scaled_row = []
        for i in range(len(row)):
            scaled_row.append((row[i] - averages[i]) / std_devs[i])
        scaled.append(scaled_row)

    return scaled, y


# params: path to the psa csv
# returns the scaled training, validation and test sets, plus the raw test set
def preprocess(path):
    # each split stores feature rows and labels
    train = ([], [])
    validation = ([], [])
    test = ([], [])

    with decimal.localcontext() as ctx:
        ctx.prec = 40

        with open(path, encoding="utf-8-sig", newline="") as file:
            # psa exports either comma or semicolon csvs
            text = file.read()
            delimiter = ";" if text.count(";") > text.count(",") else ","
            rows = list(csv.reader(text.splitlines(), delimiter=delimiter))

        header = next(row for row in rows if row and row[0] == "Geolocation")

        # keep monthly columns through september 2026
        # skip the yearly averages so 12 columns back is always the same month last year
        last_month = header.index("2026 Sep")
        months = []
        for i in range(2, last_month + 1):
            if not header[i].endswith(" Ave"):
                months.append(i)

        all_items = [
            row for row in rows
            if len(row) == len(header) and row[1] == "0 - ALL ITEMS"
        ]

        # national cpi, used as a feature for every region
        national_row = next(row for row in all_items if row[0] == "PHILIPPINES")
        national = [to_decimal(national_row[i]) for i in months]

        for row in all_items:
            # keep regions and skip national totals, provinces and cities
            # areas outside ncr is just the other regions combined, so it's skipped too
            region = row[0]
            if not region.startswith("..") or region.startswith("...."):
                continue
            if "Areas Outside" in region:
                continue

            cpi = [to_decimal(row[i]) for i in months]

            # k is the target month, starting where every feature has data
            for k in range(15, len(cpi)):
                # year-on-year inflation, same as how psa reports it
                rate_now = pct_change(cpi, k, k - 12)
                rates = [pct_change(cpi, j, j - 12) for j in (k - 3, k - 2, k - 1)]

                # month-on-month changes
                base = pct_change(cpi, k - 12, k - 13)
                monthly = [pct_change(cpi, j, j - 1) for j in (k - 3, k - 2, k - 1)]
                national_last = pct_change(national, k - 1, k - 2)

                # skip if any month needed is missing
                if None in [rate_now, base, national_last] + rates + monthly:
                    continue

                # compare before rounding and skip ties
                if rate_now > rates[2]:
                    label = "increase"
                elif rate_now < rates[2]:
                    label = "decrease"
                else:
                    continue

                # inflation level for the last 3 months
                # bsp targets 2-4%, so rates far from that tend to drift back
                rate_t3, rate_t2, rate_t1 = [float(rate) for rate in rates]

                # whether inflation has been going up or down lately
                trend_1 = rate_t1 - rate_t2
                trend_2 = rate_t2 - rate_t3

                # base effect
                # (cpi[t] / cpi[t-12]) / (cpi[t-1] / cpi[t-13])
                #   = (cpi[t] / cpi[t-1]) / (cpi[t-12] / cpi[t-13])
                # so inflation goes up only if this month's change beats the
                # same month's change last year, which we already know
                base = float(base)

                # this month's change is unknown, so use the last 3 months as a guess
                recent = float(sum(monthly) / 3)

                # naive bayes and knn can't subtract features on their own
                gap = recent - base

                # fuel, rice and typhoons hit all regions at once
                # only last month's national value, since this month's isn't out yet
                features = [
                    rate_t3, rate_t2, rate_t1,
                    trend_1, trend_2,
                    base, recent, gap, float(national_last),
                ]

                # split using the target month so we never train on the future
                year = int(header[months[k]][:4])
                if year <= 2023:
                    X, y = train
                elif year == 2024:
                    X, y = validation
                else:
                    X, y = test

                X.append(features)
                y.append(label)

    # get the mean and standard deviation of each training feature
    averages = []
    std_devs = []

    for column in zip(*train[0]):
        average = sum(column) / len(column)
        variance = sum((value - average)**2 for value in column) / len(column)
        std_dev = variance**0.5

        averages.append(average)
        std_devs.append(std_dev or 1.0)

    # use the training scaling values for all three splits
    # raw test set is returned too for the baseline
    return (
        scale(train, averages, std_devs),
        scale(validation, averages, std_devs),
        scale(test, averages, std_devs),
        test,
    )