# cmsc170-project-Kevin-Quiambao
Project Repository of Group Kevin Quiambao (Section WX-5L) for CMSC170 Project

## Preprocessing

### Data
`data/psa_cpi.csv` is PSA OpenStat table 2M4ACP22, Consumer Price Index for All Income Households by Commodity Group (2018=100), Jan 2018 to Oct 2026. Selected all geolocations, `0 - ALL ITEMS`, all years and periods. Downloaded Nov 13, 2026.

National year-on-year inflation computed from this file matches PSA's published rates (Dec 2022 = 8.1%, Jan 2023 = 8.7%, Sep 2024 = 1.9%, Jan 2025 = 2.9%, Sep 2025 = 1.7%).

### Task
For each region and month, predict whether year-on-year inflation, `100 * (CPI[t] / CPI[t-12] - 1)`, is higher or lower than the month before.

### Rows kept
- 18 regions (the 17 regions plus Negros Island Region)
- Dropped the national total, provinces and cities, and Areas Outside NCR (it's the other regions combined)
- Months up to Sep 2026, without the yearly "Ave" columns
- Samples with a blank cell (`..`) or an exact tie are skipped

### Features

| # | name | what it is | why |
|---|---|---|---|
| 0–2 | `rate_t3`, `rate_t2`, `rate_t1` | YoY inflation 3, 2 and 1 months before | BSP targets 2–4%, so rates far from that tend to drift back |
| 3–4 | `trend_1`, `trend_2` | change in inflation over the last two months | recent direction |
| 5 | `base` | the target month's month-on-month change, one year earlier | base effect, see below |
| 6 | `recent` | average month-on-month change over the last 3 months | estimate of this month's change |
| 7 | `gap` | `recent - base` | NB and k-NN can't compute this difference themselves |
| 8 | `national` | national month-on-month change last month | fuel, rice and typhoons hit all regions at once |

The base effect comes from this identity:

```
(CPI[t] / CPI[t-12]) / (CPI[t-1] / CPI[t-13])  =  (CPI[t] / CPI[t-1]) / (CPI[t-12] / CPI[t-13])
```

YoY inflation goes up only if this month's price change is bigger than the same month's change last year. Last year's change is already known; this month's has to be estimated, which is what `recent` and `national` are for. The identity held for all 1,612 samples.

No feature uses data from the target month.

### Usage
```python
from preprocess import preprocess, FEATURES

train, test, raw_test = preprocess("data/psa_cpi.csv")
X_train, y_train = train
```
Returns 3 values: scaled train and test, plus the unscaled test set for the baseline. `FEATURES` has the column names in order.

### Baseline
`python baseline.py` prints the sample counts and the persistence baseline (predict the same direction as last month), which gets 58.1% on test.

### Limitations
- Regions move together, so the 375 test samples are closer to ~21 months of evidence. Small gaps between models may be noise.
- Training covers COVID and the 2022–23 spike; the test period is calmer.
- A 0.01-point move gets the same label as a 1-point move.
