# params: predicted and actual
# returns true positive, true negative, false positive, false negative
def count_metrics(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    tn = sum(p == a and a == 'decrease' for p, a in zip(predicted, actual))
    fp = sum(p != a and a == 'decrease' for p, a in zip(predicted, actual))
    fn = sum(p != a and a == 'increase' for p, a in zip(predicted, actual))

    return tp, tn, fp, fn  
 
# params: predicted and actual
# returns precision
def precision(predicted, actual, tp=None, fp=None):
    if tp==None: tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    if fp==None: fp = sum(p != a and a == 'decrease' for p, a in zip(predicted, actual))  
    return tp / (tp + fp)

# params: predicted and actual
# returns recall
def recall(predicted, actual, tp=None, fn=None):
    if tp==None: tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    if fn==None: fn = sum(p != a and a == 'increase' for p, a in zip(predicted, actual))  
    return tp / (tp + fn)

# params: predicted and actual labels
# returns the fraction of matching labels
def accuracy(predicted, actual, tp=None, tn=None):
    if tp==None or tn==None: return sum(p == a for p, a in zip(predicted, actual)) / len(actual)
    else: return (tp + tn) / len(actual)

# params: tp, tn, fp, fn, predicted_y, actual_y
# prints metrics
def print_metrics(tp, tn, fp, fn, predicted, actual):
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f"Accuracy: {round(accuracy(predicted, actual, tp=tp, tn=tn) * 100, 2)}%")
    print(f"Recall: {round(recall(predicted, actual, tp=tp, fn=fn) * 100, 2)}%")
    print(f"Precision: {round(precision(predicted, actual, tp=tp, fp=fp) * 100, 2)}%")