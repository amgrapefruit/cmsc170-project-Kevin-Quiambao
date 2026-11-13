# Author: Aaron Michael M. Ariedo
# CMSC 170 WX-5l
# Group: Kevin Quiambao
# Implementation from scratch of Linear Classification using Multi-Class Perceptron

from preprocess import preprocess, FEATURES         # for fetching training data
from baseline import accuracy          # for testing weights
from decimal import Decimal

# params: learning_rate, bias, features, vectors, d_labels, and classes
# prints and returns weights for all classes if successful convergence
# else, return -1 
def perceptron(*params, l_rate=None, bias=None, f_vectors=None, features=None, d_labels=None, classes=None):
    # new working vectors with bias and decimals instead of floats
    working_fv = [[Decimal(str('1.0'))] + [Decimal(str(x)) for x in fv] for fv in f_vectors]

    # parse desired labels
    desired_labels = [Decimal('1.0') if row == 'increase' else Decimal('-1') for row in d_labels]

    l_rate = Decimal(str(l_rate))
    
    # find number of features
    n_feat = len(features) + 1              # includes bias

    # print initial values
    # print(f'Learning rate = {l_rate}')
    # print(f'Bias, w0 = {bias}')
    # print(f'Number of Classes = {2}')       # binary classification
    # if classes != None:
    #     for n in range(2): print(f'\t- {classes[n]}')
    # print(f'Features = {n_feat-1}')
    # for f in features: print(f'\t- {f}')

    # initialize weights
    weights = [Decimal(str(bias))]+[Decimal(0)]*(n_feat-1)

    MAX_ITERATIONS = 2000

    # learning loop
    for iteration in range(MAX_ITERATIONS):
        success = True      # convergence indicator

        # print header
        # print()
        # print(f'Iteration {iteration+1}')
        # for x in range(n_feat):
        #     print(f'x{x}', end='\t')
        # for w in range(n_feat):
        #     print(f'w{w}', end='\t')
        # print('v\ty\td')

        # go thru all features
        for fv in range(len(working_fv)):
            # print('=============================================')

            # get true values
            feat = working_fv[fv]
            desired_y = desired_labels[fv]

            # print and calculate summation vector and weights
            sum = 0
            # for x in range(n_feat):
            #     print(f'{feat[x]:.2f}', end='\t')
            for w in range(n_feat):
                # print(f'{weights[w]:.2f}', end='\t')
                sum += feat[w] * weights[w]

            y = Decimal('1.0') if sum >= 0 else Decimal('-1.0')
    
            # print feature calculations
            # print(f'{sum:.2f}\t{y:.2f}\t{desired_y:.2f}')

            # update weight if necessary
            if y != desired_y:
                success = False

                # update each weight by its corresponding feature
                for index in range(n_feat):
                    weights[index] += feat[index] * l_rate * (desired_labels[fv] - y)

        if success == True:
            # successful convergence
            print(f'Iterations: {iteration}')
            print('Correct classification (y = d)')
            print('Final Adjusted Weights: ', end='')
            for w in weights:
                print(f'{w:.2f}', end=' ')
            print()
            return weights, iteration

    # failed to converge within max iterations
    print()
    print(f'Maximum number of iterations ({MAX_ITERATIONS}) reached.')
    print('Possibly non-linearly separable.')
    print('Last Adjusted Weights: ', end='')
    for w in weights:
        print(f'{w:.2f}', end=' ')
    print()
    return weights, MAX_ITERATIONS

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
def precision(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    fp = sum(p != a and a == 'decrease' for p, a in zip(predicted, actual))  
    return tp / (tp + fp)

# params: predicted and actual
# returns recall
def recall(predicted, actual):
    tp = sum(p == a and a == 'increase' for p, a in zip(predicted, actual))
    fn = sum(p != a and a == 'increase' for p, a in zip(predicted, actual))  
    return tp / (tp + fn)

# fetch data
train, test, raw_test = preprocess("data/psa_cpi.csv")
X_train, y_train = train

l_rate = 1.0
bias = 1.0
weights, iterations = perceptron(l_rate=l_rate, bias=bias, f_vectors=X_train, features=FEATURES, d_labels=y_train)

# predict with test data
X_test, y_test = test
predictions = []

for x in X_test:
    total = 0

    for i in range(len(FEATURES)):
        total += Decimal(str(x[i])) * weights[i+1]

    total += Decimal('1.0') * weights[0]          # bias

    if total > Decimal('0'): predictions.append('increase')
    else: predictions.append('decrease')

# print training parameters and evaluation metrics
tp, tn, fp, fn = count_metrics(predictions, y_test)
print(f'Learning rate: {l_rate}, Bias (w0): {bias}, Iterations: {iterations}')
print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
print(f"Accuracy: {round(accuracy(predictions, y_test), 4) * 100}%")
print(f"Recall: {round(recall(predictions, y_test), 4) * 100}%")
print(f"Precision: {round(precision(predictions, y_test), 4) * 100}%")