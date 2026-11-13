# Author: Aaron Michael M. Ariedo
# CMSC 170 WX-5l
# Group: Kevin Quiambao
# Implementation from scratch of Linear Classification using Multi-Class Perceptron

from preprocess import preprocess, FEATURES
from decimal import Decimal

### insert dummy data

# params: learning_rate, bias, features, vectors, d_labels, and classes
# prints and returns weights for all classes if successful convergence
# else, return -1 
def perceptron(*params, l_rate=None, bias=None, f_vectors=None, features=None, d_labels=None, classes=None):
    # new working vectors with bias and decimals instead of floats
    working_fv = [[Decimal(str(bias))] + [Decimal(str(x)) for x in fv] for fv in f_vectors]

    # parse desired labels
    desired_labels = [Decimal('1.0') if row == 'increase' else Decimal('-1') for row in d_labels]

    l_rate = Decimal(str(l_rate))
    
    # find number of features
    n_feat = len(features) + 1              # includes bias

    # print initial values
    print(f'Learning rate = {l_rate}')
    print(f'Bias, w0 = {bias}')
    print(f'Number of Classes = {2}')       # binary classification
    if classes != None:
        for n in range(2): print(f'\t- {classes[n]}')
    print(f'Features = {n_feat-1}')
    for f in features: print(f'\t- {f}')

    # initialize weights
    weights = [Decimal(0)]*(n_feat)

    MAX_ITERATIONS = 10000000

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

            y = Decimal('1.0') if sum > 0 else Decimal('-1.0')
    
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
            print('Final Adjusted Weight: ', end='')
            for w in weights:
                print(f'{w:.2f}', end=' ')
            return weights

    # failed to converge within max iterations
    print()
    print('Maximum number of iterations reached.')
    print('Possibly non-linearly separable.')


# fetch data

train, validation, test, raw_test = preprocess("data/psa_cpi.csv")
X_train, y_train = train

l_rate = 1.0
bias = 1.0
weights = perceptron(l_rate=l_rate, bias=bias, f_vectors=X_train, features=FEATURES, d_labels=y_train)