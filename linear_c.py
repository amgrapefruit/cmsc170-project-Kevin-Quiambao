# Author: Aaron Michael M. Ariedo
# CMSC 170 WX-5l
# Group: Kevin Quiambao
# Implementation from scratch of Linear Classification using Multi-Class Perceptron


### insert dummy data

# params: learning_rate, bias, features and classes
# prints and returns weights for all classes if successful convergence
# else, return -1 
def mc_perceptron(params, l_rate=None, bias=None, f_vectors=None, classes=None):
    # parse parameters if necessary
    # get features
    # get desired labels
    n_feat = 2          # dummy for now, should include bias
    desired_labels = [1, 1, -1]

    # print initial values
    print(f'Learning rate = {l_rate}')
    print(f'Bias, w0 = {bias}')
    print(f'Number of Classes = {2}')       # binary classification
    if classes != None:
        for n in range(2): print(f'\t- {classes[n]}')

    # initialize weights
    weights = [bias] + [0.0]*(n_feat-1)

    MAX_ITERATIONS = 300

    # learning loop
    for iteration in range(MAX_ITERATIONS):
        success = True      # convergence indicator

        # print header
        print()
        print(f'Iteration {iteration+1}')
        for x in range(n_feat):
            print(f'x{x}', end='\t')
        for w in range(n_feat):
            print(f'w{w}', end='\t')
        print('v\ty\td')

        # go thru all features
        for fv in range(len(f_vectors)):
            print('=============================================')

            # get true values
            feat = f_vectors[fv]
            desired_y = desired_labels[fv]

            # print and calculate summation vector and weights
            sum = 0
            for x in range(n_feat):
                print(f'{feat[x]:.2f}', end='\t')
            for w in range(n_feat):
                print(f'{weights[w]:.2f}', end='\t')
                sum += feat[w] * weights[w]

            y = 1.0 if sum > 0 else -1.0
    
            # print feature calculations
            print(f'{sum:.2f}\t{y:.2f}\t{desired_y:.2f}')

            # update weight if necessary
            if y != desired_y:
                success = False

                # update each weight by its corresponding feature
                for index in range(n_feat):
                    weights[index] += feat[index] * l_rate * (desired_labels[fv] - y)

        if success == True:
            # successful convergence
            print()
            print('Correct classification (y = d)')
            print('Final Adjusted Weight: ', end='')
            for w in weights:
                print(f'{w:.2f}', end=' ')
            return weights

    # failed to converge within max iterations
    print()
    print('Maximum number of iterations reached.')
    print('Possibly non-linearly separable.')
            