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
    n_feat = 2          # dummy for now
    desired_labels = [0, 1, 2]

    n_classes = 3       # dummy for now

    # print initial values
    print(f'Learning rate = {l_rate}')
    print(f'Bias, w0 = {bias}')
    print(f'Number of Classes = {n_classes}')
    if classes != None:
        for n in range(n_classes): print(f'\t- {n}: {classes[n]}')

    # initialize weights
    c_weights = [[bias] + [0.0]*n_feat] * n_classes

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
        for v in range(n_classes):
            print(f'v{v}', end='\t')
        print('argmax\ty*')

        # go thru all features
        for fv in range(len(f_vectors)):
            print('=============================================')

            # find argmax of the summation of fv * weight_y
            feat = f_vectors[fv]
            desired_y = desired_labels[fv]

            # print and calculate summation of first vector and weights
            max_sum = 0
            argmax = 0
            for x in range(n_feat):
                print(f'{feat[x]:.2f}', end='\t')
            for w in range(n_feat):
                print(f'{c_weights[w]:.2f}', end='\t')
                max_sum += feat[w] * c_weights[0][w]

            # print feature calculations
            print(f'{max_sum:.2f}\t{argmax:.2f}\t{desired_y:.2f}\t')
            
            for c in range(1, n_classes):
                print('-----------------------------------')

                # print and calculate rest of weights and compare to max
                sum = 0
                for x in range(n_feat):
                    print(f'{feat[x]:.2f}', end='\t')
                for w in range(n_feat):
                    print(f'{c_weights[w]:.2f}', end='\t')
                    sum += feat[w] * c_weights[c][w]

                # print feature calculations
                print(f'{sum:.2f}\t{c:.2f}\t{desired_y:.2f}\t')

                # update argmax if necessary
                if sum > max_sum:      
                    max_sum = sum
                    argmax = c

            # check found argmax to desired label
            if argmax != desired_y:
                success = False

                # update weights
                for i in range(n_feat):
                    c_weights[desired_y] += feat[i] * l_rate
                    c_weights[argmax] -= feat[i] * l_rate

            if success == True:
                # successful convergence
                print()
                print('Correct classification (y = d)')
                print('Final Adjusted Weight: ', end='')
                for n in n_classes:
                    print(f'n: {c_weights[n]}')  

                return c_weights

        # failed to converge within max iterations
        print()
        print('Maximum number of iterations reached.')
        print('Possibly non-linearly separable.')
            