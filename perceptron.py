# Author: Joshua C. otero
# CMSC 170 W-5L
# Group: Kevin Quiambao

def classificationCalc(row, weights, threshold):
    # x0 is 1 and w0 is the bias
    a = 0.0
    for i in range(len(row)):
        a += row[i] * weights[i]

    if a >= threshold:
        prediction = 1
    else:
        prediction = -1

    return a, prediction


# params: learning rate, bias, training rows with x0 and numeric labels
# returns the learned weights
def perceptron(inputData, maxIterations=100, threshold=0.0):
    learningRate, bias, trainingData, labels = inputData
    featureCount = len(trainingData[0]) - 1

    # set w0 to the bias and the other weights to 0
    weights = [bias]
    for i in range(featureCount):
        weights.append(0.0)

    # go thru all training rows in each iteration
    for iteration in range(maxIterations):
        changed = False

        for i in range(len(trainingData)):
            row = trainingData[i]
            target = labels[i]
            a, prediction = classificationCalc(row, weights, threshold)
            diff = target - prediction

            # update weights when the prediction is wrong
            if diff != 0:
                for j in range(len(weights)):
                    # wa = wcurrent + rx*(d-y)
                    weights[j] += learningRate * row[j] * diff
                changed = True

        # stop when a whole pass finishes without any updates
        if not changed:
            break

    # use the final weights even when the iteration limit is reached
    return weights
