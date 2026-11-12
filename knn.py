# Author: Joshua C. otero
# CMSC 170 W-5L
# Group: Kevin Quiambao
# From scratch implementation of K-Nearest Neighbots


# params: training row, test row and distance order
# returns the minkowski distance between the two rows
def minkowskiDistance(training, testing, p):
    # d=(summation of |xi-vi|^p)^1/p
    total = 0
    i = len(testing)

    for j in range(i):
        total += (abs(testing[j] - training[j]))**p

    return total**(1/p)


# params: training features and labels, test features, k and p
# returns an increase or decrease label for each test row
def predict(training_features, training_labels, prediction_features, k, p=2):
    predictions = []

    # go thru all test rows
    for i in range(len(prediction_features)):
        # initialize vote counts
        increase = 0
        decrease = 0

        # get distances and labels for all training rows
        distance = []
        for j in range(len(training_features)):
            # calculate distance to the training row
            d = minkowskiDistance(prediction_features[i], training_features[j], p)

            distance.append((d, training_labels[j]))

        # sort by distance
        distance.sort()

        # count votes from the k nearest neighbors
        for idx in range(k):
            if distance[idx][1] == "increase":
                increase += 1
            else:
                decrease += 1

        # get the label with more votes
        if increase > decrease:
            predictions.append("increase")
        # use the closest neighbor if the votes are tied
        elif increase == decrease:
            predictions.append(distance[0][1])
        else:
            predictions.append("decrease")

    return predictions
