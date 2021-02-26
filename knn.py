import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint

data = np.genfromtxt("data.csv", delimiter=",")
np.random.shuffle(data)

# calculate the euclidean distance between two points


def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# get the nearest k neighbors to a point


def get_nearest_k_neighbors(train, test_point, k):
    distances = []

    # get the distance between the test point and the rest of the dataset
    for i in range(len(train)):
        dist = calculate_euclidean_distance(test_point, train[i])
        distances.append([train[i], i, dist])

    # sort the distances descending according to the distance
    distances.sort(key=lambda element: element[-1])

    # get the first k neighbors
    return distances[:k]


# predict the label for a single test point
def predict(data, test_point, k, plot=False):
    # get the nearest k neighbors first
    neighbors = get_nearest_k_neighbors(data, test_point, k)
    positives = negatives = 0

    # loop through all neighbors
    for neighbor in neighbors:
        # if the label is 0, increment the negative value
        if neighbor[0][2] == 0:
            negatives += 1
        # if the label is 1, increment the positive value
        else:
            positives += 1

    # plot the data
    if plot:
        plot_data(data, test_point, neighbors[-1][-1])

    # calculate the accuracy of the prediction
    if positives > negatives:
        accuracy = (positives / (positives + negatives))*100
        predicted = 1
    else:
        accuracy = (negatives / (positives + negatives))*100
        predicted = 0

    # send the predicted, actual and the accuracy
    return predicted, int(test_point[-1]), accuracy


# return the total and average accuracy for predicting all cases
def predict_all(data, k):
    positives = negatives = total_accuracy = 0
    # loop through each point
    for test in data:
        # predict the label for each point
        predicted, actual, accuracy = predict(data, test, k, False)
        # increase the total accuracy
        total_accuracy = total_accuracy + accuracy
        # if the prediction was correct increment the positive
        if predicted == actual:
            positives += 1
        # if the prediction was incorrect increment the negative
        else:
            negatives += 1
    # send the total and average accuracy
    return positives * 100 / (positives + negatives), total_accuracy / len(data)


# function to plot the data when needed
def plot_data(data, point, radius):
    plotter = plt.subplots()[1]

    plotter.scatter(data[:, 0], data[:, 1], c=data[:, 2])

    # draw the circle to show the selected pointed, with radius of the maximum distance
    region = mpatches.Circle(point[0:2], radius, color="r", alpha=0.3)
    plotter.add_patch(region)
    plotter.set(aspect=1)

    plt.scatter(point[0], point[1], c="r")
    plt.show()


# function to introduce noise to the dataset according to the dataset
def introduce_noise(data, percent):
    # loop according to 1000 * 10% = 100, this means we will flip only 100 points
    for i in range(int(1000 * (percent/100))):
        # select a randim index
        rand_index = randint(0, 999)
        # flip the label
        if(data[rand_index][-1] == 0):
            data[rand_index][-1] = 1
        else:
            data[rand_index][-1] = 0

    return data

# function to remove tomek links


def remove_tomek_links(data):
    tomek_links = np.array([])
    new_data = []
    # loop through the whole data
    for i in range(len(data)):
        # get the nearest neighbor
        nearest = get_nearest_k_neighbors(data, data[i], 2)[-1]
        # if they have different labels, add their indices to the tomek links array
        if(data[i][-1] != nearest[0][-1]):
            tomek_links = np.append(tomek_links, i)
            tomek_links = np.append(tomek_links, nearest[1])

    # delete duplicate indices
    tomek_links = np.unique(tomek_links)
    # sort
    tomek_links.sort()

    # add all the data, excpet the ones inside tomek links
    for i in range(len(data)):
        if i in tomek_links:
            continue
        new_data.append(data[i])
        
    return np.array(new_data)


def main():
    # Test all examples
    # total_accuracy, average_accuracy = predict_all(data, 25)
    # predicted, actual, accuracy = predict(data,data[randint(0,len(data)-1)],25,True)
    # print(f'Predicted: {predicted}')
    # print(f'Actual: {actual}')
    # print(f'Accuracy: {round(accuracy, 2)}%',end='\n==========================================\n')
    # print(f'Total Accuracy: {total_accuracy}%')
    # print(f'Average Accuracy: {average_accuracy}%')

    # Test prediction after introducing noise
    # noise_data = introduce_noise(data, 20)
    # predicted, actual, accuracy = predict(noise_data,noise_data[randint(0,len(noise_data)-1)],25,True)
    # print(f'Predicted: {predicted}')
    # print(f'Actual: {actual}')
    # print(f'Accuracy: {round(accuracy, 2)}%',end='\n==========================================\n')
    # total_accuracy, average_accuracy = predict_all(noise_data, 25)
    # print(f'Total Accuracy: {total_accuracy}%')
    # print(f'Average Accuracy: {average_accuracy}%')

    # Test after removing tomek links
    noise_data = introduce_noise(data, 50)
    new_data = remove_tomek_links(noise_data)
    # predicted, actual, accuracy = predict(new_data,new_data[randint(0,len(new_data)-1)],15,True)
    # print(f'Predicted: {predicted}')
    # print(f'Actual: {actual}')
    # print(f'Accuracy: {round(accuracy, 2)}%',end='\n==========================================\n')
    total_accuracy, average_accuracy = predict_all(new_data, 15)
    print(f'Total Accuracy: {total_accuracy}%')
    print(f'Average Accuracy: {average_accuracy}%')


main()
