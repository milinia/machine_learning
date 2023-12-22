import random
import sys
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def random_points(n):
    point = []
    for i in range(n):
        point.append(Point(random.randint(0, 100), random.randint(0, 100)))
    return point

def show_points(points):
    for elem in points:
        plt.scatter(elem.x, elem.y, color='g')
    plt.draw()
    plt.show()

def show_color_points(points, centroids):
    colors = ['b', 'orange', 'magenta', 'cyan', 'g', 'purple', 'black', 'pink', 'y', 'brown', 'grey']
    for point in points:
        plt.scatter(point[0].x, point[0].y, color=colors[point[1]])
    for centroid in centroids:
        plt.scatter(centroid.x, centroid.y, color='r')
    plt.draw()
    plt.show()

def dist(pointA, pointB):
    return np.sqrt((pointA.x - pointB.x)**2 + (pointA.y - pointB.y)**2)

def first_centroids(points, n): #n - кол-во кластеров
    pointCenter = Point(0, 0)
    for elem in points:
        pointCenter.x += elem.x
        pointCenter.y += elem.y
    pointCenter.x /= len(points)
    pointCenter.y /= len(points)
    max_dist = 0 #это радиус окружности
    for elem in points:
        localDist = dist(elem, pointCenter)
        if (localDist > max_dist):
            max_dist = localDist
    centroids = []
    for i in range(n):
        centroids.append(Point(max_dist * np.cos(2 * np.pi * i / n) + pointCenter.x,
                         max_dist * np.sin(2 * np.pi * i / n) + pointCenter.y
                         ))
    for i in range(n):
        plt.scatter(centroids[i].x, centroids[i].y, color='r')
    return centroids

def divide_into_clusters(centroids, points):
    new_points = []
    for point in points:
        min = sys.maxsize
        num = 0
        for i in range(len(centroids)):
            point_dist = dist(point, centroids[i])
            if point_dist < min:
                min = point_dist
                num = i
        new_points.append((point, num))
    return new_points

def calculate_new_centroids_location(centroids, points):
    sum_x = [0] * len(centroids)
    sum_y = [0] * len(centroids)
    k = [0] * len(centroids)
    for point in points:
        sum_x[point[1]] += point[0].x
        sum_y[point[1]] += point[0].y
        k[point[1]] += 1
    for i in range(len(centroids)):
        centroids[i].x = sum_x[i] / k[i]
        centroids[i].y = sum_y[i] / k[i]
    return centroids

def is_centroids_location_change(prev_centroids, new_centroids):
    for i in range(len(prev_centroids)):
        if ((prev_centroids[i].x != new_centroids[i].x)
                | (prev_centroids[i].y != new_centroids[i].y)):
            return True
    return False

def calculate_j(centroids, points):
    sum = 0
    for point in points:
        sum += dist(point[0], centroids[point[1]])**2
    return sum

def k_means_for_n_clusters(n, is_plot_showing):
    centroids = first_centroids(points, n)
    if is_plot_showing:
        show_points(points)
    new_points = divide_into_clusters(centroids, points)
    if is_plot_showing:
        show_color_points(new_points, centroids)
    temp = copy.deepcopy(centroids)
    centroids = calculate_new_centroids_location(centroids, new_points)

    while (is_centroids_location_change(temp, centroids) == True):
        new_points = divide_into_clusters(centroids, points)
        if is_plot_showing:
            show_color_points(new_points, centroids)
        temp = copy.deepcopy(centroids)
        centroids = calculate_new_centroids_location(centroids, new_points)
    return calculate_j(centroids, new_points)

def calculate_min_num_of_clusters(j):
    min = sys.maxsize
    num = 0
    for i in range(1, len(j) - 1):
        d = abs(j[i] - j[i+1]) / abs(j[i - 1] - j[i])
        if (d < min):
            min = d
            num = i
    return num

if __name__ == '__main__':
    points = random_points(100)
    j = [0] * int(np.sqrt(100))
    for i in range(1, int(np.sqrt(100)) + 1):
        j[i - 1] = k_means_for_n_clusters(i, False)

    plt.figure(figsize=(10, 6))
    plt.plot(j, marker='o', linestyle='-', color='b')
    plt.grid(True)
    plt.show()

    num = calculate_min_num_of_clusters(j)
    k_means_for_n_clusters(num, True)

    # data = [[point.x, point.y] for point in points]
    # kmeans = KMeans(n_clusters=num, n_init='auto').fit(data)
    # centers = kmeans.cluster_centers_
    # for point in points:
    #     plt.scatter(point.x, point.y, color='g')
    # plt.scatter(centers[:, 0], centers[:, 1], color='r')
    # plt.show()