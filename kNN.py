import math
import sys

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def normalize_data(iris_data, index):
    min = sys.maxsize
    max = -sys.maxsize
    for element in iris_data:
        if element[index] < min:
            min = element[index]
        if element[index] >= max:
            max  = element[index]
    for element in iris_data:
        element[index] = (element[index] - min)/(max - min)
    return (min, max)

def draw_data(iris):
    _, ax = plt.subplots(3,3, figsize=(12, 9))
    for i in range(3):
        for j in range(i + 1, 4):
            scatter = ax[i][j - 1].scatter(iris.data[:, i], iris.data[:, j], c=iris.target)
            ax[i][j - 1].set(xlabel=iris.feature_names[i], ylabel=iris.feature_names[j])
            ax[i][j - 1].legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # получение данных
    iris_data = load_iris()

    # отрисовка данных
    draw_data(iris_data)
    print(iris_data.data)
    max_min = []
    # нормализация данных
    for i in range(4):
        max_min.append(normalize_data(iris_data.data, i))
    draw_data(iris_data)
    # деление данных на выборки
    train_sample = []
    train_target = []
    test_sample = []
    test_target = []
    for i in range(len(iris_data.data)):
        if (i in range(45, 50)) or (i in range(95, 100)) or (i in range(145, 150)):
            test_sample.append(iris_data.data[i])
            test_target.append(iris_data.target[i])
        else:
            train_sample.append(iris_data.data[i])
            train_target.append(iris_data.target[i])
    # алгоритм knn
    accuracy_array = []
    for i in range(int(math.sqrt(len(train_sample)))):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(train_sample, train_target)
        predict = knn.predict(test_sample)
        accuracy_array.append(accuracy_score(test_target, predict))

    k = 0
    for i in range(len(accuracy_array)):
        if accuracy_array[i] > k:
            k = i + 1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_sample, train_target)

    # ввод нового объекта
    try:
        num1 = float(input("Введите длину чашелистика в см: "))
        num2 = float(input("Введите ширину чашелистика в см: "))
        num3 = float(input("Введите длину лепестка в см: "))
        num4 = float(input("Введите ширину лепестка в см: "))

        num1 = (num1 - max_min[0][0]) / (max_min[0][1] - max_min[0][0])
        num2 = (num2 - max_min[1][0]) / (max_min[1][1] - max_min[1][0])
        num3 = (num3 - max_min[2][0]) / (max_min[2][1] - max_min[2][0])
        num4 = (num4 - max_min[3][0]) / (max_min[3][1] - max_min[3][0])
        predict = knn.predict([[num1, num2, num3, num4]])
        iris_classes = ['setosa', 'versicolor', 'virginica']
        print(f"Введенные данные относятся к {iris_classes[predict[0]]}")
    except ValueError:
        print("Пожалуйста, введите только числа от 0 до 1")
