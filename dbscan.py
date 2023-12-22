import pygame
import numpy as np

def drawing():
    points = []
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # размер окна
    screen.fill(color='#FFFFFF')  # цвет окна
    pygame.display.update()  # без параметров - тот же flip
    last_coordinates = (0,0)
    flag = True
    is_drawing = False
    result = []
    while (flag):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                is_drawing = False
                if event.button == 1:
                    is_drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False
            if (is_drawing):
                coordinates = event.pos
                if dist(coordinates, last_coordinates) < 20:
                    continue
                last_coordinates = coordinates
                points.append(coordinates)
                # points.append((coordinates[0], coordinates[1] * (-1)))
                pygame.draw.circle(screen, color='purple', center=coordinates, radius=5)
            if event.type == pygame.KEYDOWN:
                if event.key == 13: #enter
                    result = dbScan(points, 30, 2)
                    show_color_points(result[1], result[2], result[3], screen)
                if event.key == 99: #c
                    print("c")
                    show_cluster_points(result[0], screen)
                if event.key == 27: #esc
                    screen.fill(color='white')
                    points = []
            pygame.display.update()
    return points

def dist(pointA, pointB):
    return np.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

def find_neighbors(radius, center, points):
    neighbors = []
    for point in points:
        if (dist(center, point) < radius) & (dist(center, point) > 0):
            neighbors.append(point)
    return neighbors

def add_unique_neighbors(neighbors, visited_points):
    neighbors_points = set()
    for neighbor in neighbors:
        if neighbor not in visited_points:
            neighbors_points.add(neighbor)
    return neighbors_points
def dbScan(points, radius, num_of_neighbors):
    green_points = []
    yellow_points = []
    red_points = []
    points_with_clusters = []

    neighbors_points = set()
    clusters_num = 0
    visited_points = set()

    while len(visited_points) != len(points):
        for point in points:
            if point in visited_points:
                continue
            else:
                neighbors = find_neighbors(radius, point, points)
                if len(neighbors) >= num_of_neighbors:
                    clusters_num += 1
                    visited_points.add(point)
                    green_points.append(point)
                    points_with_clusters.append((point, clusters_num))
                    neighbors_points.update(add_unique_neighbors(neighbors, visited_points))
                    while neighbors_points:
                        point = neighbors_points.pop()
                        neighbors = find_neighbors(radius, point, points)
                        if len(neighbors) >= num_of_neighbors:
                            visited_points.add(point)
                            green_points.append(point)
                            points_with_clusters.append((point, clusters_num))
                            neighbors_points.update(add_unique_neighbors(neighbors, visited_points))
                        else:
                            if point in red_points:
                                red_points.remove(point)
                            points_with_clusters.append((point, clusters_num))
                            visited_points.add(point)
                            yellow_points.append(point)
                else:
                    if point in red_points:
                        visited_points.add(point)
                    red_points.append(point)
    return points_with_clusters, green_points, yellow_points, red_points

def show_cluster_points(points, screen):
    screen.fill(color='white')
    colors = ['black', 'orange', 'magenta', 'cyan', 'green', 'purple', 'black',
              'pink', 'yellow', 'brown', 'grey', 'red']
    for point in points:
        pygame.draw.circle(screen, color=colors[point[1]], center=point[0], radius=10)

def show_color_points(green_points, yellow_points, red_points, screen):
    screen.fill(color='white')
    dbscan_colors = ['green', 'yellow', 'red']
    for point in green_points:
        pygame.draw.circle(screen, color=dbscan_colors[0], center=point, radius=10)
    for point in yellow_points:
        pygame.draw.circle(screen, color=dbscan_colors[1], center=point, radius=10)
    for point in red_points:
        pygame.draw.circle(screen, color=dbscan_colors[2], center=point, radius=10)

if __name__ == '__main__':
    drawing()

# зажимание ставит точки с каким-то промежутком
# алгоритм раздачи флажков - зеленый, желтый или красный
# отрисовка графиков - один с флагами, другой с разбиванием на кластеры уже после алгоритма

# если есть нужное количество соседей - зеленый
# идет по незакрашенным, если есть сосед зеленый - желтый
# идет по незакрашенным, остальные - красные
