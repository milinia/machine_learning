import pygame
from sklearn import svm

def drawing():
    model = ()
    is_gray_point = False
    points = []
    points_classes = [] # 0 - левые, 1 - правые
    colors = ['red', 'blue']
    mouse_position = ()
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # размер окна
    screen.fill(color='#FFFFFF')  # цвет окна
    pygame.display.update()  # без параметров - тот же flip
    flag = True
    while (flag):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.MOUSEMOTION:
                mouse_position = event.pos
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 1 represents the left mouse button
                    coordinates = event.pos
                    points.append(coordinates)
                    points_classes.append(0)
                    pygame.draw.circle(screen, color=colors[0], center=coordinates, radius=10)
                if event.button == 3:  # 3 represents the right mouse button
                    coordinates = event.pos
                    points.append(coordinates)
                    points_classes.append(1)
                    pygame.draw.circle(screen, color=colors[1], center=coordinates, radius=10)
            if event.type == pygame.KEYDOWN:
                if event.key == 13:  # enter
                    if is_gray_point:
                        is_gray_point = False
                        point_class = svm_prediction(model, [mouse_position])
                        pygame.draw.circle(screen, color=colors[point_class[0]], center=mouse_position, radius=10)
                        points.append(mouse_position)
                        points_classes.append(point_class[0])
                    else:
                        (model, ab, c) = svm_study(points, points_classes)
                        intersection_points = find_intersections_with_screen(ab[0][0], ab[0][1], c[0])
                        redraw(points, points_classes, intersection_points, screen)
                if event.key == 32:  # whitespace
                    is_gray_point = True
                    pygame.draw.circle(screen, color='gray', center=mouse_position, radius=10)
            pygame.display.update()

def redraw(points, point_classes, intersection_points, screen):
    screen.fill(color='#FFFFFF')
    colors = ['red', 'blue']
    for i in range(len(points)):
        pygame.draw.circle(screen, color=colors[point_classes[i]], center=points[i], radius=10)
    pygame.draw.line(screen, color='black', start_pos=intersection_points[0],
                     end_pos=intersection_points[1], width=3)

def svm_study(points, points_classes):
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(points, points_classes)

    weights = svm_model.coef_
    intercept = svm_model.intercept_
    return (svm_model, weights, intercept)

def find_intersections_with_screen(a, b, c):
    points = []
    intersection_points = []
    points.append((600, (-600*a - c)/b))
    points.append((0, -c/b))
    points.append(((-400*b - c)/a, 400))
    points.append((-c/a, 0))
    for point in points:
        if 0 <= point[0] <= 600 and 0 <= point[1] <= 400:
            intersection_points.append(point)
    return intersection_points

def svm_prediction(svm_model, new_point):
    return svm_model.predict(new_point)

if __name__ == '__main__':
    drawing()