import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import cv2

def drawing(model):
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  # размер окна
    screen.fill(color='white')  # цвет окна
    pygame.display.update()  # без параметров - тот же flip
    flag = True
    is_drawing = False
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
                pygame.draw.circle(screen, color='black', center=coordinates, radius=7)
            if event.type == pygame.KEYDOWN:
                if event.key == 13:  # enter
                    pygame.image.save(screen, "number.jpg")
                    determine_number(model)
                if event.key == 27:  # esc
                    screen.fill(color='white')
            pygame.display.update()

def prepare_train_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def prepare_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return test_loader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Преобразование входных данных в плоский вектор
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model():
    train_loader = prepare_train_data()
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    epochs = 20
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
    model.eval()
    return model

def divide_number_into_digits():
    image = cv2.imread('number.jpg')
    is_negative = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > h: #знак минуса
            is_negative = True
        digit_roi = image[y:y + h, x:x + w]
        cv2.imwrite(f'digit_{i}.png', digit_roi)
    return (len(contours), is_negative)

def determine_number(model):
    number_of_digits, is_negative = divide_number_into_digits()
    number = []
    start = 0
    if is_negative:
        start = 1
    for i in range(start, number_of_digits):
        num = determine_digit(model, f'digit_{i}.png')
        number.append(num)
    print_number(number, is_negative)

def print_number(number, is_negative):
    if is_negative:
        print("-", end="")
    for num in number:
        print(num, end="")
    print("")

def add_margin(img, top, right, bottom, left, color):
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result

def download_image(image_name):
    image = Image.open(image_name).convert('L')
    image = add_margin(image, 42, 42, 42, 42, 'white')
    image = np.array(image)
    image = (255 - image) / 255.0
    image = Image.fromarray((image * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)
    return input_image

def determine_digit(model, image_name):
    image = download_image(image_name)

    image_array = image.numpy()
    image_array = np.reshape(image_array, (28, 28))
    plt.imshow(image_array, cmap='gray')
    plt.show()
    # print(image)
    with torch.no_grad():
        output = model(image)
    ps = torch.exp(output)
    probab = list(ps.numpy()[0])
    return probab.index(max(probab))

def evaluate_model(model):
    train_loader = prepare_test_data()
    correct_count = 0
    all_count = 0
    for images, labels in train_loader:
        for i in range(len(labels)):
            img = images[i]
            # print(img)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1
    print("Accuracy =", (correct_count / all_count))

def test_model(model):
    for i in range(10):
        dataiter = iter(prepare_test_data())
        images, labels = dataiter.__next__()
        # plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
        # plt.show()
        with torch.no_grad():
            output = model(images)
        ps = torch.exp(output)
        probab = list(ps.numpy()[0])
        print(probab.index(max(probab)))

if __name__ == '__main__':
    # model = train_model()
    # torch.save(model.state_dict(), 'simple_model.pth')
    model = NeuralNetwork()
    model.load_state_dict(torch.load('simple_model.pth'))
    model.eval()
    # test_model(model)
    evaluate_model(model)
    drawing(model)