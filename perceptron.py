import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os 
import random

def sigmoid(z):
    return (1.0/(1.0 + np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def reveal(self):
        return self.biases, self.weights
    
    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.matmul(w, a) + b)
        z = np.matmul(self.weights[-1], a) + self.biases[-1]
        se = self.sumofexp(z)
        a = np.multiply(np.exp(z), 1 / se)
        return a

    def train(self, data, lrate, lmda):
        epochs = 10
        itr = epochs * len(data)
        eta = lrate
        for k in range (0, epochs):
            random.shuffle(data)
            for x, y in data:
                grad_b, grad_w = self.backprop(x, y)
                self.weights = [w * (1 - eta * lmda) - eta * gw for w, gw in zip(self.weights, grad_w)]
                self.biases = [b - eta * gb for b, gb in zip(self.biases, grad_b)]
                eta = np.max([eta - (lrate / (itr)), 0])
        
    def backprop(self, x, pre_y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        y = np.array(pre_y).reshape((10, 1))
        activation = np.array(x).reshape((784, 1))
        activations = [activation]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(np.array(w), np.array(activation)) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        se = self.sumofexp(zs[-1])
        finaloutput = np.multiply(np.exp(z), 1 / se)
        activations[-1] = finaloutput

        error = np.multiply(self.costDerivative(activations[-1], y), 1)

        grad_b[-1] = error
        grad_w[-1] = np.matmul(error, np.array(activations[-2]).transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            error = np.multiply(np.matmul(self.weights[-l+1].transpose(), error), sp)
            grad_b[-l] = error
            grad_w[-l] = np.matmul(error, np.array(activations[-l-1]).transpose())
        
        return grad_b, grad_w

    def costDerivative(self, outActivation, y):
        return (np.add(outActivation, np.multiply(y, -1)))

    def sumofexp(self, a):
        b = np.exp(a)
        return np.sum(b)

def vOneAt(j):
    vec = np.zeros((10, 1))
    vec[j] = 1.0
    return vec

def preprocesscustomimg(img):
    #function to preprocess custom image (number written on white paper)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _ , img = cv.threshold(img, 80, 255, cv.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    img = (255 - img)
    img = cv.dilate(img, kernel, iterations = 3)
    img = (cv.resize(img, [28, 28]))
    return img

def preprocessimage(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, [28, 28])
    return img

def loadData(root):
    images = []
    y = []
    for i in range(0, 10):
        folder = root + "/" + str(i)
        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(preprocessimage(img))
                y.append(vOneAt(i))
    return images, y

def test(network, root):
    correct = 0
    total = 0
    y = []
    for i in range(0, 10):
        folder = root + "/" + str(i)
        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                img = preprocessimage(img)
                img = np.array(img).flatten()
                img = np.multiply(img, 1.0/255.0)
                img = np.array(img).reshape((784, 1))
                ans = network.feedforward(img)
                if(np.argmax(ans) == i):
                    correct = correct + 1
                total = total + 1
    return (correct / total , correct, total)

def shuffle(a, b):
    k = np.array(b).shape[0]
    p = np.random.permutation(k)
    return a[p], b[p]


def dataprep(images):
    finaldata = []
    for arr in images:
        arrimg = np.array(arr).flatten()
        arrimg = np.multiply(arrimg, 1.0/255.0)
        finaldata.append(arrimg)
    return finaldata

images, y = loadData('train')
images = np.array(images, dtype = np.float32)
y = np.array(y)
images, y = shuffle(images, y)
train_data = dataprep(images)
train_data = np.array(train_data)
final_train_data = [(img, x) for img, x in zip(train_data, y)]

network = Network([784, 16, 16, 10])
network.train(final_train_data, 0.6, 0.0007)