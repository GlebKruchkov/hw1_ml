from skimage import io, color, transform
import os
import numpy as np
import matplotlib.pyplot as plt

X = []
y = []

# parsing images into an array
def load_images(image_folder, value):
    for filename in os.listdir(image_folder):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = io.imread(os.path.join(image_folder, filename))
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = color.rgb2gray(img)
        elif img.ndim == 2:
            pass
        
        img = transform.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(value)

load_images('/Users/glebkruckov/ML/lesson1_dataset/box', 1)
load_images('/Users/glebkruckov/ML/lesson1_dataset/no_box', 0)

X = np.array(X) # array of image arrays
y = np.array(y) # array that shows whether there is a box


class LogisticRegression:
    def __init__(self, w: np.array, b: float):
        self.w = w
        self.b = b

    def predict(self, X):
        return 1 / (np.exp(-X) + 1) # sigmoid
    
def mse(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss

    
model = LogisticRegression(np.zeros(X.shape[1]), 0)
mse_history = []

for i in range(10000):
    dw = np.dot(X.T, (model.predict(np.dot(X, model.w) + model.b) - y)) / X.shape[0]
    db = np.sum(model.predict(np.dot(X, model.w) + model.b) - y) / X.shape[0]

    model.w -= 0.1 * dw # i chose step 0.1, because among all the ones I tested on, it seemed ideal
    model.b -= 0.1 * db

    # link to gradient preference : https://www.overleaf.com/read/mxypxyyrsmkq#b21827

    mse_history.append(mse(y, (model.predict(np.dot(X, model.w) + model.b))))

for image in X:
    prediction = model.predict(np.dot(image, model.w) + model.b)
    if prediction > 0.97:
        print(prediction, end=' ')
        print(" ==> the box is in the picture")
    else:
        print(prediction, end=' ')
        print(" ==> the box is not in the picture")


plt.title('mle history')
plt.plot(mse_history)
plt.show()