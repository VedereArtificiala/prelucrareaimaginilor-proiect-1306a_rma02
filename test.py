import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

#load model
model = tf.keras.models.load_model('D:\Facultate\PIM\Workspace\PIM Proiect\my_model')
print("Model loaded")

#load dataset
y_test=pd.read_csv("D:\Facultate\PIM\Workspace\PIM Proiect\input\Test.csv")
labels=y_test['Path'].to_numpy()
y_test1=y_test['ClassId'].values

data=[]
height = 30
width = 30

image = cv2.imread('D:\Facultate\PIM\Workspace\PIM Proiect\input\Test\\00024.png')
#"D:\Facultate\PIM\Workspace\PIM Proiect\input\Test\00000.png"
image_from_array = Image.fromarray(image, 'RGB')
size_image = image_from_array.resize((height, width))
data.append(np.array(size_image))

print("Images loaded")
X_test=np.array(data)
X_test = X_test.astype('float32')/255



pred = model.predict(X_test)
print(pred)
print(np.argmax(pred))
from sklearn.metrics import accuracy_score
#accuracy_score(y_test, pred)


#print(mean_squared_error(y_test, pred))