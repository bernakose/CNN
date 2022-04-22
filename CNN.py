# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 00:58:30 2022

@author: Berna
"""

#Kütüphaneleri yükle
import keras
import numpy as np
import matplotlib.pyplot as plt

#Veri setimizi yükleyelim
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
#Veri setinin boyutları
#Eğitim setinin boyutu
print(train_images.shape)
#Test setinin boyutu
print(test_images.shape)

#Modelimizi oluşturalım
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten


model=Sequential()
#1. Convolutional layer
model.add(Conv2D(64,kernel_size=3,activation="relu",input_shape=(28,28,1)))
# 2. Convolutional Layer
model.add(Conv2D(32,kernel_size=3,activation="relu"))

#Flatten Layer
model.add(Flatten())

#Tam Bağlı Katman 
model.add(Dense(10,activation="softmax"))

#Modelimizi inceleyelim.
model.summary()
#Modeli derle

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_images=train_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)

#Özellik ölçeklendirme yapalım
train_images=train_images/255.0
test_images=test_images/255.0

#Önce modeli eğitip sonra test etme
model.fit(train_images,train_labels,epochs=3)
test_loss,test_accuracy=model.evaluate(test_images,test_labels)

#Daha önce eğitilen modelin ağırlıklarını yükleme
model.load_weights("agirliklar2022.h5")

#Modeli hem eğitip hem test etme
histories=model.fit(train_images,train_labels,epochs=3, validation_data=(test_images,test_labels))

#Grafik çizme
plt.figure()
plt.title("Sınıflandırma Doğruluk Grafiği")
plt.plot(histories.history['accuracy'],color="red",   label='eğitim')
plt.plot(histories.history['val_accuracy'],color="blue",  label='doğrulama')
plt.legend()
plt.show()

# Yeni bir resimi sınıflandırma.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

img = load_img("sample_image.png", color_mode="grayscale", target_size=(28, 28))

img = img_to_array(img)
img = img.reshape(-1, 28, 28, 1)

#img = img.astype('float32')
img = img / 255.0

class_name=['Tişört / üst', 'Pantalon','Kazak','Elbise','Ceket','Sandalet', 'Gömlek','Spor Ayakkabı','çanta', 'Çizme']

result= np.argmax(model.predict(img), axis=-1)
print("Sonuc :")
print(result[0])

d=result[0]
print(class_name[d])