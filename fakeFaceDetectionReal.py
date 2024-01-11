import os
import cv2

import numpy as np

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

path = "data"

myList=os.listdir(path)

noOfClasses=2
#len(myList)

print("Label (sınıf) sayısı : ",noOfClasses)

images=[]
labels=[]

size=64
color=1

epochs=16

batch_size=32


myImageList=os.listdir(path+"\\"+"training_fake")

for i in myImageList:
    img=cv2.imread(path+"\\training_fake\\"+i)
    img=cv2.resize(img,(size,size))
    images.append(img)
    labels.append("fake")
    
myImageList=os.listdir(path+"\\"+"training_real")

for i in myImageList:
    img=cv2.imread(path+"\\training_real\\"+i)
    img=cv2.resize(img,(size,size))
    images.append(img)
    labels.append("real")
    
    
# print(len(images))
# print(len(labels))

images=np.array(images)
labels=np.array(labels)

# print(images.shape)
# print(labels.shape)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

labels = onehot_labels(labels)

# veriyi ayırma
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=42)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.2,random_state=42)

# print(images.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(x_validation.shape)

# preprocess
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    
    return img

x_train=np.array(list(map(preProcess,x_train)))
x_test=np.array(list(map(preProcess,x_test)))
x_validation=np.array(list(map(preProcess,x_validation)))

x_train=x_train.reshape(-1,size,size,color)
x_test=x_test.reshape(-1,size,size,color)

x_validation=x_validation.reshape(-1,size,size,color)

# # data generate
dataGen=ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            rotation_range=10)

dataGen.fit(x_train)

# y_train=to_categorical(y_train,noOfClasses)
# y_test=to_categorical(y_test,noOfClasses)
# y_validation=to_categorical(y_validation,noOfClasses)

# %% Fully Connected Layer

model=Sequential()
model.add(Conv2D(input_shape=(size,size,color), filters=8,kernel_size=(5,5), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu",padding="same"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu",padding="same"))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation="softmax"))

model.compile(loss="binary_crossentropy",optimizer=("Adam"),metrics=["accuracy"])

dataGen=ImageDataGenerator()

hist=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batch_size),
                         validation_data=(x_validation,y_validation),
                         epochs=epochs,steps_per_epoch=x_train.shape[0]//batch_size,shuffle=1)

# pickle_out=open("model_trained_new.p","wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()

model.save("my_model")
model.save_weights("weights.h5")

score_train = model.evaluate(x_train,y_train)
print("Eğitim doğruluğu : %",score_train[1]*100)

score_test = model.evaluate(x_test,y_test)
print("Test doğruluğu : %",score_test[1]*100)

score_validation = model.evaluate(x_validation,y_validation)
print("Validation doğruluğu : %",score_validation[1]*100)

# %% Matplotlib..
hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()


# score = model.evaluate(x_test, y_test, verbose = 1)
# print("Test loss: ", score[0])
# print("Test accuracy: ", score[1])


y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
Y_true = np.argmax(y_validation, axis = 1)
cm = confusion_matrix(Y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")  
plt.title("cm")
plt.show()

""