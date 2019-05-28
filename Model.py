from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,MaxPool2D,ZeroPadding2D,Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical



dataframe = pd.read_csv('sign_mnist_train.csv') # The dataset file from Kaggle https://www.kaggle.com/datamunge/sign-language-mnist
print(dataframe.head())

y=[]
for i in dataframe['label']:
    y.append(i)
y_data = np.array(y)


y_data=np.reshape(y_data,(27455,1))
y= np.array(y_data)        # Just keeping a track of original labels, will make it easy to convert one-hot to label.

y_data = to_categorical(y_data)



x = []
for i in range(0, 27455):
    temp = []
    for j in range(1, 785):
        temp.append(dataframe[('pixel' + str(j))][i])
    x.append(temp)
x_data=np.array(x)
x=[]



x_data=np.reshape(x_data,(27455,28,28,1))
np.shape(x_data)

np.save('HandGesturetrainX.npy',x_data)
np.save("HandGEsturetrainY.npy",y_data)



#
# x_data= np.load('HandGesturetrainX.npy')
# y = np.load('HandGEsturetrainY.npy')
# print(np.shape(x_data),np.shape(y))


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data)
print(np.shape(x_train),np.shape(y_train),np.shape(x_test),np.shape(y_test))

# Normalizing the input

x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)



###############################################################################

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.50))
model.add(Dense(25, activation = 'softmax'))



model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train,batch_size=200,validation_data=(x_test,y_test),epochs=20)

###################################################################################


###################################################################################
''' An example of converting predicted output from one-hot to true label'''
x= x_data[0]

predictions = model.predict(x)
prediction = predictions[0]

alpha = np.zeros((1,25))
# print(np.argmax(prediction))
alpha[0][np.argmax([prediction])] = 1

c=0
index=0
flag = True
for i in y_data:
    i = list(i)

    if i == list(alpha[0]):
        # print('Found at:')
        # print(c)
        index = c
        flag = False
        break
    c += 1
if flag:
    print('NOT FOUND')


y_label = y[index]    # Original label at index

print(y_label)







