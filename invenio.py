#-----------importing all necessary modules------------
import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
plt.style.use('dark_background')

#one-hot encoding(turning categorical data into numerical data)

encoder = OneHotEncoder()
encoder.fit([[0],[1]])#fitting values of 0 and 1. 1 = Normal, 0 = Tumor

#--------------creating lists for data----------------
data = []##for loop will add all data from folder to list
paths_yes = []
result = []

#if a tumor is present
#for loop for adding images to paths
for r, d, f in os.walk("C:/Users/ajeer/Downloads/mri2/Testing/pituitary"):
    for file in f:
        if '.jpg' in file:
            paths_yes.append(os.path.join(r,file))
#for loop for transforming image to array
for path in paths_yes:
    img = Image.open(path)#open image
    img = img.resize((128,128))
    img = np.array(img)
    if (img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

paths_no = []
for r, d, f in os.walk("C:/Users/ajeer/Downloads/mri2/Testing/notumor"):
    for file in f:
        if '.jpg'in file:
            paths_no.append(os.path.join(r,file))
for path in paths_no: 
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if (img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())
#creating dataset through arrays
data = np.array(data)
result = np.array(result)
result = result.reshape(559 ,2)
x_train,x_test,y_train,y_test = train_test_split(data,result,train_size = 0.85,test_size = 0.15 , random_state = 0, shuffle = True)#train-test splitting


# Count the number of Tumor and Normal samples
tumor_count = np.sum(result[:, 0] == 1)
normal_count = np.sum(result[:, 1] == 1)

# Create labels and counts for the bar plot
labels = ['Tumor', 'Normal']
counts = [tumor_count, normal_count]

# Example colors that are complementary and visually pleasing
colors = ['skyblue', 'lightcoral']
# Create a bar plot
plt.bar(labels, counts, color=colors)
plt.title('Distribution of Tumor and Normal Samples')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs = 15, batch_size = 60, verbose = 1,validation_data = (x_test, y_test))
evaluation_results = model.evaluate(x_test, y_test)
accuracy = evaluation_results[1]
accuracy = accuracy*100

print(f'Model accuracy = {accuracy}%')
def names(number):
    if number == 0:
        return 'you have a tumor'
    else:
        return 'you dont have a tumor'

from matplotlib.pyplot import imshow
img = Image.open("C:/Users/ajeer/Downloads/mri/brain_tumor_dataset/yes/Y65.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128 ,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% sure ' + names(classification))





