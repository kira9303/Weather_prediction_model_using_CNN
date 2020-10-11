import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
#import tensorflow.keras.utils.to_categorical as to_cat
from tensorflow.keras.utils import to_categorical

DATADIR = "D:/Detection_project"

#CATEGORIES = ["Cloudy", "Rain", "Sunrise"]

CATEGORIES = ["Cloudy", "Rain", "Sunrise"]

counter = 1

counter_new = 1

new_counter = 0

test_image_cloudy = "D:/test_data/CLoud_test/cloudy301.jpg"

test_image_sunrise = "D:/test_data/Sunrise_test/sunrise12.jpg"

test_image_rain = "D:/test_data/Rain_test/rain19.jpg"

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  # iterate over each image per class
        if(counter==3):
            break
        img_array = cv2.imread(os.path.join(path,img))
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)# convert to RGB
        #print(img_rgb)
        img_rgb = np.array(img_rgb)
        print(img_rgb)
        plt.imshow(img_rgb)  
        plt.show()  # display!
        
        counter = counter  + 1
        
    break  



IMG_SIZE = 50

def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_array)
    new_array = cv2.resize(img_array, (50, 50))  # resize image to match model's expected sizing
    return new_array.reshape(-1, 50, 50, 3)  # ret
    #return new_array



training_data = []
output_empty = [0] * len(CATEGORIES)

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
               
                new_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                #new_array = np.array(new_array)
                
                output_row = list(output_empty)
    
                output_row[(class_num)] = 1
                #adding the output row to the training set
                training_data.append([new_array, output_row])  # add this to our training_data
                #new_counter = new_counter + 1
                
                
                plt.imshow(img_rgb)
                plt.show
                print("This is class_num for above image {}".format(class_num))
                    
                
                
                
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))



import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[0])
    print(sample[1])
    
X = []
y = []

#train_x = list(training_data[:,0])
#print(train_x)
#train_y = list(training_data[:,1])

for features,label in training_data:
    X.append(features)
    y.append(label)
    
#print("train_X is: ")
#print(train_x)

#print("train_y is: ")
#print(train_y)
    
X = np.array(X)
y = np.array(y)



print("X here is: ")
print(X)

print("Y here is:   ")
print(y)

print("here is the first element od X: {}".format(X[0]))

print(" \n")

print("here is the first element of Y: {}".format(y[0]))

print("here is the shape of the first element of X:  ")

first_x = X[0]

shape_x = X.shape
print(shape_x)

print("shape of entire X is:   {}".format(first_x.shape))

y_shape = y[0]

print("shape of y is: -   {}".format(y_shape.shape))


    
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X = X/255.0


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3)))
#model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
#model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2,2)))
#model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))

#model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

#model.add(Dropout(0.2))
# output layer
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


hist = model.fit(X, y, batch_size=32, epochs=30, validation_split=0.3, verbose=1)

#prediction1 = model.predict([prepare(test_image_cloudy)])


print("shape of prepared_img is: ---   ")









#model.save('weatehr_model', save_format='h5')
model.save('weather_model.h5', hist)


