import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model
model = load_model('weather_model.h5')


test_image_cloudy = "D:/test_data/Cloud_test/cloudy299.jpg"  #specify the path to your testing cloudy image here

test_image_sunrise = "D:/Detection_project/sunrise_test.jpg" #specify the path to your testing sunrise image

test_image_rain = "D:/test_data/Rain_test/rain20.jpg" #specify the path to your testing rain image



def print_prediction(test_image):
    prediction = model.predict([prepare(test_image)])
    counter = 0
    for i in (prediction[0]):
        if(counter==0 and i==1.0):
            print("It's CLoudy!")
            img = cv2.imread(test_image)
            plt.imshow(test_image)
            plt.show()
        if(counter==1 and i==1.0):
            print("It's rainy!")
            img = cv2.imread(test_image)
            plt.imshow(img)
            plt.show()
        if(counter==2 and i==1.0):
            print("It's Sunrise!")
            img = cv2.imread(test_image)
            plt.imshow(img)
            plt.show()
        else:
            counter = counter + 1
            
    return prediction[0]



def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) #convert the image to BGR to RGB
    img_array = np.array(img_array)
    new_array = cv2.resize(img_array, (50, 50))  # resize image to match model's expected sizing
    return new_array.reshape(-1, 50, 50, 3)  # ret

    
my_input = input("how many images do you wanna predict?: ")
for i in range(int(my_input)):
    image_path = input("please enter the image path for {}st image, Image should be in jpg: ".format(i+1))
    print_prediction(image_path)








 
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image

