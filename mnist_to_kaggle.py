# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:37:38 2018

@author: schiejak
"""
#Importing libraries
import tflearn
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

#Function to establish all used variables by means of user input
def get_variables():
    path = input('Input full model path: ')
    train_csv = input('Name of the test .csv file: ')
    test_csv = input('Name of the test .csv file: ')
    test_data = np.load('{}\\test_data.npy'.format(path))
    size = str(input('Input image size: '))
    model = model = tflearn.DNN(sit, checkpoint_path='model_MNIST', 
                    max_checkpoints=1, tensorboard_verbose=0)

    model.load('{}\\mnist_5ep.tflearn'.format(path))


#Opening the training csv file, going line by line to separate
#labels from image data, while one-hot encoding the labels and
#reshaping the image to be able to run it through the network.
#Finally shullfling, and appending the resulting numpy arrays to
#feed list variable and saving it as an .npy to be used later.
def prepare_train_data():
    mnist = open('{}\\{}.csv'.format(path, train_csv), 'r')
    data = mnist.readlines()
    mnist.close()
    feed = []
    for line in data[1:]:
        splitting = line.split(',')
        print(splitting[0])
        curr_label = np.eye(10)[int(splitting[0])]
        makearray = np.asfarray(splitting[1:]).reshape((28,28))
        feed.append([makearray, curr_label])
        shuffle(feed)
        np.save('train_data.npy', feed)


#Similar to the prepare_train_data() function, only this time
#for test data preparation.
def prepare_test_data():
    mnist = open('{}\\{}.csv'.format(path, test_csv), 'r')
    data = mnist.readlines()
    mnist.close()
    test_feed = []
    for line in data[1:]:
        splitting = line.split(',')
        makearray = np.asfarray(splitting).reshape((28,28))
        test_feed.append([makearray])
    np.save('test_data.npy', test_feed)



#Function to plot first couple of images to check how is the
#neural net actually doing
def nastrel():
    
    fig = plt.figure()

    for num, data in enumerate(test_data[:12]):
#    0: [1,0,0,0,0,0,0,0,0,0]
#    1: [0,1,0,0,0,0,0,0,0,0]
#    2: [0,0,1,0,0,0,0,0,0,0]
#    3: [0,0,0,1,0,0,0,0,0,0]
#    4: [0,0,0,0,1,0,0,0,0,0]
#    5: [0,0,0,0,0,1,0,0,0,0]
#    6: [0,0,0,0,0,0,1,0,0,0]
#    7: [0,0,0,0,0,0,0,1,0,0]
#    8: [0,0,0,0,0,0,0,0,1,0]
#    9: [0,0,0,0,0,0,0,0,0,1]
    
        img_num = num + 1
        img_data = data
        
        y = fig.add_subplot(3, 4, img_num)
        data = img_data.reshape(-1, 28, 28, 1)
        prediction = model.predict(data)[0]
        if np.argmax(prediction) == 0:
            label = '0'
        elif np.argmax(prediction) == 1:
            label = '1'
        elif np.argmax(prediction) == 2:
            label = '2'
        elif np.argmax(prediction) == 3:
            label = '3'
        elif np.argmax(prediction) == 4:
            label = '4'
        elif np.argmax(prediction) == 5:
            label = '5'
        elif np.argmax(prediction) == 6:
            label = '6'
        elif np.argmax(prediction) == 7:
            label = '7'
        elif np.argmax(prediction) == 8:
            label = '8'
        else:
            label = '9'
        
        y.imshow(img_data.reshape(28, 28), cmap='gray')
        plt.title(label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        plt.show()
        
        
#Building a kaggle .csv file ready to submit without any
#need for additional editing.
def kaggle_csv():
    with open('kaggle_mnist_submission.csv', 'w') as f:
        f.write('ImageId,Label\n')
        
    with open('kaggle_mnist_submission.csv', 'a') as f:
        for imid, data in enumerate(test_data):
            imid = imid + 1
            prediction = np.argmax(model.predict(data.reshape(-1, size, size, 1)))
            f.write('{},{}\n'.format(imid, prediction))

#Run get_variables() first, and then run functions as needed depending
#on if you want to train the network or use a trained one to evaluate
#results.
get_variables()            