# mnist-vgg16
Simple implementation of VGG16 on MNIST Dataset using Keras.

### Dataset : MNIST

Dataset of 60,000 28x28 gray scale images of the 10 digits, along with a test set of 10,000 images. I have used 25% of training set as validation set. 
It was not necessary to download the dataset separately from official website and load, as Keras Library already contains entire dataset.

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

### Some samples from the dataset

![image](https://user-images.githubusercontent.com/46073809/55661859-77b37680-57c3-11e9-9fa0-0fdf07201e0b.png)

### Model : CNN (VGG16)

I decided to use VGG16 model (Convolutional Neural Network) to classify Images.

![image](https://user-images.githubusercontent.com/46073809/55661883-9c0f5300-57c3-11e9-92b4-e125efb76785.png)

 
### Model Summary

Number of Epochs = 50 (because it is observed that accuracy starts converging after 20-25 epochs)

    nb_epoch = 50

Batch size = 128 (seems to be a good amount of images, not too big, not too small)

    batch_size = 128

![image](https://user-images.githubusercontent.com/46073809/55661899-c06b2f80-57c3-11e9-8cb7-f8e339ce1d02.png)
 
### Accuracy

    Test accuracy: 99.21 %
 
Test Accuracy =  (Number of Correctly Classified Test Samples)/(Total Number of Test Samples)

![image](https://user-images.githubusercontent.com/46073809/55661934-fdcfbd00-57c3-11e9-875c-d5de2b65b941.png)
 
### Loss

![image](https://user-images.githubusercontent.com/46073809/55661943-1049f680-57c4-11e9-9b95-99d3133fbab9.png)

    Epoch 50/50
    45000/45000 [==============================] - 15s 331us/step - loss: 0.0123 - acc: 0.9990 - val_loss: 0.0329 - val_acc: 0.9979

It is observed that network converges to Training accuracy of 99.9 % and Validation accuracy of 99.79 % at 50th epoch.

### Runtime

    CPU times: user 10min 13s, sys: 1min 55s, total: 12min 8s
    Wall time: 12min 31s

It only took about 13 minutes to train as I was using GPU.

    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

### Some incorrect classified samples from the test set

![image](https://user-images.githubusercontent.com/46073809/55661982-82bad680-57c4-11e9-8ad2-50223f0b77f3.png)
 
We can see, our CNN network has performed really well in terms of accuracy as many of these incorrect classified images can be difficult to classify even for humans. For instance, take 2nd image which is digit 6 but looks more like 0, 3rd image which is 5 but looks like partial 3 or 7th image which is 6 but can easily be confused with 4.
