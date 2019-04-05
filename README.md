# mnist-vgg16
Simple implementation of VGG16 on Cifar10 Dataset.

Dataset : MNIST

Dataset of 60,000 28x28 gray scale images of the 10 digits, along with a test set of 10,000 images. I have used 25% of training set as validation set. 
It was not necessary to download the dataset separately from official website and load, as Keras Library already contains entire dataset.

 

Some samples from the dataset

 



Model : CNN (VGG16)

I decided to use VGG16 model (Convolutional Neural Network) to classify Images.

 
Model Summary

Number of Epochs = 50 (because it is observed that accuracy starts converging after 20-25 epochs)
Batch size = 128 (seems to be a good amount of images, not too big, not too small)

 
 
Accuracy 

 
Test Accuracy =  (Number of Correctly Classified Test Samples)/(Total Number of Test Samples)

 
Loss
 
 

It is observed that network converges to Training accuracy of 99.9 % and Validation accuracy of 99.79 % at 50th epoch.

Runtime

 

It only took about 13 minutes to train as I was using GPU.

 

Some incorrect classified samples from the test set

 
We can see, our CNN network has performed really well in terms of accuracy as many of these incorrect classified images can be difficult to classify even for humans. For instance, take 2nd image which is digit 6 but looks more like 0, 3rd image which is 5 but looks like partial 3 or 7th image which is 6 but can easily be confused with 4.

 

