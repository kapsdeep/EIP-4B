# EIP-4B
CIFAR10 training problem

# Problem statement
#### (Initial)Deep Network(56) trained on small dataset(60k) resulted in underfitting & low training accuracy(~76%)
* _training speed: good_
* _training accuracy: low_
* _dataset: small (60k images, 10 classes)_
* _layers =56_

## Strategy to beat challenge

| Plan        | AIs          | Comments  |
  | ------------- |:-------------:| -----:|
  | Revise notes from session#1-#4 & note important points      | Feature extraction, Normalization, Channels, Regularization, Data Augmentation, Hyperparameter tuning, Dropout, Optimizers  | Try in parallel tracks |
  | Revise hyperparamter tuning notes from coursera      | underfit=>shallow n/w, more data, low train_acc=>deep n/w, more data, different arch      |   tackle separately |
  | Keep an on MLBLR discussion forum | compression, image resize, start with simpler n/w      |    didn't understand compression, try img resize in last |
| Start with multiple tracks to imrpove train/val accuracy | 1. Data augmentation, 2. Hyperparamter tuning, 3. Deep n/w 4. Network Arch     |    Merge best of individual tracks |

## Solution summary
* Deep network:  Itdid hit train_acc upwards of 87% but was too slow to train (>8hrs) hence not the best solution
* Data Augmentation: Started with max augmentation methods but was hit with & network ended val_acc=0.18@epoch#50 :( but eventually figured out that more augmentation causes regularization & all augmented images might not be helpful hence figured out min augmentation which did show improvement so combined it with other Network Arch track
* Network Arch: shuffling was a great boost & combined with l=2 (calculated from receptive field) was the major breakthrough pushing the network accuracy upwards of 88% in just 2.5hrs !! Reduced layers allowed to set #filters=128 & batch_size=64....bingo !!! combined with data augmentation....val_acc=.9032% :) time taken(sec): 7290.32763671875

_Also hit another model with early stopping at epoch#41 with validation accuracy of .9028% but decided to submit below one with 0.9032 at epoch50_

## Track#1 - Network Arch
### trial#1
* compression = 0.7
* batch_size = 32
* shuffle = True

#### Analysis
shuffle=true makes network learn faster

### trial#2
* compression = 1
* batch_size = 32
* shuffle = True

#### Results
* Epoch 18/20 50000/50000 [==============================] - 384s 8ms/step - loss: 0.4003 - acc: 0.8612 - val_loss: 0.5075 - val_acc: 0.8409

#### Analysis 
Network learning well

### trial#3
_adding time & graphs_
* compression = 1
* batch_size = 32
* shuffle = True
* l = 12
* epoch =50

#### Results
* Epoch 10/50 - loss: 0.5427 - acc: 0.8106 - val_loss: 0.6204 - val_acc: 0.7964
* Epoch 20/50 - loss: 0.3767 - acc: 0.8687 - val_loss: 0.5512 - val_acc: 0.8264
* Epoch 30/50 - loss: 0.3009 - acc: 0.8950 - val_loss: 0.5790 - val_acc: 0.8364
* Epoch 40/50 - loss: 0.2513 - acc: 0.9112 - val_loss: 0.6085 - val_acc: 0.8302
* Epoch 50/50 - loss: 0.2154 - acc: 0.9232 - val_loss: 0.4695 - val_acc: 0.8721

#### Analysis
* network training slowed down drastically from epoch#30 to #50
* slowed training rate also caused n/w to overfit a bit

### trial#4 
* enable bias in both dense & transition blocks and increase dropout to address any overfitting at later stages
* increase filters to allow n/w to learn more features without increasing n/w density
* batch_size = 32
num_classes = 10
epochs = 50
l = 40
num_filter = 16
compression = 1
dropout_rate = 0.25

#### Results

* Epoch 00010: val_acc improved from -inf to 0.76120, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.76120 to 0.85190, saving model to weights_best.hdf5
* Epoch 00030: val_acc improved from 0.85190 to 0.85380, saving model to weights_best.hdf5
* Epoch 00040: val_acc did not improve from 0.85380
* Epoch 00050: val_acc improved from 0.85380 to 0.87670, saving model to weights_best.hdf5
* Epoch 20/50 50000/50000 [===] - 450s 9ms/step - loss: 0.2982 - acc: 0.8952 - val_loss: 0.4824 - val_acc: 0.8519
* Epoch 30/50 50000/50000 [===] - 450s 9ms/step - loss: 0.2170 - acc: 0.9235 - val_loss: 0.5197 - val_acc: 0.8538
* Epoch 40/50 50000/50000 [===] - 450s 9ms/step - loss: 0.1693 - acc: 0.9393 - val_loss: 0.6405 - val_acc: 0.8387
* Epoch 50/50 50000/50000 [===] - 452s 9ms/step - loss: 0.1394 - acc: 0.9510 - val_loss: 0.4760 - val_acc: 0.8767


#### Analysis
* Best training accuracy - 0.9510 (epoch#50: traninig accuracy 0.9232@trial#3 => 0.9510@trial#4)
* Final validation accuracy improved marginally 0.8721=> 87.67
* After epoch#20, validation accuracy started lagging behind indicating overfitting
_Mistakenly dropout was set to 0.2, correcting to 0.25_

### trial#5
Objective: Regularize network AND maintain/push train_accuracy
* maintaining/pushing train_accuracy: l=12, channels=20
* Regularize to address overfit: data augmentation, bias, dropout
* data augmentation: width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
* bias: True for all (dense, transition, FirstConv)
* dropout: 0

#### Results
* Epoch 00010: val_acc improved from -inf to 0.81160, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.81160 to 0.84130, saving model to weights_best.hdf5
* Epoch 00030: val_acc improved from 0.84130 to 0.88700, saving model to weights_best.hdf5
* Epoch 00040: val_acc improved from 0.88700 to 0.89210, saving model to weights_best.hdf5
* Epoch 00050: val_acc improved from 0.89210 to 0.89950, saving model to weights_best.hdf5

* Epoch 10/50 1562/1562 [===] - 491s 314ms/step - loss: 0.4538 - acc: 0.8432 - val_loss: 0.5676 - val_acc: 0.8116
* Epoch 20/50 1562/1562 [===] - 490s 314ms/step - loss: 0.2871 - acc: 0.9002 - val_loss: 0.4959 - val_acc: 0.8413
* Epoch 30/50 1562/1562 [===] - 490s 314ms/step - loss: 0.2045 - acc: 0.9301 - val_loss: 0.3672 - val_acc: 0.8870
* Epoch 40/50 1562/1562 [===] - 490s 314ms/step - loss: 0.1538 - acc: 0.9450 - val_loss: 0.3773 - val_acc: 0.8921
* Epoch 50/50 1562/1562 [===] - 494s 316ms/step - loss: 0.1269 - acc: 0.9547 - val_loss: 0.3748 - val_acc: 0.8995

#### Analysis
* dropout=0 combined with better data augmentation did help regularization
* network did marginally better on training accuracy

### trial#7
Receptive field concept to arrive #layers - https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807 
```python
import math
import numpy as np

convnet=np.array([[2,1,0],[3,1,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]])
layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
imsize = 32

def calc_op_param(n, r, j)
  for x in range(convnet.shape[0]):
    n=((n-(convnet[x][0])+(2*convnet[x][2]))/convnet[x][1])+1
    r=r+((convnet[x][0])-1)*j
    j=j*convnet[x][1]
    print('layer#',x,'n_out:',n,'r_out:',r,'j_out:',j)
    
# input layer has following parameters
n_in=imsize
r_in=1
j_in=1
start_in=0.5
print(convnet.shape[0])
print(convnet[0][0])

calc_op_param(n_in, r_in, j_in)
```

### trial#7.4
* trial#7 workedbest hence trying same with batch=64

### trial#7.7
try with dropout=0.1

## Track#2 - Data Augmentation
### trial#1
*   imports: time, ModelCheckpoint, ImageDatagenerator
*   No change in hyperparameters: 
            batch_size = 128
            num_classes = 10
            epochs = 50
            l = 40
            num_filter = 12
            compression = 0.5
            dropout_rate = 0.2
* No change in dense/transition/output block
* Added data augmentation

#### Results
* Epoch 00010: val_acc improved from -inf to 0.14490, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.14490 to 0.17020, saving model to weights_best.hdf5
* Epoch 00030: val_acc did not improve from 0.17020
* Epoch 00040: val_acc did not improve from 0.17020
* Epoch 00050: val_acc improved from 0.17020 to 0.18930, saving model to weights_best.hdf5

#### Training
* Epoch 50/50
321/390 [==============================] - 145s 371ms/step - loss: 0.7760 - acc: 0.7243 - val_loss: 12.0266 - val_acc: 0.1893

#### Analysis
* overfitting hence increase regularization

### trial#2
* set bias True
* increase dropout from 0.2 => 0.5

Results
* no improvements

### trial#3
* try reduced data augmentation

Results
* Epoch 00010: val_acc improved from -inf to 0.38430, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.38430 to 0.47110, saving model to weights_best.hdf5

Training (interrupted by power off)
* Epoch 27/50
321/390 [==============================] - 214s 548ms/step - loss: 0.8863 - acc: 0.6832 - val_loss: 1.7246 - val_acc: 0.6014

### Analysis
Positive improvements from 18%@epoch#50 to 60%@epoch#28 hence lesser augmentation techniques helped.

### trial#4
* plot history
* include start/end time
* include hyperparameter tuning from hyperparam-trial#2
* decrease dropout: 0.25
* compression: 0.5
* l=12
* batch = 32
* shuffle=True
* augmentation +rotation, +vertical_flip

#### Results

#### Analysis
Train accuracy is abysmally low indicating underfit

### trial#5
* decrease dropout: 0.20
* compression: 1
* l=12
* batch = 32
* shuffle=True
* augmentation +rotation, +vertical_flip

#### Results
* Epoch 35/35
 327/1562  [==============================] - 416s 266ms/step - loss: 0.6746 - acc: 0.7613 - val_loss: 0.7955 - val_acc: 0.7349
 
 #### Analysis
*  Network train & validation performed well together however training accuracy@76% for epoch#35 is far lower than other networks(with just hyperparameter tuning)
* try reducing augmentation methods to help reduce regularization caused by execessive augmentation
* try making n/w deeper with either channel or layer

### trial#6
To make this network train faster as other networks, let's try restricting augmentation to one method 
* datagen = ImageDataGenerator(
    #featurewise_center=False,
    #featurewise_std_normalization=False,
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    horizontal_flip=True,
    #vertical_flip=False)
* batch_size = 32
num_classes = 10
epochs = 50
l = 40
num_filter = 12
compression = 1
dropout_rate = 0.25
* remove bias from transition block & keep in dense block

#### Results
* Epoch 00010: val_acc improved from -inf to 0.71670, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.71670 to 0.78560, saving model to weights_best.hdf5
* Epoch 00030: val_acc improved from 0.78560 to 0.82150, saving model to weights_best.hdf5
* Epoch 20/50 [==============================] - 377s 242ms/step - loss: 0.4676 - acc: 0.8373 - val_loss: 0.6929 - val_acc: 0.7856
* Epoch 30/50 [==============================] - 378s 242ms/step - loss: 0.4008 - acc: 0.8615 - val_loss: 0.5902 - val_acc: 0.8215
* Epoch 39/50 [==============================] - 377s 242ms/step - loss: 0.3541 - acc: 0.8771 - val_loss: 0.5688 - val_acc: 0.8272

#### Analysis
* changes from trail#6 helped train the network faster - restrict data augmentation to "horizontal flip", selective bias, 

#### trial#7
Objective: achieve training accuracy > 97%
* try increasing channels to 16 (taking inspiration from hyperparamter tuning n/w)

## Track#3 - hyperparameter tuning
### trail#1
* dropout = 0.5
* compression = 0.3
* l =12

#### Results
* Epoch 00010: val_acc improved from -inf to 0.35480, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.35480 to 0.54030, saving model to weights_best.hdf5
* Epoch 00030: val_acc did not improve from 0.54030
* Epoch 00040: val_acc improved from 0.54030 to 0.54440, saving model to weights_best.hdf5
* Epoch 00050: val_acc did not improve from 0.54440

* Epoch 50/50 [==============================] - 104s 2ms/step - loss: 1.1760 - acc: 0.5706 - val_loss: 2.1455 - val_acc: 0.4609

#### Analysis
train & test accuracy both low indicating "underfitting"


### trial#2
* decrease dropout: 0.25
* restore compression: 0.5
* increase layers: 16
* batch_size: 64

#### Results
* Epoch 00010: val_acc improved from -inf to 0.69480, saving model to weights_best.hdf5
* Epoch 00020: val_acc improved from 0.69480 to 0.71180, saving model to weights_best.hdf5
* Epoch 1/50
43776/50000 [=========================>....] - ETA: 34s - loss: 1.7092 - acc: 0.358050000/50000 [==============================] - 293s 6ms/step - loss: 1.6807 - acc: 0.3695 - val_loss: 2.3834 - val_acc: 0.3543
* Epoch 2/50
26240/50000 [==============>...............] - ETA: 2:03 - loss: 1.3981 - acc: 0.485450000/50000 [==============================] - 275s 5ms/step - loss: 1.3700 - acc: 0.4975 - val_loss: 1.8666 - val_acc: 0.4557
------------------
* Epoch 24/50
15104/50000 [========>.....................] - ETA: 3:02 - loss: 0.5948 - acc: 0.791450000/50000 [==============================] - 277s 6ms/step - loss: 0.5945 - acc: 0.7923 - val_loss: 0.8123 - val_acc: 0.7575
* Epoch 25/50
15232/50000 [========>.....................] - ETA: 3:00 - loss: 0.5989 - acc: 0.789550000/50000 [==============================] - 275s 6ms/step - loss: 0.5923 - acc: 0.7930 - val_loss: 0.7605 - val_acc: 0.7583

#### Analysis
Train & Validation accuracies moved very closely to each other indicating network was learning good.

### trial#3
* in l=40, batch=32, dropout=0, bias=false network: test_acc was 97% at epoch#25 but network had overfitting issue
* taking inspiration from above & retatinig best of current network
** l=32
** bias= true
** batch = 32
** droput = 0.25
** compression = 0.5
* shuffle=True

#### Results
* Epoch 00010: val_acc improved from 0.50790 to 0.75000, saving model to weights_best.hdf5
* Epoch 00020: val_acc did not improve from 0.78440
* Epoch 00030: val_acc improved from 0.78440 to 0.82130, saving model to weights_best.hdf5

#### Analysis
Network slowed down on training: 82%@epoch#20  => 87%@epoch#35

### trial#4
Objective: Push network to learn faster epoch#20 onwards
* compression=1
* l=16
* filter=16

## Session topics/notes

Session#1 
1. Convolution
2. Feature extraction
3. Receptive field ( how many layers?)
4. Pooling, 
5. Dimension reductionality, 
6. 1x1 convolution

Session#2 
7. Normalization, 
8. Back Propogation, 
9. Activations, 
10. Gradient descent calculation 

Session#3 
11. Concept of Channels, 
12. Conv Maths,
13. Dialated Conv,
14. Deconv, 
15. Depthwise Conv, 
16. Stride, 
17. Pooling, 
18. 1x1, 
19. Separable Conv, 
20. Grouped Conv, 
21. Dropout, 
22. ResNets 

Session#4 
23. Keras code
24. Sequential & Functional models
25. LSTM
26. Embedding Layers
27. Merge Layers
28. Loss functions
29. Optimizers
30. Image Augmentation
