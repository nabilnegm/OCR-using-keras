import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import cv2


#load letters training dataset
train_x = []
train_y = []
j = 1

for i in range(53760):
    a = "train_letters/id_"+str(i+1)+"_label_"+str(j)+".png"
    train_x.append(cv2.imread(a)[:,:,1])
    train_y.append(j)
    if (i+1) % 8 == 0:
        if j+1 == 29:
            j = 1
        else:
            j = j + 1


#load digits training dataset from mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range (30000):
    train_x.append(cv2.resize(cv2.bitwise_not(X_train[i]),(32,32)))
    train_y.append(y_train[i]+29)



#preprossecing
train_x = np.array(train_x)
train_x = train_x/255
train_x = train_x.reshape(train_x.shape[0], 32, 32, 1).astype('float32')

train_y = np.array(train_y)
train_y = to_categorical(train_y)
train_y = train_y[:,1:]


#load letters testing dataset
test_x = []
test_y = []
j = 1
for i in range (3360):
    a = "test/id_"+str(i+1)+"_label_"+str(j)+".png"
    test_x.append(cv2.imread(a)[:,:,1])
    test_y.append(j)
    if (i+1) % 2 == 0:
        if j+1 == 29:
            j = 1
        else:
            j = j + 1

#load digits training dataset from mnist
for i in range (X_test.shape[0]):
    test_x.append(cv2.resize(cv2.bitwise_not(X_test[i]),(32,32)))
    test_y.append(y_test[i]+29)

#preprossecing
test_x = np.array(test_x)
test_x = test_x/255
test_x = test_x.reshape(test_x.shape[0],32,32,1)

test_y = np.array(test_y)
test_y = to_categorical(test_y)
test_y = test_y[:,1:]


#model initialization
#sequential model
#conv=>maxpool=>dropout=>norm=>conv=>maxpool=>dropout=>norm=>flatten=>fullyconnected=>dropout=>norm=>fullyconnected=>softmax
#adam optimization
model = Sequential()

conv1 = model.add(Conv2D(80, (5, 5), activation='relu', input_shape=(32, 32, 1)))
pool1 = model.add(MaxPooling2D(pool_size=(2, 2)))
drop1 = model.add(Dropout(0.5))
norm1 = model.add(BatchNormalization())
conv2 = model.add(Conv2D(64, (5, 5), activation='relu'))
pool2 = model.add(MaxPooling2D(pool_size=(2, 2)))
drop2 = model.add(Dropout(0.5))
norm2 = model.add(BatchNormalization())
model.add(Flatten())
fc1 = model.add(Dense(1024, activation='relu'))
drop3 = model.add(Dropout(0.5))
norm3 = model.add(BatchNormalization())
fc2 = model.add(Dense(38, activation='softmax'))

# model.load_weights('weights_file.h5')
# model.summary()



adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
history = model.fit(train_x, train_y, batch_size = 50, epochs = 100, shuffle = True, verbose = 1, validation_data = (test_x,test_y))


#save the model and the weights
model.save('OCR_model.h5')
model.save_weights('weights_file.h5')


#evaluate on test sets
# score = model.evaluate(test_x, test_y, batch_size = 200)
# print(100-score[1]*100)

#plot accuracy and loss with respect to epochs
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
