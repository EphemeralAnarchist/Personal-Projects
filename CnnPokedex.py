import matplotlib.pyplot as plt
import numpy as np
import keras

!wget 'https://www.dropbox.com/sh/s9r1av3m4eatd3y/AAA8zYti5b5tnyKfcah2Reaja'

!unzip AAA8zYti5b5tnyKfcah2Reaja -d dataset/


#Image Augmentation/Data Generator/Data Loader

#image augmentation basically edits the data slightly to expand training data pool
from keras.preprocessing import image

train_datagen=image.ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.3, horizontal_flip=True, rescale=1/255)
#rescale is very important to normalise the data to a small range

val_datagen=image.ImageDataGenerator(rescale=1/255)
#validation data doesnt need to be augmented

train_generator=train_datagen.flow_from_directory('dataset/Train',target_size=(100,100))
#size determines the dimensions of the generated pics

train_generator.class_indices

imgs, labels = next(train_generator)

plt.imshow(imgs[1])
labels[1]

val_generator=val_datagen.flow_from_directory('dataset/Test',target_size=(100,100))
val_generator.class_indices

from keras.models import Sequential
from keras.layers import  Conv2D, MaxPool2D, Flatten, Dense #ANN layers are Dense

model = Sequential()
  # input shape is only required for the first layer
model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu',input_shape=(100,100,3)))
  # adding the convolution layer
model.add(MaxPool2D(pool_size=(2,2)))
  # adding the pooling layer

model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))#input shape removed cos it will be working on otp of last layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu'))
model.add(Flatten())

model.add(Dense(units=3, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist= model.fit_generator(train_generator, steps_per_epoch=304//32, epochs=10, validation_data=val_generator, validation_steps=123//32)

model_history=hist.history

plt.plot(model_history['accuracy'], label='accuracy')
plt.plot(model_history['val_accuracy'], label='val accuracy')
plt.legend()

plt.plot(model_history['loss'], label='loss')
plt.plot(model_history['val_loss'], label='val loss')
plt.legend()

model.evaluate(val_generator)


#Test for images

!wget https://cdn.bulbagarden.net/upload/b/b8/025Pikachu_LG.png

test_img=image.load_img('./025Pikachu_LG.png',target_size=(100,100))
test_img=np.array(test_img)/255
plt.imshow(test_img)

test_img=test_img.reshape(1,100,100,3)
pred = model.predict_classes(test_img)
dic = train_generator.class_indices
rev_dic={v:k for k,v in dict.items() }
rev_dic[pred[0]]
