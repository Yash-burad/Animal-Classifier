# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:09:03 2020

@author: kus
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:20:28 2020

@author: kus
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
num_classes=7
weights_path='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model=Sequential()
model.add(ResNet50(include_top=False,pooling='avg'))
model.add(Dense(num_classes,activation='softmax'))

model.layers[0].trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input


data_generator=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=data_generator.flow_from_directory('C:/Users/kus/Desktop/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/raw-img',target_size=(224,224),batch_size=12,class_mode='categorical')

validation_generator=data_generator.flow_from_directory('C:/Users/kus/Desktop/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/test1',target_size=(224,224),batch_size=20,class_mode='categorical')

model.fit_generator(train_generator,steps_per_epoch=1000,validation_data=validation_generator,validation_steps=100)

model.save('IOT.h5')
model.save_weights('IOT1.h5')

from tensorflow.keras.models import load_model
model1=load_model('IOT.h5')


from tensorflow.keras.preprocessing import image
import numpy as np
img_width, img_height = 224, 224  
# loading up our datasets
#train_data_dir = 'data/train'  
validation_data_dir = 'data/validation'  
test_data = 'C:/Users/kus/Desktop/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/cat.4001.jpg'
num_classes = 7

test_data=image.load_img(test_data,target_size=(img_width,img_height))
test_data=image.img_to_array(test_data)
test_data=np.expand_dims(test_data,axis=0)

result=model1.predict(test_data)
result.tolist()
classes=['Dog','Horse','Elephant','Hen','Cat','Cow','Goat']
#animal=result[classes.index(max(classes))]
maxm=0
ind=-1
#for x in range(0,len(result)):
print(result)
result = [item for sublist in result for item in sublist]
for j in range(len(result)):  # or range(len(theta))
   if result[j]>maxm:
       maxm=result[j]
       ind=j
print(classes[ind])