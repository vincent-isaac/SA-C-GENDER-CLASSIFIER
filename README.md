# SA-C-GENDER-CLASSIFIER
# Algorithm
1.
2.
3.
4.

## Program:
```
/*
Program to implement 
Developed by   :Vincent isaac jeyaraj J
RegisterNumber :212220230060  
*/
```
```python
import splitfolders  # or import split_folders
splitfolders.ratio("Male and Female face dataset", output="output", seed=1337, ratio=(.9, .1), group_prefix=None) # default values

import matplotlib.pyplot as plt
import matplotlib.image as mping
img = mping.imread('steve.jpg')
plt.imshow(img)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

from tensorflow.keras.preprocessing import image

test_image = image.load_img('steve.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image,axis=0)
test_image = test_image/255.
test_image.shape

import tensorflow_hub as hub
m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(2, activation='softmax')])

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

history = m.fit(train,epochs=2,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))

classes=train.class_indices
classes=list(classes.keys())

m.predict(test_image)

classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]

import pandas as pd
pd.DataFrame(history.history).plot()

```


## OUTPUT:
```
/*
1. CODE :
![SKILL ASSESSMENT OUTPUT](XXX.png)

2. DEMO VIDEO YOUTUBE LINK:

*/
```

