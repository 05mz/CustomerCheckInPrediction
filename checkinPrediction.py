#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import streamlit as st

#importing dataset
training_df = pd.read_csv('training.csv')

st.write("training_df.head(5)")
training_df.head(5)
st.write("training_df.shape")
training_df.shape
st.write("training_df.describe()")
training_df.describe()

#creating labels and features
labels = training_df['BookingsCheckedIn']
features = training_df.drop(columns=['BookingsCheckedIn'])

#data preprocessing
labels.replace(0,0,inplace=True)
labels.replace(not 0,1,inplace=True)
st.write("Labels")
labels[0:5]
#data preprocessing, here we replace textual data with random numeric data
features = pd.get_dummies(features)
st.write("features")
features[0:5]

features = features.values.astype('float32')
labels = labels.values.astype('float32')
st.write("features after preprocessing",features[0:2])
st.write("labels after preprocessing",labels[0:2])
st.write("lenght of features",len(features[0]))

#splitting training and testing and validation data
features_train, features_test, labels_train, labels_test=train_test_split(features,labels,test_size=0.2)
features_train, features_validation, labels_train, labels_validation = train_test_split(features,labels,test_size=0.2)

#creating a sequential model
import tensorflow as tf
from tensorflow import keras
classifier = keras.Sequential([keras.layers.Dense(32, input_shape=(85,)),
                          keras.layers.Dense(20, activation=tf.nn.relu),
                         keras.layers.Dense(3,activation='softmax')])

#compiling the model
classifier.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])

#training the model
history = classifier.fit(features_train, labels_train, epochs=20, validation_data=(features_validation, labels_validation))

#evaluating the model
prediction_features = classifier.predict(features_test)
performance = classifier.evaluate(features_test, labels_test)
st.write("performance",performance)

#testing the test data

test_df = pd.read_csv('test.csv')

labelsTest = test_df['BookingsCheckedIn']
featuresTest = test_df.drop(columns=['BookingsCheckedIn'])

labelsTest.replace(0,0,inplace=True)
labelsTest.replace(not 0,1,inplace=True)

featuresTest = pd.get_dummies(featuresTest)
featuresTest[0:5]

featuresTest = featuresTest.values.astype('float32')
labelsTest = labelsTest.values.astype('float32')
# st.write(featuresTest[0:2])
# st.write(labelsTest[0:2])
st.write(len(featuresTest[0]))
  
features_train1, features_test1, labels_train1, labels_test1=train_test_split(featuresTest,labelsTest,test_size=0.9)
features_train1, features_validation1, labels_train1, labels_validation1 = train_test_split(featuresTest,labelsTest,test_size=0.9)

import tensorflow as tf
from tensorflow import keras
classifier = keras.Sequential([keras.layers.Dense(32, input_shape=(225,)),
                          keras.layers.Dense(20, activation=tf.nn.relu),
                         keras.layers.Dense(100,activation='softmax')])

classifier.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])

prediction_features = classifier.predict(features_test1)
performance1 = classifier.evaluate(features_test1, labels_test1)
st.write(performance1)

def graph1():
  x=training_df['Age']
  y=training_df['LodgingRevenue']
  plt.figure(1 , figsize = (15 ,6))
  plt.bar(x,y)
  plt.xlabel('Age') , plt.ylabel('LodgingRevenue')
  plt.show()
data=training_df
x=training_df['Age']
y=training_df['LodgingRevenue']
width=15
height=6
use_container_width=True
# st.bar_chart(graph1())
# st.bar_chart(data=pd.read_csv('training.csv'), x=data['Age'], y=data['BookingsCheckedIn'], width=15, height=6, use_container_width=True)
#st.bar_chart(data,None,x(training_df['Age']),y(training_df['LodgingRevenue']), width(15), height(6), use_container_width(True))
st.write("graph shows the relationship between age groups and lodging revenue \n incase of surplus, we need to prioritize the middle aged people in 40s \n as they spend more on the booking for amenties")
chart_data = pd.DataFrame(
     np.random.randn(50, 3),
     columns=["a", "b", "c"])

st.bar_chart(chart_data)

# def graph2():
#   x=training_df['Age']
#   y=training_df['BookingsCheckedIn']
#   plt.figure(1 , figsize = (15 ,6))
#   plt.bar(x,y)
#   plt.xlabel('Age') , plt.ylabel('BookingsCheckedIn')
#   plt.show()
# st.image(graph2())
# st.write("graph shows that people in 20s and 30s are prone to not checking in after the booking")

# import seaborn
# def graph3():
#   plt.figure(1 , figsize = (15 , 5))
#   seaborn.countplot(y = 'DistributionChannel' , data = pd.read_csv('training.csv'))
#   plt.show()
# st.image(graph3())
# st.write("above graph shows that a lot of people come through travel agents \n so we should focus on advertising ourselves through travel agents")
