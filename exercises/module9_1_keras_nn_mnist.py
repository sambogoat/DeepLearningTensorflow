# Module 9 Keras
# NN Model on MNIST dataset

import keras
from keras.models import Sequential
from keras.layers import Dense


# Hyper Parameters
n_features = 784
n_classes = 10
learning_rate = 0.5
training_epochs = 2

# Step 1: Pre-process the  Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

# Step 2: Define the Model

L1 = 200
L2 = 100
L3 = 60
L4 = 30

model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
model.add(Dense(L4, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
print(model.summary())
#
# # Step 3: Compile the Model
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# # Step 4: Train the Model
# model.fit(X_train, y_train,epochs=training_epochs)
#
# # Step 5: Evaluate the Model
# loss,acc = model.evaluate(X_test, y_test)
# print("\nTraining Accuracy = ",acc)
#
# # Step 6: Save the Model
# model.save("./models/mnist_nn.h5")

