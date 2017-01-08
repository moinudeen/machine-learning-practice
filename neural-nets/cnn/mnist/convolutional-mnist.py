from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential

#load dataset and split into training and testing sets
(X_train,y_train),(X_test,y_test) = mnist.load_data()
#training set has 60,000 images with dim 28x28
n_train, height, width = X_train.shape
#validation set has 10000 images with same dim
n_test, _, _, = X_test.shape

#Preprocessing the data into right form 
X_train = 	X_train.reshape(n_train, height, width,1).astype('float32')
X_test = X_test.reshape(n_test, height, width,1).astype('float32')

#Normalize the values from [0,255] to [0,1] Probability
X_train /= 255
X_test /= 255

#numbers 0-9 , so we have 10 classes
n_class = 10
#convert to categorical values
y_train = to_categorical(y_train,n_class)
y_test = to_categorical(y_test,n_class)

# Build the model 
# Sequential model is initialized
model = Sequential()

# define the hyperparameters 
#no. of filters
n_filters = 32
#conv filter size
n_conv = 3
#pooling window size
n_pool = 2

#we begin adding our conv and pooling layers using the super simple Keras API.
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model.add(Convolution2D(
		#no. of filters and dimensions of the filter
		n_filters, n_conv, n_conv,
		#apply the filter only within the image border
		border_mode='valid',
		#define input shape
		input_shape=(height,width,1)
	)) 
#add an activation layer - here we add Rectified Linear Unit
model.add(Activation('relu'))

#second conv layer - no need to specify input and output dimensions. Keras figures it out :)
model.add(Convolution2D(n_filters,n_conv,n_conv))
model.add(Activation('relu'))

#add the Pooling layer 
model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))

# add the dropout, FC layers
from keras.layers import Dropout,Flatten, Dense
model.add(Dropout(0.25))

#flatten the data for 1D layers
model.add(Flatten()) 

#Dense(n_outputs) for FC layer
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#softmax layer gives probability for each class
model.add(Dense(n_class))
model.add(Activation('softmax'))

#Compiling the model - Specify the loss function and Optimizer
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#no. of examples to look at during each iteration of training
batch_size = 128
#no. of times to run through full training set
n_epochs = 10

#fit the training data to train our model
model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=n_epochs,validation_data=(X_test,y_test))

#evaluate the model 
loss,accuracy = model.evaluate(X_test,y_test)
print('loss:', loss)
print('accuracy:', accuracy)

#serialize model to JSON
model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

#serialize weights to HDF5
model.save_weights("model.h5")
print("saved model to disk")


#print(model.predict(X_test[0]))