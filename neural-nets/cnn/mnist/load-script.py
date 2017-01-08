import numpy as np
import keras.models
from keras.models import model_from_json
from keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()
n_train, height, width = X_train.shape
n_test, _, _, = X_test.shape

X_train = 	X_train.reshape(n_train,28,28,1).astype('float32')
X_test = X_test.reshape(n_test, 28, 28,1).astype('float32')

X_train /= 255
X_test /= 255

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
out = loaded_model.predict(X_test[0:1])
print(out)
print(np.argmax(out,axis=1))
#print()