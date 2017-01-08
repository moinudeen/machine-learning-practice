import numpy as np
X = np.array(([3,5],[5,1],[10,2]),dtype=float)
Y = np.array(([75],[82],[93]),dtype=float)
x = X/np.amax(X,axis=0)
y= Y/100

class Neural_Network(object):
	"""Neural Network to predict test scores based on number of hours of study and sleep"""
	def __init__(self):
		#define hyperparamters- size of each layer
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#weights(Paramters defining each synapse) Initialized as random values
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

	def forward(self,X):
		#propagate inputs through network
		self.z2 = np.dot(X,self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat
   
	def sigmoid(self,z):
		#apply sigmoid activation function 
		return 1/(1+np.exp(-z))

	def sigmoidprime(z):
		#derivative of sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
	  	#Compute cost for given X,y, use weights already stored in class.
  		self.yHat = self.forward(X)
  		J = 0.5*sum((y-self.yHat)**2)
  		return J

	def  costFunctionPrime(self,X,y):
		#computes derivative with respect to w1 and w2
		self.yHat = self.forward(X)
		
		#delta3 is the back propagating error
		#delta3 = -(y-yhat)*f'(z3) (sigmoidprime function)
		delta3 = np.multiply(-(y-self.yHat),self.sigmoidprime(self.z3))
		#djdw2 is the derivative from the last layer
		#djdw2 = a2.Trans * delta3
		dJdW2 = np.dot(self.a2.T,delta3)

		#delta2 is the cost derivative from the second layer
		# delta2 = delta3.w2.trans * f'(z2)
		delta2 = np.dot(delta3,self.W2.T)*self.sigmoidprime(self.z2)
		#djdw1 is the derivative from the second layer
		#djdw1 = X.trans * delta2
		dJdW1 = np.dot(X.T,delta2)

		return dJdW1,dJdW2

	  #Helper Functions for interacting with other classes:
	def getParams(self):
	  #Get W1 and W2 unrolled into vector:
	  params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
	  return params
	    
	def setParams(self, params):
	  #Set W1 and W2 using single paramater vector.
	  W1_start = 0
	  W1_end = self.hiddenLayerSize * self.inputLayerSize
	  self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
	  W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
	  self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
	  
	def computeGradients(self, X, y):
	  dJdW1, dJdW2 = self.costFunctionPrime(X, y)
	  return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
  paramsInitial = N.getParams()
  numgrad = np.zeros(paramsInitial.shape)
  perturb = np.zeros(paramsInitial.shape)
  e = 1e-4

  for p in range(len(paramsInitial)):
#Set perturbation vector
perturb[p] = e
N.setParams(paramsInitial + perturb)
loss2 = N.costFunction(X, y)

N.setParams(paramsInitial - perturb)
loss1 = N.costFunction(X, y)

#Compute Numerical Gradient
numgrad[p] = (loss2 - loss1) / (2*e)

#Return the value we changed to zero:
perturb[p] = 0

  #Return Params to original value:
  N.setParams(paramsInitial)

return numgrad
  

NN = Neural_Network()
yHat = NN.forward(x)
print(yHat)