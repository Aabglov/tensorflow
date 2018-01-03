# Totally ripped from https://iamtrask.github.io/2017/03/21/synthetic-gradients/
# Messing around with for now


import numpy as np
import sys

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)

    return (x,y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
    return out * (1 - out)

class DNI(object):

    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,name,alpha = 0.1):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = (np.random.randn(input_dim, output_dim) * 2) - 1
        self.bias = (np.random.randn(output_dim) * 2) - 1

        self.weights_0_1_synthetic_grads = (np.random.randn(output_dim,output_dim) * .0) - .0
        self.bias_0_1_synthetic_grads = (np.random.randn(output_dim) * .0) - .0

        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha

    def forward_and_synthetic_update(self,input,debug=False):

        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights)  + self.bias)

        self.synthetic_gradient = (self.output.dot(self.weights_0_1_synthetic_grads) + self.bias_0_1_synthetic_grads)
        self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)

        self.weights -= self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        self.bias -= np.average(self.weight_synthetic_gradient,axis=0) * self.alpha

        if debug:
            print("")
            print(np.min(self.output),np.max(self.output))
            nonlin_deriv_output = self.nonlin_deriv(self.output)
            print(np.min(nonlin_deriv_output),np.max(nonlin_deriv_output))
            print(np.array_equal(self.nonlin_deriv(self.output),self.output))
            #print(np.min(self.synthetic_gradient),np.max(self.synthetic_gradient))
            #print(np.min(self.weight_synthetic_gradient),np.max(self.weight_synthetic_gradient))

        return self.weight_synthetic_gradient.dot(self.weights.T), self.output

    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = (self.synthetic_gradient - true_gradient)
        self.weights_0_1_synthetic_grads -= self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
        self.bias_0_1_synthetic_grads -= np.average(self.synthetic_gradient_delta,axis=0) * self.alpha


class Layer(object):

    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,name,alpha):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha

    def forward(self,input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        return self.output

    def backward(self,output_delta,debug=False):
        self.weight_output_delta = output_delta * self.nonlin_deriv(self.output)
        #print(self.weight_output_delta[0][:5])
        if debug:
            print("")
            print(np.min(self.output),np.max(self.output))
            nonlin_deriv_output = self.nonlin_deriv(self.output)
            print(np.min(nonlin_deriv_output),np.max(nonlin_deriv_output))
            print(np.array_equal(self.nonlin_deriv(self.output),self.output))
        return self.weight_output_delta.dot(self.weights.T)

    def update(self):
        self.weights -= self.input.T.dot(self.weight_output_delta) * self.alpha


np.random.seed(1)

num_examples = 1000
output_dim = 12
iterations = 10000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 100#0
alpha = 0.01#0.0001

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = DNI(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,"layer_1",alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,"layer_2",alpha)
layer_3 = Layer(layer_2_dim, output_dim, sigmoid, sigmoid_out2deriv,"layer_3",alpha)

for iter in range(iterations):
    error = 0
    synthetic_error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

        _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
        layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out,debug=False)

        # Normal update
        layer_3_out = layer_3.forward(layer_2_out)
        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.backward(layer_3_delta,debug=False)
        layer_3.update()

        layer_2.update_synthetic_weights(layer_2_delta)
        layer_1.update_synthetic_weights(layer_1_delta)

        #error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))
        error += (np.sum(np.abs(layer_3_delta)))
        #error -= np.sum( (batch_y * np.log(layer_3_out)) + ((1.-batch_y) * np.log(1.-layer_3_out)) )


    if(error < 0.01):
        print("\rIter:" + str(iter) + " Loss:" + str(error/batch_size))
        break

    print("\rIter:" + str(iter) + " Loss:" + str(error/batch_size))
    if(iter % 100 == 99):
        print("")
