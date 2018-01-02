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

class Layer(object):

    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha):

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
iterations = 1000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 100#0
alpha = 0.3

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = Layer(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_2 = Layer(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_3 = Layer(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv,alpha)

for iter in range(iterations):
    error = 0
    #total = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y

        layer_2_delta = layer_3.backward(layer_3_delta,debug=False)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)

        layer_1.update()
        layer_2.update()
        layer_3.update()

        #error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))
        #error +=  np.sum(np.abs(layer_3_delta ** 2))
        #ylna+(1−y)ln(1−a)]
        error -= np.sum( (batch_y * np.log(layer_3_out)) + ((1.-batch_y) * np.log(1.-layer_3_out)) )
        #total += 1

    #error /= total

    if error < 1e-2:
        print("\rIter:" + str(iter) + " Loss:" + str(error/batch_size))

        pred_1_out = layer_1.forward(x[0].reshape((1,24)))
        pred_2_out = layer_2.forward(pred_1_out)
        pred_3_out = layer_3.forward(pred_2_out)

        bin_rep = ''.join([str(int(round(i))) for i in x[0]])
        x1 = bin_rep[:output_dim]
        x2 = bin_rep[output_dim:]
        bin_y = ''.join([str(int(round(i))) for i in y[0]])
        bin_pred = ''.join([str(int(round(i))) for i in pred_3_out[0]])

        print("x1:   {}".format(x1))
        print("x2:   {}".format(x2))
        print("y   : {}".format(bin_y))
        print("pred: {}".format(bin_pred))
        break

    print("\rIter:" + str(iter) + " Loss:" + str(error/batch_size))
    if(iter % 100 == 99):
        print("")


batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]
layer_1_out = layer_1.forward(batch_x)
layer_2_out = layer_2.forward(layer_1_out)
layer_3_out = layer_3.forward(layer_2_out)

print(batch_x[0])
print(batch_y[0])
for i in range(10):
    print("")
    print(layer_3_out[i])
