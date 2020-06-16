import numpy as np
import random, os


# class for the main neural network which is Perceptron
class Perceptron:
    
    # intializaition of the variables
    def __init__(self, noInputs):
        
        # bias will later be randomized with the last weight
        self.bias = 1

        # weights are randomized
        self.weights = []
        for i in range(noInputs+1):
            self.weights.append(random.random())

    # this is the function being used for the data so that we can fit the neural network
    def activation(self,guess):

        # step function
        if guess > 0:
            guess = 1
        else:
            guess = 0

        # returns the guess in binary form
        return guess


    def train(self, inputs, outputs):

        # goes through all the inputs and outputs
        for i in range(len(outputs)):

            # we find the error in the neural network
            guess = self.think(inputs[i])
            error = outputs[i] - guess



            # weights are adjusted according to the error in the neural network
            for j in range(len(self.weights)):
                # first we change the errors of the weights of inputs and then for the bias
                try:
                    self.weights[j] += error * inputs[i][j] 
                except IndexError:
                    self.weights[j] += error * self.bias 


    # the calculations that need to be done for finding the value of the next neuron
    def think(self,inputs):

        # summation of the product of weights and input neurons (NOTE: I could have done it with dot method in numpy)
        sum = 0
        for i in range(len(self.weights)):
            # first we add the inputs with the products and the final one is the product of weight and bias
            try:
                sum += inputs[i] * self.weights[i]
            except IndexError:
                sum += self.bias * self.weights[i]


        # we find the activation of the required summation
        return self.activation(sum)



# two input neurons perceptrons
'''
# we create object for all the possible binary perceptrons with two inputs

# traditional logic gates
por = Perceptron(2)
pxor = Perceptron(2)
pand = Perceptron(2)
pxand = Perceptron(2)

# made up logic gates
pinput1 = Perceptron(2)
pinput2 = Perceptron(2)
pxinput1 = Perceptron(2)
pxinput2 = Perceptron(2)

# all possible inputs
inputs = [[0,0],[0,1],[1,0],[1,1]]


# training
for i in range(50):

    #  perceptron for or is trained
    outputs = [0,1,1,1]
    por.train(inputs,outputs)


    #  perceptron for and is trained
    outputs = [0,0,0,1]
    pand.train(inputs,outputs)


    #  perceptron for xor is trained
    outputs = [1,0,0,0]
    pxor.train(inputs,outputs)


    #  perceptron for xand is trained
    outputs = [1,1,1,0]
    pxand.train(inputs,outputs)


    #  perceptron for input1 = output is trained
    outputs = [0,0,1,1]
    pinput1.train(inputs,outputs)


    #  perceptron for not(input1) = output is trained
    outputs = [1,1,0,0]
    pxinput1.train(inputs,outputs)


    #  perceptron for input2 = output is trained
    outputs = [0,1,0,1]
    pinput2.train(inputs,outputs)


    #  perceptron for not(input2) = output is trained
    outputs = [1,0,1,0]
    pxinput2.train(inputs,outputs)




# the binary inputs are taken from the user
x = int(input("Enter x:"))
y = int(input("Enter y:"))


# the predicted output of the perceptron is given
print("or is", por.think([x,y]))
print("and is", pand.think([x,y]))
print()
print("xor is", pxor.think([x,y]))
print("xand is", pxand.think([x,y]))
print()
print()
print("input1 is", pinput1.think([x,y]))
print("xinput1 is", pxinput1.think([x,y]))
print()
print("input2 is", pinput2.think([x,y]))
print("xinput2 is", pxinput2.think([x,y]))
'''


# three input neurons perceptron
'''

# defining the perceptron with three input neurons
p = Perceptron(3)


# this perceptron takes the first input as its answer ...
# too lazy to add all possible perceptrons...
# now that i think about it i could've automated that process as well... but i'm too lazy to do that as well...
# what i only did this to check whether a perceptron with more than two inputs work... basically i'm not gonna do it... You're still here?... Why are you still here?... hum hum hummm hum... go away you're not going to get anything... you're still here... well okay you're cool... just leave stop wasting your time... okay i'm just going to stop... i'm seriously worried about you... you need to leave... you really have a problem i'm just going to stop... okay fine i'll do the automation of making perceptrons later... happy now... you're creepy i'm just going to leave... and not complete the automation of perceptrons... okay fine i'll do it if you agree to leave... you're still here... omg can you stop... i'm not going to do it... okay then bye)

inputs = [[0,0,0],[1,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,1],[1,1,1],[0,0,1],[1,1,0]]
outputs = [0,1,1,0,1,0,1,0,1]


# the perceptron is trained
for i in range(50):
    p.train(inputs,outputs)


# change the value manually (too lazy to write the inputs)
x = int(input("Enter x:"))
y = int(input("Enter y:"))
z = int(input("Enter z:"))
print(p.think([x,y,z]))
'''
