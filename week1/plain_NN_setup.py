import tensorflow as tf
import math

# general model set up for this problem

# having the model be global so all functions can access it
model = tf.keras.models.Sequential()

# input layer (2 inputs = function and nth derivate to take)
model.add(tf.keras.Input(shape=(2,)))

# hidden layer
model.add(tf.keras.layers.Dense(4, activation='relu'))

# output layer -> the nth derivate of the function inputted
model.add(tf.keras.layers.Dense(1))


# train the model on functions whose derivatives we know
# might try loading known derivatives into a .txt or .json file
# then loading them in

# the function to derive (can change this so the user can input a function)
@tf.function
def f(x):
    return math.sin(x)

n = 1 # default to taking the first derivative

beyond_first_der = input('Do you want to take a derivative past the first? ')
if beyond_first_der in ['y', 'yes']:
    n = int(input('What derivative do you want to take? Enter a number: '))

# pass the function and n through the (trained) model and hopefully get
# the requested derivative as an output
