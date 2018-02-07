#!/usr/bin/python3
#
# If the output is close to the input,
# the naive solution would be to just pass the input through the network.
# If the output is mostly close to 0,
# then the naive solution would be to always return 0.
#
# Calculate the loss of this two naive solutions.
# The real loss of the network should be below these values.


from keras.models import Model
import sys
from keras.layers import Input, Conv2D
from keras.initializers import Ones, Zeros
import h5py

h5f = h5py.File(sys.argv[1], "r")
x = h5f["x"][:]
y = h5f["y"][:]

mashup = Input(shape=(None, None, 1), name='input')

# model with zero output
conv0 = Conv2D(1, 1, activation='linear',
               kernel_initializer=Zeros(), padding='same')(mashup)
model0 = Model(inputs=mashup, outputs=conv0)
model0.compile(loss='mean_squared_error', optimizer='adam')
model0.summary(line_length=150)

# model with output=input
conv1 = Conv2D(1, 1, activation='linear',
               kernel_initializer=Ones(), padding='same')(mashup)
model1 = Model(inputs=mashup, outputs=conv1)
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.summary(line_length=150)

error0 = model0.evaluate(x, y, batch_size=8)
error1 = model1.evaluate(x, y, batch_size=8)

print("MSE for output=all_zeros: %f" % error0)
print("MSE for output=input: %f" % error1)
