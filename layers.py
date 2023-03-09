# custom L1Dist layer module

# import dependencies
import tensorflow as tf
from keras.layers import Layer

class L1Dist(Layer):
    def __init__(self, **kwargs): # keyword argument
        super().__init__()
     # calculate similaroty   
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)