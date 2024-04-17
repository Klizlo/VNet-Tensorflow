from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.models import Model

from VNet import VNet
import tensorflow as tf


input_shape = (1000, 64, 64, 1)
input = Input(shape=input_shape)

vnet = VNet(1, keep_prob=0.001)
logits = vnet.network_fn(x=input, is_training=True)
model = Model(inputs=input, outputs=logits)
model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="Adam",
              metrics=['accuracy'])

# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)