"""
  @Time    : 2018-9-13 22:46
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : draw_model.py
  @Function: model visualize.
  
"""
import os
import mirror
from keras.utils import plot_model
import mrcnn.attention as modellib

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs_attention")
config = mirror.MirrorConfig()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
layers = ".*"
model.set_trainable(layers)
model.compile(config.LEARNING_RATE, config.LEARNING_MOMENTUM)

plot_model(model.keras_model, to_file="model.png", show_shapes=True, show_layer_names=True, rankdir="LR")
print("Model has been visualized!")
