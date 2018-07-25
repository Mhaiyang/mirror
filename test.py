"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
x = PyramidROIAlign([pool_size, pool_size],
                    name="roi_align_mask")([rois, image_meta] + feature_maps)

# Conv layers
x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                       name="mrcnn_mask_conv1")(x)
x = KL.TimeDistributed(BatchNorm(),
                       name='mrcnn_mask_bn1')(x, training=train_bn)
x = KL.Activation('relu')(x)

x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                       name="mrcnn_mask_conv2")(x)
x = KL.TimeDistributed(BatchNorm(),
                       name='mrcnn_mask_bn2')(x, training=train_bn)
x = KL.Activation('relu')(x)

x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                       name="mrcnn_mask_conv3")(x)
x = KL.TimeDistributed(BatchNorm(),
                       name='mrcnn_mask_bn3')(x, training=train_bn)
x = KL.Activation('relu')(x)

x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                       name="mrcnn_mask_conv4")(x)
x = KL.TimeDistributed(BatchNorm(),
                       name='mrcnn_mask_bn4')(x, training=train_bn)
x = KL.Activation('relu')(x)

x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                       name="mrcnn_mask_deconv")(x)
x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                       name="mrcnn_mask")(x)
return x