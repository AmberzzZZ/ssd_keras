from ssd_back import vgg16_back
from ssd_layers import L2Normalization, Priorbox
from keras.layers import Input, Conv2D, Flatten, concatenate, Activation, Reshape
from keras.models import Model
import keras.backend as K


def ssd(input_shape, n_classes):

    inpt = Input(input_shape)
    if input_shape[0]>300:
        conv4, conv7, conv8, conv9, conv10, conv11 = vgg16_back(inpt, gap=True)
    else:
        conv4, conv7, conv8, conv9, conv10, conv11 = vgg16_back(inpt)

    # predict heads:
    n_boxes = [3,6,6,6,6,6]
    conv4 = L2Normalization()(conv4)
    # conf head [N,H,W,n_boxes*n_classes]
    conv4_conf = Conv2D(n_boxes[0]*n_classes, 3, strides=1, padding='same')(conv4)
    conv7_conf = Conv2D(n_boxes[1]*n_classes, 3, strides=1, padding='same')(conv7)
    conv8_conf = Conv2D(n_boxes[2]*n_classes, 3, strides=1, padding='same')(conv8)
    conv9_conf = Conv2D(n_boxes[3]*n_classes, 3, strides=1, padding='same')(conv9)
    conv10_conf = Conv2D(n_boxes[4]*n_classes, 3, strides=1, padding='same')(conv10)
    conv11_conf = Conv2D(n_boxes[5]*n_classes, 3, strides=1, padding='same')(conv11)
    # loc head [N,H,W,n_boxes*4]
    conv4_loc = Conv2D(n_boxes[0]*4, 3, strides=1, padding='same')(conv4)
    conv7_loc = Conv2D(n_boxes[1]*4, 3, strides=1, padding='same')(conv7)
    conv8_loc = Conv2D(n_boxes[2]*4, 3, strides=1, padding='same')(conv8)
    conv9_loc = Conv2D(n_boxes[3]*4, 3, strides=1, padding='same')(conv9)
    conv10_loc = Conv2D(n_boxes[4]*4, 3, strides=1, padding='same')(conv10)
    conv11_loc = Conv2D(n_boxes[5]*4, 3, strides=1, padding='same')(conv11)
    # priorbox: [N,n_priorboxes,8]
    img_size = (input_shape[0], input_shape[1])
    conv4_prior = Priorbox(img_size, 30., aspect_ratios=[2], variances=[0.1,0.1,0.2,0.2])(conv4)
    conv7_prior = Priorbox(img_size, 60., max_size=114., aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2])(conv7)
    conv8_prior = Priorbox(img_size, 114., max_size=168., aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2])(conv8)
    conv9_prior = Priorbox(img_size, 168., max_size=222., aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2])(conv9)
    conv10_prior = Priorbox(img_size, 222., max_size=276., aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2])(conv10)
    conv11_prior = Priorbox(img_size, 276., max_size=330., aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2])(conv11)
    # flatten & concat multi layers & reshape
    # loc head: raw output
    loc_head = concatenate([Flatten()(conv4_loc),
                            Flatten()(conv7_loc),
                            Flatten()(conv8_loc),
                            Flatten()(conv9_loc),
                            Flatten()(conv10_loc),
                            Flatten()(conv11_loc)],axis=1)
    num_boxes = K.int_shape(loc_head)[-1] // 4
    loc_head = Reshape((num_boxes,4))(loc_head)
    # conf head: softmax
    conf_head = concatenate([Flatten()(conv4_conf),
                             Flatten()(conv7_conf),
                             Flatten()(conv8_conf),
                             Flatten()(conv9_conf),
                             Flatten()(conv10_conf),
                             Flatten()(conv11_conf)],axis=1)
    conf_head = Reshape((num_boxes,n_classes))(conf_head)
    conf_head = Activation('softmax')(conf_head)
    # prior box head
    prior_head = concatenate([conv4_prior,
                              conv7_prior,
                              conv8_prior,
                              conv9_prior,
                              conv10_prior,
                              conv11_prior],axis=1)
    output = concatenate([loc_head, conf_head, prior_head], axis=2)     # 4+n_classes+8

    model = Model(inpt, output)

    return model


if __name__ == '__main__':

    model = ssd(input_shape=(300,300,3), n_classes=10)
    model.summary()














