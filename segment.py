import sys
import tensorflow as tf
import numpy as np
import cv2
import tensorflow.keras.backend as K


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 3
categories = [
    (  0,  0,  0), 
    (244, 35,232), 
    (128, 64,128),
    ( 70, 70, 70),
    (153,153,153),
    (107,142, 35),
    ( 70,130,180),
    (220, 20, 60), 
    (  0,  0,142)
]
class_weights = [
    1.0127322460863948,
    2.6085443999046314,
    0.31666787465874224,
    0.5282767855660234,
    9.547994201504986,
    0.7136052450682393,
    4.016557513088011,
    6.336719440431782,
    1.5025437504442105
]

def one_hot_to_colors(mask):
    new_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype='int32')
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            new_mask[i, j, :] = categories[np.argmax(mask[i][j])]
    return new_mask


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

class DiceCoefLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=[]):
        super(DiceCoefLoss, self).__init__()
        self.class_weights = class_weights

    def get_weight_multiplier(self, y_true):
        axis = -1
        classSelectors = K.argmax(y_true, axis=axis)
        classSelectors = tf.cast(classSelectors, 'int32')
        classSelectors = [K.equal(i, classSelectors) for i in range(len(self.class_weights))]
        classSelectors = [tf.cast(x, 'float32') for x in classSelectors]

        weights = [sel * w for sel,w in zip(classSelectors, self.class_weights)]
        weightMultiplier = weights[0]

        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        return weightMultiplier
    
    def loss(self, y_true, y_pred, smooth=1e-7):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = self.loss(y_true, y_pred) 
        if self.class_weights:
            weightMultiplier = self.get_weight_multiplier(y_true)

            loss = tf.math.reduce_sum(loss * weightMultiplier)
            return loss
        return loss



if __name__ == '__main__':
    try:
        img_path = sys.argv[1]
    except:
        print("You have to pass a file path")

    model = tf.keras.models.load_model('unet_vgg19_backbone_30_epochs.h5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=DiceCoefLoss(class_weights),
            # loss = weighted_loss(dice_coef_loss, class_weights),
              metrics=[dice_coef, tf.keras.metrics.CategoricalAccuracy()])

    image = cv2.imread(img_path)
    cv2.imshow("Original image", image)
    
    original_size = image.shape[:2][::-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = np.array(image).astype('float32')
    image /= 255.

    predicted_mask = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_mask = one_hot_to_colors(predicted_mask)
    predicted_mask = predicted_mask.astype(np.uint8)
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2BGR)
    predicted_mask = cv2.resize(predicted_mask, original_size)


    cv2.imshow("Predicted mask", predicted_mask)

    cv2.waitKey()
    