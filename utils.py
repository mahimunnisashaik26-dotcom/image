import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

def predict_image(img, model):
    # ✅ Convert to RGB (fix grayscale error)
    img = img.convert("RGB")

    img = img.resize((224, 224))
    img = np.array(img)

    # Extra safety (rare case)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    return decoded