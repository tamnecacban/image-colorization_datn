import gradio as gr
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# ---- Load Generator Architecture ----
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer='he_normal', use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer='he_normal', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def build_generator():
    inputs = tf.keras.layers.Input(shape=[None, None, 1])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# ---- Load model weights from the second code base ----
model = build_generator()
model.load_weights("D:\\ColorizationImage\\source\\source\\pix2pix_checkpoints_minimal\\best_generator.weights.h5")

# ---- Inference function ----
from PIL import Image, ImageEnhance
import numpy as np

from PIL import Image
import numpy as np
import tensorflow as tf

from PIL import Image
import numpy as np
import tensorflow as tf

def colorize(inp_path):
    """
    Nhận đường dẫn ảnh trắng đen (grayscale),
    trả về ảnh màu hóa bằng mô hình có đầu ra ∈ [0, 1],
    và được clip đúng như trong test_single_image().
    """

    # Bước 1: Load ảnh grayscale, resize về 256x256
    img = Image.open(inp_path).convert("L")
    original_size = img.size
    img_resized = img.resize((256, 256), Image.LANCZOS)

    # Bước 2: Tiền xử lý → chuẩn hóa [0, 1], reshape thành (1, H, W, 1)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    input_tensor = np.expand_dims(img_array, axis=0)

    # Bước 3: Dự đoán, clip về [0, 1]
    output = model(input_tensor, training=False)[0]
    output = tf.clip_by_value(output, 0.0, 1.0).numpy()

    # Bước 4: Scale về [0, 255] và convert sang uint8
    output = (output * 255).astype(np.uint8)

    # Bước 5: Resize về kích thước gốc
    color_img = Image.fromarray(output).resize(original_size, Image.LANCZOS)

    return inp_path, color_img




def display(bw_image_path, gen_image):
    os.makedirs("./demo", exist_ok=True)
    save_path = "./demo/result.jpg"
    gen_image.save(save_path)
    return save_path

def predict(inp):
    original_input, colorized_image = colorize(inp)
    return display(original_input, colorized_image)

def get_example_images():
    examples = os.listdir("./examples")
    return ["./examples/" + name for name in examples]

gr.Interface(
    fn=predict,
    title="Black & White Image Colorization (Pix2Pix)",
    inputs=gr.Image(type="filepath", label="Upload Grayscale Image"),
    outputs=gr.Image(label="Colorized Image"),
    examples=get_example_images()
).launch()
