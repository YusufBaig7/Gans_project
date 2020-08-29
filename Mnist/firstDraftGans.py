import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import time

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

plt.imshow(train_images[1])

train_images = train_images/255
test_images = test_images/255

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 100
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(7,(3,3), padding ='same', input_shape = (28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(50, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])    
    return model

model_discriminator = make_discriminator_model()

model_discriminator(np.random.rand(1, 28, 28, 1).astype('float32'))

discriminator_optimizer = tf.optimizers.Adam(1e-3)

def get_discriminator_loss(real_predictions, fake_predictions):
    real_predictions = tf.sigmoid(real_predictions)
    fake_predictions = tf.sigmoid(fake_predictions)
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
    return real_loss + fake_loss

def make_generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, input_shape = (100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, (3, 3), padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, (3, 3), strides = (2, 2), padding = 'same')
    ])
    return model

generator = make_generator_model()

generator_optimizer = tf.optimizers.Adam(1e-4)

def get_generator_loss(fake_predictions):
    fake_predictions = tf.sigmoid(fake_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions), fake_predictions)
    return fake_loss

def train(dataset, epochs):
    for _ in range(epochs):
        for images in dataset:
            images = tf.cast(images, tf.dtypes.float32)
            train_step(images)

def train_step(images):
    fake_image_noise = np.random.randn(BATCH_SIZE, 100).astype('float32')
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fake_image_noise)
        real_output = model_discriminator(images)
        fake_output = model_discriminator(generated_images)
    
        gen_loss = get_generator_loss(fake_output)
        disc_loss = get_discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model_discriminator.trainable_variables))
    
        print('Generator loss is:', np.mean(gen_loss))
        print('Discriminator loss is:', np.mean(disc_loss))

train(train_dataset, 10)

plt.imshow(tf.reshape(generator(np.random.randn(1,100)),(28,28)),cmap = "gray")