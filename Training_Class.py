
import os
import random

import numpy as np
import cv2
import tensorflow as tf

import tensorflow_hub as hub

class ModelTraining:

    IMAGE_SIZE=(224,224)
    INPUT_LAYER_SIZE = IMAGE_SIZE + (3,)

    NUM_IMAGES_TO_CHOOSE = 100

    starting_dir = None
    other_example_dir_name = "Other"
    other_example_dir = None

    user_example_dir = os.path.join(os.getcwd(), "User")

    num_classes = 0

    other_images = np.ndarray()
    user_images = np.ndarray()

    def __init__(self):
        self.model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5")
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(2)
        ])


        self.model.build((None,) + ModelTraining.INPUTLAYERSIZE)

        # model compile parameters copied from tutorial at https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
            metrics=['accuracy'])

        starting_dir = os.getcwd()
        other_example_dir = os.path.join(ModelTraining.starting_dir, ModelTraining.other_example_dir_name)

    def load_in_other_data(self):
        del self.other_images
        self.other_images = np.ndarray()

        all_picture_numbers = np.linspace(0, 99999, num=100000, dtype=np.int64)

        selected_image_numbers = np.random.choice(all_picture_numbers, ModelTraining.NUM_IMAGES_TO_CHOOSE)

        for image_number in selected_image_numbers:
            image = cv2.imread(f"{f'{image_number}'.zfill(5)}.png")
            self.other_images.append(image)

    def load_in_user_data(self):
        del self.user_images
        self.user_images = np.ndarray()

        filenames = os.listdir(self.user_example_dir)

        for filename in filenames:
            image = cv2.imread(filename)
            self.user_images.append(image)

    def train(self):
        self.load_in_other_data()
        y0 = np.zeros(len(self.other_images.shape))

        self.load_in_user_data()
        y1 = np.ones(len(self.user_images))

        x_train = np.concatenate(self.other_images, self.user_images)
        y_train = np.concatenate(y0, y1)

        assert len(x_train) == len(y_train)


        for i in range(len(x_train)):
            index1 = random.randint(range(len(x_train)))
            index2 = random.randint(range(len(x_train)))

            # Shuffle in Unison
            temp = x_train[index1]
            x_train[index1] = x_train[index2]
            x_train[index2] = temp

            temp = y_train[index1]
            y_train[index1] = y_train[index2]
            y_train[index2] = temp

        epochs = 8
        self.model.fit(x_train, y_train, epochs=epochs, verbose=1)

    def predict(self, image):

        image = image.expand_dims(image, axis=0)

        predictions = self.model.predict(image)

        highest_prediction = predictions[0]
        highest_answer = np.argmax(highest_prediction)

        return highest_answer

if __name__ == "__main__":
    trainingInstance = ModelTraining()

    trainingInstance.train()