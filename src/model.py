import os
from src.logger import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional, 
                                     MaxPool3D, Activation, TimeDistributed, Flatten)

class LipReadingModel:
    def __init__(self, input_shape=(75, 46, 140, 1)):
        self.input_shape = input_shape
        self.model = self.build_model()
        self.load_weights()

    def build_model(self) -> Sequential:
        """Builds the lip-reading model."""
        logger.info("Building the model...")
        try:
            model = Sequential()

            model.add(Conv3D(128, 3, input_shape=self.input_shape, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool3D((1, 2, 2)))

            model.add(Conv3D(256, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool3D((1, 2, 2)))

            model.add(Conv3D(75, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPool3D((1, 2, 2)))

            model.add(TimeDistributed(Flatten()))

            model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
            model.add(Dropout(0.5))

            model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
            model.add(Dropout(0.5))

            model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

            logger.info("Model built successfully.")
            return model
        except Exception as e:
            logger.error(f"Error while building model: {e}")
            raise

    def load_weights(self):
        """Loads pre-trained weights from a checkpoint."""
        logger.info("Loading model weights...")
        try:
            weight_path = os.path.join('..', 'models', 'checkpoint')
            self.model.load_weights(weight_path)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

    def predict(self, video):
        """Runs prediction on the provided video input."""
        logger.info("Running model prediction...")
        try:
            return self.model.predict(tf.expand_dims(video, axis=0))
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
