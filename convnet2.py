from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras import backend as K

class Convnet:

    def __init__(self, img_w=150, img_h=150):
        self.img_width = img_w
        self.img_height = img_h
        self.training_directory = 'data/train'
        self.validation_directory = 'data/validation'

        # data augmentation
        self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = self.train_datagen.flow_from_directory(self.training_directory,
                                                                target_size=(self.img_width, self.img_height),
                                                                batch_size=self.batch_size,
                                                                class_mode='categorical')

        self.validation_generator = self.test_datagen.flow_from_directory(self.validation_directory,
                                                                    target_size=(self.img_width, self.img_height),
                                                                    batch_size=self.batch_size,
                                                                    class_mode='categorical')
        
        # hyperparameters
        self.num_training_samples = 3000
        self.num_validation_samples = 600
        self.epochs = 3
        self.batch_size = 32

        # Convnet model
        self.model = self.build_model()

        self.model.compile(loss='categorical_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

        

    def build_model(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        return model

    def train_model(self):
        self.model.fit_generator(self.train_generator,
                                steps_per_epoch=self.num_training_samples // self.batch_size,
                                epochs=self.epochs,
                                validation_data=self.validation_generator,
                                validation_steps=self.num_validation_samples // self.batch_size)

        self.model.save('model.h5')
        self.model.save_weights('weights.h5')
