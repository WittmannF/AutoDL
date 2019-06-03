from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

class AutoDL():
    def __init__(self, data_dir='./data/', input_shape=(224,224,3)):
        # 1. Get Image Data Generators
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_loader = datagen.flow_from_directory(f'{data_dir}train', target_size=input_shape[:2], class_mode='sparse', batch_size=10)
        self.valid_loader = datagen.flow_from_directory(f'{data_dir}valid', target_size=input_shape[:2], class_mode='sparse', batch_size=10)

        # 2. Initialize base model
        base_model = VGG16(include_top=False, input_shape=input_shape)

        # 3. Freeze layers from the base model
        for layer in base_model.layers:
            layer.trainable=False

        # 4. Add Fully connected layer
        
        class_indices = self.train_loader.class_indices
        
        self.index_to_class = {v: k for k, v in class_indices.items()}

        number_of_classes = len(class_indices)

        self.model = Sequential([base_model,
                            Flatten(),
                            Dropout(rate=0.5),
                            Dense(1024, activation='relu'),
                            Dense(number_of_classes, activation='softmax')])
        
        
    def img_from_filepath(self, filepath):
        img = load_img(filepath, target_size=(224, 224))
        return img

    def img_from_url(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        return img

    def predict_from_url(self, url, model='best_model.h5', show_img=True):
        img = self.img_from_url(url)
        return self.predict_from_img(img, show_img=show_img)

    def predict_from_filepath(self, filepath, show_img=True):
        img = self.img_from_filepath(filepath)
        return self.predict_from_img(img, show_img=show_img)

    def predict_from_img(self, img, best_model_path='best_model.hdf5', show_img=True):
        
        # Showing image
        if show_img:
            plt.imshow(img)
            plt.show()
            
        # Convert to a Numpy array
        img = np.asarray(img)

        # Reshape by adding 1 in the beginning to be compatible as input of the model
        img = img.reshape(1,224,224,3)

        # Prepare the image for the VGG model
        img = preprocess_input(img)
        
        # Load weights
        print("Loading model with best weigths")
        self.model.load_weights(best_model_path)

        # Decode output of model into classes and probabilities
        result = self.model.predict(img)
        print(f'Predicted Class: {self.index_to_class[result.argmax()]}')
        #return self.index_to_class[result.argmax()]


    def run(self, best_model_filename='best_model.hdf5'):

        # 5. Define checkpoint
        checkpoint = ModelCheckpoint(best_model_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        # 6. Train the model
        self.model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(self.train_loader, self.train_loader.n//self.train_loader.batch_size, epochs=10,
                        validation_data=self.valid_loader, validation_steps=self.valid_loader.n//self.valid_loader.batch_size, callbacks=[checkpoint])
