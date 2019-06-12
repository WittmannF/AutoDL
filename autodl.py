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
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback
import copy

class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch = self.total_samples // self.batch_size)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset.  
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, n_iterations=None):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_iterations = n_iterations or 50 # replace 50 later
        self.learning_rates = np.geomspace(min_lr, max_lr, num=n_iterations+1)
        self.iteration = 0
        self.history = {'lr': [], 'iterations': []}
        
    def set_lr(self):
        '''Calculate the learning rate.'''
        next_lr = self.learning_rates[self.iteration]
        K.set_value(self.model.optimizer.lr, next_lr)
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        self.set_lr()
        #K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history['lr'].append(K.get_value(self.model.optimizer.lr))
        self.history['iterations'].append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        self.set_lr()
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()
        

class AutoDL():
    def __init__(self, data_dir='./data/', best_model_name='best_model.hdf5',
                 input_shape=(224,224,3), valid_folder_name='valid', 
                 train_folder_name='train', batch_size=32):
        
        self.data_dir = data_dir
        self.best_model_name = best_model_name
        self.best_model_path = f'{data_dir}{best_model_name}'
        
        # 1. Get Image Data Generators
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.train_loader = datagen.flow_from_directory(f'{data_dir}{train_folder_name}', target_size=input_shape[:2], class_mode='sparse', batch_size=batch_size)
        self.valid_loader = datagen.flow_from_directory(f'{data_dir}{valid_folder_name}', target_size=input_shape[:2], class_mode='sparse', batch_size=batch_size)

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
        
        
    def lr_finder(self, min_lr=1e-5, max_lr=1e-2, optimizer='adam', loss='sparse_categorical_crossentropy'):
        
        n_batches = self.train_loader.n//self.train_loader.batch_size
        
        lr_finder = LRFinder(min_lr=min_lr, max_lr=max_lr, 
                             n_iterations=n_batches)
        
        model_lr = copy.deepcopy(self.model)
        
        model_lr.compile(optimizer=optimizer, loss=loss)
        
        model_lr.fit_generator(self.train_loader, n_batches, epochs=1,
                        callbacks=[lr_finder])
            
        lr_finder.plot_loss()
    
    def img_from_filepath(self, filepath):
        img = load_img(filepath, target_size=(224, 224))
        return img

    def img_from_url(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        return img

    def predict_from_url(self, url, show_img=True):
        img = self.img_from_url(url)
        return self.predict_from_img(img, show_img=show_img)

    def predict_from_filepath(self, filepath, show_img=True):
        img = self.img_from_filepath(filepath)
        return self.predict_from_img(img, show_img=show_img)

    def predict_from_img(self, img, best_model_path=None, show_img=True):
        """best_model_path: If None, then it will be looked under path given in
        in the initialization"""
        
        best_model_path = best_model_path or self.best_model_path
        
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


    def run(self, epochs=10, learning_rate=1e-4):

        # 5. Define checkpoint
        checkpoint = ModelCheckpoint(self.best_model_path, monitor='val_acc', 
                                     verbose=1, save_best_only=True, mode='max')

        # 6. Train the model
        self.model.compile(optimizer=Adam(lr=learning_rate), 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

        self.model.fit_generator(self.train_loader, self.train_loader.n//self.train_loader.batch_size, epochs=epochs,
                        validation_data=self.valid_loader, validation_steps=self.valid_loader.n//self.valid_loader.batch_size, callbacks=[checkpoint])
