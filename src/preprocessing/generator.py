import numpy as np
import tensorflow as tf
    
class MultiTaskGenerator(tf.keras.utils.Sequence):
    '''
    Generates data for Keras.\n
    Applies preprocessing and augmentation to the data with the help of the preprocessor and augmentator classes.\n
    '''
    def __init__(self, 
                 ids,
                 loader,
                 config,
                 batch_size=None,
                 shuffle=True, 
                 preprocessor=None):
        
        'Initialization'
        self.ids = ids
        self.loader = loader
        self.shuffle = shuffle
        self.preprocessor = preprocessor
        self.on_epoch_end()

        'Configuration'
        self.config = config
        self.labels = self.config['labels']
        self.dim = self.config['input_shape']
        self.dataset_path = self.config['dataset_path']
        self.n_input_channels = self.config['in_channels']
        self.n_classes = self.config['num_classes']

        'Batch size'
        if batch_size is None:
            self.batch_size = self.config['batch_size']
            print(f'Batch size not provided. Using default batch size: {self.batch_size}')
        else:
            self.batch_size = batch_size
            
        print(f'Generator configuration:\n'
            f'- Dataset path: {self.dataset_path}\n'
            f'- Dimensions: {self.dim}\n'
            f'- Batch size: {self.batch_size}\n'
            f'- Number of input channels: {self.n_input_channels}\n'
            f'- Number of classes: {self.n_classes}\n'
            f'- Shuffle: {self.shuffle}\n')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(batch_ids)
        return X, Y
    
    def __getcase__(self, index):
        'Returns the case of the given index'
        return self.ids[index]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.zeros((self.batch_size,*(self.dim),self.n_input_channels)) # Processed image
        Y = np.zeros((self.batch_size,*(self.dim),self.n_classes)) # Processed ground truth
        M = np.zeros((self.batch_size,*(self.dim),1)) # Processed brain mask

        # Generate data
        for i, rodent_key in enumerate(batch_ids):

            # Get mouse object from loader
            mouse = self.loader.get_subject(rodent_key)
            
            # Extract images and masks
            image, roi_ground_truth, brain_mask_ground_truth = mouse.get_images()

            # Preprocess and augment the images
            prep_image, [prep_ground_truth, prep_brain_mask] = self.preprocessor.preprocess(image, [roi_ground_truth, brain_mask_ground_truth])
            
            # Get data from the images
            prep_image_data = prep_image.get_fdata()
            prep_ground_truth_data = prep_ground_truth.get_fdata()
            prep_brain_mask = prep_brain_mask.get_fdata()
        
            # Convert to one hot encoding
            prep_ground_truth_data = tf.keras.utils.to_categorical(prep_ground_truth_data, num_classes=self.n_classes)

            # Stacking input into n input channels
            X[i] = np.stack((prep_image_data,)*self.n_input_channels, axis=-1)
            # One hot encoding
            Y[i] = prep_ground_truth_data
            # Brain mask
            M[i] = np.expand_dims(prep_brain_mask, axis=-1)

        return X, [Y, M]
    

class DomainAdaptationGenerator(tf.keras.utils.Sequence):
    '''
    Generates data for Keras.\n
    Applies preprocessing and augmentation to the data with the help of the preprocessor and augmentator classes.\n
    '''
    def __init__(self, 
                 ids,
                 loader,
                 config,
                 batch_size=None,
                 shuffle=True, 
                 preprocessor=None):
        
        'Initialization'
        self.ids = ids
        self.loader = loader
        self.shuffle = shuffle
        self.preprocessor = preprocessor
        self.on_epoch_end()

        'Configuration'
        self.config = config
        self.labels = self.config['labels']
        self.dim = self.config['input_shape']
        self.dataset_path = self.config['dataset_path']
        self.n_input_channels = self.config['in_channels']
        self.n_classes = self.config['num_classes']
        if batch_size is None:
            self.batch_size = self.config['batch_size']
        else:
            self.batch_size = batch_size
            
        print(f'Generator configuration:\n'
            f'- Dataset path: {self.dataset_path}\n'
            f'- Dimensions: {self.dim}\n'
            f'- Batch size: {self.batch_size}\n'
            f'- Number of input channels: {self.n_input_channels}\n'
            f'- Number of classes: {self.n_classes}\n'
            f'- Shuffle: {self.shuffle}\n')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.ids[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(batch_ids)
        return X, Y
    
    def __getcase__(self, index):
        'Returns the case of the given index'
        return self.ids[index]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        'Generates data containing batch_size samples'
        
        # Initialization
        X = np.zeros((self.batch_size,*(self.dim),self.n_input_channels)) # Processed image
        Y = np.zeros((self.batch_size,*(self.dim),self.n_classes)) # Processed ground truth
        M = np.zeros((self.batch_size,*(self.dim),1)) # Processed brain mask
        S = np.zeros((self.batch_size, 1)) # Sham

        # Generate data
        for i, rodent_key in enumerate(batch_ids):

            # Get mouse object from loader
            subject = self.loader.get_subject(rodent_key)

            # Extract images and masks
            image, roi_ground_truth, brain_mask_ground_truth = subject.get_images()

            unavailable_labels = subject.get_unavailable_labels()
            sham = subject.get_sham()

            # One hot encode ignore labels based on the number of classes
            if unavailable_labels:
                _one_hot_ignore = np.zeros(self.n_classes)
                for j, label in enumerate(self.labels):
                    if label in unavailable_labels:
                        _one_hot_ignore[j] = 1

            # Preprocess and augment the images
            prep_image, [prep_ground_truth, prep_brain_mask] = self.preprocessor.preprocess(image, [roi_ground_truth, brain_mask_ground_truth])
            prep_image_data = prep_image.get_fdata()
            prep_ground_truth_data = prep_ground_truth.get_fdata()
            prep_brain_mask = prep_brain_mask.get_fdata()
        
            # Convert to one hot encoding
            prep_ground_truth_data = tf.keras.utils.to_categorical(prep_ground_truth_data, num_classes=self.n_classes)

            # If one hot ignore is 1 for a label dimension, set the whole dimension to -1
            if unavailable_labels:
                for k, label in enumerate(_one_hot_ignore):
                    if label == 1:
                        prep_ground_truth_data[..., k] = -1

            # Stacking input into n input channels
            X[i] = np.stack((prep_image_data,)*self.n_input_channels, axis=-1)
            # One hot encoding
            Y[i] = prep_ground_truth_data
            # Brain mask
            M[i] = np.expand_dims(prep_brain_mask, axis=-1)
            # Sham
            S[i] = 0 if sham == True else 1

        return X, [Y, M, S]