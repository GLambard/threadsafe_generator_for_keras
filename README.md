# An "ultimate" thread safe data generator for Keras :zap::zap::zap: :satisfied:

If you use [Keras](https://keras.io) for training a neural network's model, you may use ```fit()``` or ```fit_generator()``` 
functions (see [here](https://keras.io/models/sequential/#sequential-model-methods) for details). The latter is particularly 
useful when dealing with big datasets but you usually need to define a data generator which fits your needs. 

## Why a thread safe data generator
1. You want to use more than 1 worker (CPU-thread)
2. **You don't want your training data to be read more than once per epoch!**

## But how?
[keras.utils.Sequence()](https://keras.io/utils/#sequence) is your new friend!

## System
- python 3.6
- keras 2.1.6
- tensorflow(-gpu) 1.8.0

## Usage
```
from keras.utils import Sequence
''' add necessary libraries from keras and others to define your model, e.g. 
from keras.model import Sequential
...
'''

<model definition and compilation>
model = Sequential()
model.add(...)

model.compile(...)

class data_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)
        
batch_size = 100
history = model.fit_generator(
    generator = data_generator(x_train_gen, y_train_gen, batch_size), 
    steps_per_epoch = math.floor(x_train_gen.shape[0]/batch_size), 
    epochs = 100, 
    validation_data = DataSequence(x_valid_gen, y_valid_gen, batch_size), 
    validation_steps = math.floor(x_valid_gen.shape[0]/batch_size), 
    max_queue_size = 10, 
    workers = multiprocessing.cpu_count(),
    use_multiprocessing = True, 
    shuffle = True,
    initial_epoch = 0, 
    ...)
```
Here, x_set and y_set are numpy arrays. Additionally, you can add any transformation to the data which pleases you in the 
```__get_item__()``` (see [here](https://keras.io/utils/#sequence) for another example with images) 

# Source 

[keras.utils.Sequence()](https://keras.io/utils/#sequence)

# I hope it helps! :smiley:
