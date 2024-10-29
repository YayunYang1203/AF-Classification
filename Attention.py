

# file handling functionality
import os
import pandas as pd
import numpy as np

# useful utilities
import time
import pickle

# let's do datascience ...
import numpy as np

# import keras deep learning functionality
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPool1D
from keras.layers import Dense
from keras.layers import Dropout

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
# 定义Attention
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer

# 假设类别数为3
num_classes = 3

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

# tell the application whether we are running on a server or not (so as to
# influence which backend matplotlib uses for saving plots)
headless = False

#
# get the data
#


# load the npz file
data_path = r'C:\Users\DELL\Desktop\rnn-based-af-detection-master\data\data\training_and_validation.npz'

af_data   = np.load(data_path, allow_pickle=True)

# extract the training and validation data sets from this data
x_train = af_data['x_train']
y_train = af_data['y_train']
x_test  = af_data['x_test']
y_test  = af_data['y_test']

#
# create and train the model才
#

# set the model parameters
n_timesteps = x_train.shape[1]
mode = 'concat'
n_epochs = 300
batch_size = 4096

# create a bidirectional lstm model (based around the model in:
# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# )
inp = Input(shape=(n_timesteps,1,))
x = Bidirectional(LSTM(200,
                       return_sequences=True,
                       dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=x)

#添加Attention
inp = Input(shape=(n_timesteps,1,))
x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inp)
x = Attention(n_timesteps)(x)  # 添加Attention层
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=x)

# 将标签转换为独热编码格式
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 创建模型
inp = Input(shape=(n_timesteps, 1,))
x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inp)
x = Attention(n_timesteps)(x)  # 使用Attention层
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(num_classes, activation='softmax')(x)  # 修改输出层为多分类
model = Model(inputs=inp, outputs=x)

# set the optimiser
opt = Adam()

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 设置模型检查点
directory = './model/af_classification_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
if not os.path.exists(directory):
    os.makedirs(directory)
filename = 'af_classification_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=directory + filename, verbose=1, save_best_only=True)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# set up a model checkpoint callback (including making the directory where to
# save our weights)
directory = './model/initial_runs_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
os.makedirs(directory)
filename  = 'af_lstm_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=directory+filename,
                               verbose=1,
                               save_best_only=True)

# fit the model
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[checkpointer])

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# pickle the history so we can use it later
with open(directory + 'training_history', 'wb') as file:
    pickle.dump(history.history, file)

# set matplotlib to use a backend that doesn't need a display if we are
# running remotely
if headless:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# plot the results

# accuracy
f1 = plt.figure()
ax1 = f1.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('training and validation accuracy of af diagnosis')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.text(0.4, 0.05,
         ('validation accuracy = {0:.3f}'.format(best_accuracy)),
         ha='left', va='center',
         transform=ax1.transAxes)
plt.savefig('af_lstm_training_accuracy_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# loss
f2 = plt.figure()
ax2 = f2.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('training and validation loss of af diagnosis')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.text(0.4, 0.05,
         ('validation loss = {0:.3f}'
          .format(min(history.history['val_loss']))),
         ha='right', va='top',
         transform=ax2.transAxes)
plt.savefig('af_lstm_training_loss_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# we're all done!
print('all done!')