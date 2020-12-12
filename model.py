from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import LeakyReLU

def build_model(init='uniform', act='relu', drop_rate=0.15):

    bias_init = RandomUniform(minval=-0.05, maxval=0.05)
    act = LeakyReLU(alpha=0.01)

    classifier = Sequential()


    classifier.add(Conv2D(filters=10,
                          kernel_size=(8,8),
                          strides=(1,1),
                          input_shape=(28,28,1),
                          activation=act))

    classifier.add(Conv2D(filters=15,
                          kernel_size=(6,6),
                          activation=act))

    classifier.add(Conv2D(filters=20,
                          kernel_size=(5,5),
                          activation=act))

    classifier.add(Conv2D(filters=20,
                          kernel_size=(5,5),
                          activation=act))

    classifier.add(Flatten())

    classifier.add(Dense(units=256,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=256,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=256,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=64,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=64,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=32,
                         activation=act,
                         kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=10, activation='softmax'))

    adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier
