from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

def build_model(init='uniform', act='relu', drop_rate=0.15):

    classifier = Sequential()


    classifier.add(Conv2D(filters=20,
                          kernel_size=(7,7),
                          input_shape=(28,28,1),
                          activation=act))

    classifier.add(Conv2D(filters=20,
                          kernel_size=(7,7),
                          input_shape=(22,22,20)))

    classifier.add(Conv2D(filters=30,
                          kernel_size=(5,5),
                          input_shape=(16,16,40)))

    classifier.add(Conv2D(filters=40,
                          kernel_size=(4,4),
                          input_shape=(14,14,70)))

    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=256, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=256, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=256, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=64, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=64, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=32, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))
    classifier.add(BatchNormalization())

    classifier.add(Dense(units=10, activation='softmax'))

    adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier