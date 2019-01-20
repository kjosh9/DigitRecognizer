from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

def build_model(init='uniform', act='relu', opt='adam', drop_rate=0.15):

    classifier = Sequential()

    classifier.add(Conv2D(filters=20,
                          kernel_size=(4,4),
                          input_shape=(28,28,1),
                          activation=act))

    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Conv2D(filters=30,
                          kernel_size=(2,2),
                          input_shape=(10,10,20)))

    classifier.add(MaxPooling2D(pool_size=(2,2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))

    classifier.add(Dense(units=128, activation=act, kernel_initializer=init))
    classifier.add(Dropout(drop_rate))

    classifier.add(Dense(units=10, activation='softmax'))

    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier