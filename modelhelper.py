from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

def build_model(init='uniform', act='relu', opt='adam'):

    classifier = Sequential()
    classifier.add(Conv2D(32, (4,4), input_shape=(28,28,1), activation=act))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation=act, kernel_initializer=init))
    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier
