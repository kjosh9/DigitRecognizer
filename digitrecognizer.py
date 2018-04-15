import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import modelhelper

def main():

    training_set = pd.read_csv('train.csv')
    y = training_set.iloc[:,0].values
    X = training_set.iloc[:,1:].values

    X = X/255
    X = X.reshape(42000, 28, 28, 1)
    y = np_utils.to_categorical(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=0)

    model = modelhelper.build_model()
    model.fit(X_train, y_train, epochs=10)

if __name__ == '__main__':
    main()
