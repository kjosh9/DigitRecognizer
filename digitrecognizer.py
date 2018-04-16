import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
import modelhelper

def create_submission_file(labels, filename='Submission.csv'):
    submission_df = pd.read_csv('sample_submission.csv')
    submission_df['Label'] = labels
    submission_df.to_csv(filename, index=False)

def main():

    training_set = pd.read_csv('train.csv')
    y = training_set.iloc[:,0].values
    X = training_set.iloc[:,1:].values

    X = X/255
    X = X.reshape(X.shape[0], 28, 28, 1)
    y = np_utils.to_categorical(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=0)

    model = modelhelper.build_model()
    model.fit(X_train, y_train, epochs=10)

    y_prediction = model.predict(X_test)
    lss = log_loss(y_test, y_prediction)
    print("Loss is: ", lss)

    test_data = pd.read_csv('test.csv')
    X_submission = test_data.iloc[:,:].values
    X_submission = X_submission/255
    X_submission = X_submission.reshape(X_submission.shape[0], 28, 28, 1)
    y_submission = model.predict(X_submission)
    y_submission = np.argmax(y_submission, axis=1)
    create_submission_file(y_submission)

if __name__ == '__main__':
    main()
