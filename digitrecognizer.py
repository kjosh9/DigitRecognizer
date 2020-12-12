import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
import model as m

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
    y = to_categorical(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.125, random_state=2)

    early_stopping = EarlyStopping(monitor='acc',
                                   min_delta=.0005,
                                   patience=5)

    model = m.build_model()
    model.fit(x=X_train,
              y=y_train,
              epochs=100,
              batch_size=128,
              callbacks=[early_stopping])

    y_prediction = model.predict(X_test)
    lss = log_loss(y_test, y_prediction)
    print("Validation loss is: ", lss)

    test_data = pd.read_csv('test.csv')
    X_submission = test_data.iloc[:,:].values
    X_submission = X_submission/255
    X_submission = X_submission.reshape(X_submission.shape[0], 28, 28, 1)
    y_submission = model.predict(X_submission)
    y_submission = np.argmax(y_submission, axis=1)
    create_submission_file(y_submission)

if __name__ == '__main__':
    print("Version: " + tf.__version__)
    print("CUDA support: " + str(tf.test.is_built_with_cuda))
    tf.config.list_physical_devices('GPU')
    main()
