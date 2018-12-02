import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.layers import LSTM 
from keras import regularizers

plt.rcParams['figure.figsize'] = (12,8)

mood_base = 276
mood_labels = {276: "Happy music",
                    277: "Funny music",
                    278: "Sad music",
                    279: "Tender music",
                    280: "Exciting music",
                    281: "Angry music",
                    282: "Scary music"}

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def data_generator(batch_size, records_in, start_frac=0, end_frac=1):
    '''
    Shuffles the Audioset training data and returns a generator of training data and one-hot mood labels
    batch_size: batch size for each set of training data and labels
    records_in: array of tfrecords to dish out
    start_frac: the starting point of the data set to use, as a fraction of total record length (used for CV)
    end_frac: the ending point of the data set to use, as a fraction of total record length (used for CV)
    '''
    max_len = 10
    records = records_in[int(start_frac * len(records_in)):int(end_frac * len(records_in))]
    num_recs = len(records)
    shuffle = np.random.permutation(range(num_recs))
    num_batches = num_recs // batch_size - 1
    j = 0

    while True:
        X = []
        y = []
        for idx in shuffle[j * batch_size:(j + 1) * batch_size]:
            example = records[idx]
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            example_label = list(np.asarray(tf_seq_example.context.feature['labels'].int64_list.value))
            mood_bin = intersection(example_label, mood_labels.keys())[0] # assume there'll always be a valid label.
            y.append(mood_bin - mood_base)

            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            audio_frame = []
            for i in range(n_frames):
                audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].
                                                         feature[i].bytes_list.value[0],np.uint8).astype(np.float32))
            pad = [np.zeros([128], np.float32) for i in range(max_len - n_frames)]
            audio_frame += pad
            X.append(audio_frame)

        j += 1
        if j >= num_batches:
            shuffle = np.random.permutation(range(num_recs))
            j = 0

        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=len(mood_labels))
        yield X, y

def logistic_regression_model():
    lr_model = Sequential()
    lr_model.add(BatchNormalization(input_shape=(10, 128)))
    lr_model.add(Flatten())
    lr_model.add(Dense(len(mood_labels), activation='sigmoid'))

    # if interested, try using different optimizers and different optimizer configs
    lr_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lr_model, 'Multiclass Logistic Regression')

def lstm_1layer_model():
    lstm_model = Sequential()
    lstm_model.add(BatchNormalization(input_shape=(None, 128)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(128, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01)))
    lstm_model.add(Dense(len(mood_labels), activation='sigmoid'))

    # if interested, try using different optimizers and different optimizer configs
    lstm_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lstm_model, 'LSTM 1-Layer')

def lstm_3layer_model():
    lstm3_model = Sequential()
    lstm3_model.add(BatchNormalization(input_shape=(None, 128)))
    lstm3_model.add(Dropout(0.5))

    lstm3_model.add(LSTM(64, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            return_sequences=True))

    lstm3_model.add(BatchNormalization())
    lstm3_model.add(Dropout(0.5))

    lstm3_model.add(LSTM(64, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01),
            return_sequences=True))

    lstm3_model.add(BatchNormalization())
    lstm3_model.add(Dropout(0.5))

    lstm3_model.add(LSTM(64, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01)))

    lstm3_model.add(Dense(len(mood_labels), activation='sigmoid'))
    
    # if interested, try using different optimizers and different optimizer configs
    lstm3_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lstm3_model, 'LSTM 3-Layer')


def train_model(model, tf_infile, batch_size_train, batch_size_validate, epochs, model_outfile):
    ''' Perform learning on the records in tf_infile, using 90/10 cross validation
        model - tuple (keras model to train, string_name_of_model)
        tf_infile - input filename
        batch_size_train - size of batches for training sets
        batch_size_validate - size of batches for validation
        epochs - number of epochs to run
        model_outfile - filename to save the resulting learned model to
    '''
    # validation set will be 10% of total set
    CV_frac = 0.1

    # load the records
    records = list(tf.python_io.tf_record_iterator(tf_infile))
    num_recs = len(records)

    # create generators for the training and validation sets
    train_gen = data_generator(batch_size_train, records, 0, 1 - CV_frac)
    val_gen = data_generator(batch_size_validate, records, 1 - CV_frac, 1)

    # learn the model
    lr_h = model[0].fit_generator(train_gen,steps_per_epoch=int(num_recs * (1 - CV_frac)) // batch_size_train, 
                                  epochs=epochs, 
                                  validation_data=val_gen, 
                                  validation_steps=int(num_recs * CV_frac) // batch_size_validate,
                                  verbose=1)
    # plot the results
    plt.plot(lr_h.history['acc'], label='training accuracy')
    plt.plot(lr_h.history['val_acc'], label='validation accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(model[1] + ' (n=' + str(num_recs) + ')')
    plt.show()
    lr_model.save('Models/' + model[1] + '_' + str(num_recs) + '.h5')

def do_it():
    # Un-comment the model you are interested in...
    # logistic regression, small balanced data set:
    #train_model(logistic_regression_model(), 'moods_balanced_subset_421recs.tfrecord', 20, 20, 100, 'Models/LogReg_421_20_20_100.h5')
    # logistic regression, large unbalanced data set:
    #TrainModel(LogisticRegressionModel(), 'moods_unbalanced_subset_16114recs.tfrecord', 32, 128, 100, 'Models/LogReg_16114_32_128_30.h5')

    # LSTM 1-layer, small balanced data set:
    #train_model(lstm_1layer_model(), 'moods_balanced_subset_421recs.tfrecord', 20, 20, 100, 'Models/LSTM1_421_20_20_100.h5')
    # LSTM 1-layer, large unbalanced data set:
    #TrainModel(lstm_1layer_model(), 'moods_unbalanced_subset_16114recs.tfrecord', 32, 128, 100, 'Models/LSTM1_16114_32_128_30.h5')

    # LSTM 3-layer, small balanced data set:
    train_model(lstm_3layer_model(), 'moods_balanced_subset_421recs.tfrecord', 20, 20, 500, 'Models/LSTM3_421_20_20_100.h5')
    # LSTM 3-layer, large unbalanced data set:
    #TrainModel(lstm_3layer_model(), 'moods_unbalanced_subset_16114recs.tfrecord', 32, 128, 100, 'Models/LSTM3_16114_32_128_30.h5')

if __name__ == "__main__":
    do_it()
    # used this once to shuffle the order of the records in the data sets and resave them.
    #shuffle_resave('moods_balanced_subset_421recs.tfrecord', 'moods_balanced_subset_421recs_shuffled.tfrecord', )
    #shuffle_resave('moods_unbalanced_subset_16114recs.tfrecord', 'moods_unbalanced_subset_16114recs_shuffled.tfrecord')