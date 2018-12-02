import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten
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
    rec_len = len(records)
    shuffle = np.random.permutation(range(rec_len))
    num_batches = rec_len // batch_size - 1
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
            shuffle = np.random.permutation(range(rec_len))
            j = 0

        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=len(mood_labels))
        yield X, y



def LogisticRegression(tf_infile, batch_size_train, batch_size_validate, epochs, model_outfile):
    ''' Perform logistic regression on the records in tf_infile, using 90/10 cross validation
        tf_infile - input filename
        batch_size_train - size of batches for training sets
        batch_size_validate - size of batches for validation
        epochs - number of epochs to run
        model_outfile - filename to save the resulting learned model to
    '''
    lr_model = Sequential()
    lr_model.add(BatchNormalization(input_shape=(10, 128)))
    lr_model.add(Flatten())
    lr_model.add(Dense(len(mood_labels), activation='sigmoid'))

    # if interested, try using different optimizers and different optimizer configs
    lr_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # validation set will be 10% of total set
    CV_frac = 0.1

    # load the records
    records = list(tf.python_io.tf_record_iterator(tf_infile))
    rec_len = len(records)

    # create generators for the training and validation sets
    train_gen = data_generator(batch_size_train, records, 0, 1 - CV_frac)
    val_gen = data_generator(batch_size_validate, records, 1 - CV_frac, 1)

    # learn the model
    lr_h = lr_model.fit_generator(train_gen,steps_per_epoch=int(rec_len * (1 - CV_frac)) // batch_size_train, 
                                  epochs=epochs, 
                                  validation_data=val_gen, 
                                  validation_steps=int(rec_len * CV_frac) // batch_size_validate,
                                  verbose=1)
    # plot the results
    plt.plot(lr_h.history['acc'], label='training accuracy')
    plt.plot(lr_h.history['val_acc'], label='validation accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()
    lr_model.save('Models/LogisticRegression_bal421_100Epochs.h5')

def shuffle_resave(infile, outfile):
    # used this function once to shuffle the order of the records in the data sets and resave them.
    records = list(tf.python_io.tf_record_iterator(infile))
    # shuffle the records
    rec_len = len(records)
    shuffle = np.random.permutation(range(rec_len))
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for i in shuffle:
            record = tf.train.SequenceExample.FromString(records[i])
            writer.write(record.SerializeToString())

def do_it():
    # small balanced data set:
    LogisticRegression('moods_balanced_subset_421recs.tfrecord', 20, 20, 100, 'Models/LogReg_421_20_20_100.h5')
    # large unbalanced data set:
    LogisticRegression('moods_unbalanced_subset_16114recs.tfrecord', 32, 128, 30, 'Models/LogReg_16114_32_128_30.h5')

if __name__ == "__main__":
    do_it()
    # used this once to shuffle the order of the records in the data sets and resave them.
    #shuffle_resave('moods_balanced_subset_421recs.tfrecord', 'moods_balanced_subset_421recs_shuffled.tfrecord', )
    #shuffle_resave('moods_unbalanced_subset_16114recs.tfrecord', 'moods_unbalanced_subset_16114recs_shuffled.tfrecord')