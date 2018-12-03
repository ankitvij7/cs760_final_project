import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.layers import LSTM, MaxPooling1D, Conv1D
from keras import regularizers
import sklearn.metrics

# validation set will be 10% of total set
CV_frac = 0.1

# comment out/in the labels you'd like to be filtered from the input and the classification
mood_labels = {
    276: "Happy music",
    #277: "Funny music",
    278: "Sad music",
    #279: "Tender music",
    #280: "Exciting music",
    281: "Angry music",
    282: "Scary music",
}
# build mapping of moods indices to 0-based indices
mood_labels_to_ordinals = dict()
mood_ordinals_to_labels = dict()
n = 0
for k in mood_labels.keys():
    mood_labels_to_ordinals[k] = n
    mood_ordinals_to_labels[n] = k
    n += 1

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

max_len = 10
def extract_record_to_xy(record):
    ''' parses a tfrecord and returns tuple (x,y) where x is the 10x128 feature vector and y is the 0-indexed label '''
    tf_seq_example = tf.train.SequenceExample.FromString(record)
    example_label = list(np.asarray(tf_seq_example.context.feature['labels'].int64_list.value))
    moods = intersection(example_label, mood_labels.keys()) # assume there'll always be a valid label.
    if len(moods) == 0:
        return None
    y = mood_labels_to_ordinals[moods[0]]

    n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
    audio_frame = []
    for i in range(n_frames):
        audio_frame.append(np.frombuffer(tf_seq_example.feature_lists.feature_list['audio_embedding'].
                                                    feature[i].bytes_list.value[0],np.uint8).astype(np.float32))
    pad = [np.zeros([128], np.float32) for i in range(max_len - n_frames)]
    audio_frame += pad
    return audio_frame, y


def data_generator(batch_size, records, start_frac=0, end_frac=1):
    '''
    Shuffles the Audioset training data and returns a generator of training data and one-hot mood labels
    batch_size: batch size for each set of training data and labels
    records_in: array of tfrecords to dish out
    start_frac: the starting point of the data set to use, as a fraction of total record length (used for CV)
    end_frac: the ending point of the data set to use, as a fraction of total record length (used for CV)
    '''
    max_len = 10
    num_recs = len(records)
    shuffle = np.random.permutation(range(num_recs))
    num_batches = num_recs // batch_size - 1
    j = 0

    while True:
        X = []
        y = []
        for idx in shuffle[j * batch_size:(j + 1) * batch_size]:
            example = records[idx]
            X.append(records[idx][0])
            y.append(records[idx][1])

        j += 1
        if j >= num_batches:
            shuffle = np.random.permutation(range(num_recs))
            j = 0

        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=len(mood_labels))
        yield X, y

def logistic_regression_model():
    ''' Creates a logistic regression model. Used by train_model '''
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
    ''' Creates a 1 layer LSTM model. Used as input to train_model() '''
    lstm_model = Sequential()
    lstm_model.add(BatchNormalization(input_shape=(10, 128)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(128, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l2(0.01)))
    lstm_model.add(Dense(len(mood_labels), activation='softmax'))

    # if interested, try using different optimizers and different optimizer configs
    lstm_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lstm_model, 'LSTM 1-Layer')

def lstm_3layer_model():
    ''' Creates a 3 layer LSTM model. Used as input to train_model() '''
    lstm3_model = Sequential()
    lstm3_model.add(BatchNormalization(input_shape=(10, 128)))
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

    lstm3_model.add(Dense(len(mood_labels), activation='softmax'))
    
    # if interested, try using different optimizers and different optimizer configs
    lstm3_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lstm3_model, '3-layer LSTM')

def nn_model():
    ''' Creates a traditional nn model. Used as input to train_model() '''
    model = Sequential()
    model.add(BatchNormalization(input_shape=(10, 128)))
    #model.add(Conv1D(20, kernel_size=5, strides=2,
    #                 activation='relu',
    #                 input_shape=(10, 128)))
    #nn_model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(len(mood_labels), activation='sigmoid'))
    model.add(Dense(len(mood_labels), activation='sigmoid'))

    # Other layer choices:
    #
    #nn_model = Sequential()
    #nn_model.add(Conv1D(20, kernel_size=5, strides=2,
    #                 activation='relu',
    #                 input_shape=(10, 128)))
    ##nn_model.add(MaxPooling1D(pool_size=2, strides=2))
    ##nn_model.add(Conv1D(64, 5, activation='relu'))
    #nn_model.add(MaxPooling1D(pool_size=2))
    #nn_model.add(Dense(128, activation='relu', input_shape=(10, 128)))
    ##nn_model.add(MaxPooling1D(pool_size=2))
    #nn_model.add(Dense(128, activation='relu'))
    #nn_model.add(Flatten())
    ##nn_model.add(Dense(len(mood_labels), activation='relu'))
    #nn_model.add(Dense(len(mood_labels), activation='softmax'))

    # if interested, try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (model, '2-layer NN')

def print_label_stats(name, records):
    ''' prints the statistical breakdown of training and validation sets '''
    ret = dict()
    for m in mood_ordinals_to_labels.keys():
        ret[m] = 0

    tot = len(records)
    for rec in records:
        ret[rec[1]] += 1

    print(name + ' stats:')
    for m in ret.keys():
        v = ret[m]
        print(' %15s: %6d (%.3f)' % (mood_labels[mood_ordinals_to_labels[m]], v, v / tot))
    print(' %15s: %6d (%.3f)\n' % ('total', tot, tot / tot))

def train_model(model, train_records, validate_records, batch_size_train, batch_size_validate, epochs):
    ''' Perform learning on the records in tf_infile, using 90/10 cross validation
        model - keras model to train
        train_records - instances to train with
        validate_records - instances to valiate with
        batch_size_train - size of batches for training sets
        batch_size_validate - size of batches for validation
        epochs - number of epochs to run
    '''

    # learn the model
    num_classes = len(mood_labels)
    num_train_recs = len(train_records)
    num_val_recs = len(validate_records)
    num_recs = num_train_recs + num_val_recs
    validation_data = (np.asarray([r[0] for r in validate_records]),
                       np.asarray([keras.utils.to_categorical(r[1], num_classes) for r in validate_records]))
    history = model.fit(x=np.asarray([r[0] for r in train_records]),
                        y=np.asarray([keras.utils.to_categorical(r[1], num_classes) for r in train_records]),
                        batch_size=batch_size_train,  
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_data)
    return (model, history)

def plot_epochs(history, title):
    # plot the results
    plt.rcParams['figure.figsize'] = (12,8)
    plt.plot(history.history['acc'], label='training accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.show()


def load_records(tf_infile):
    # load the records
    print("Loading records from '" + tf_infile + "'...")
    records = []
    for t in tf.python_io.tf_record_iterator(tf_infile):
        r = extract_record_to_xy(t)
        if r:
            records.append(r)
    print("Loaded %d records" % len(records))
    return records

def show_confusion_matrix(model, records):
    predictions = model.predict_on_batch(np.asarray([r[0] for r in records]))
    conf = sklearn.metrics.confusion_matrix([r[1] for r in records], [np.argmax(p) for p in predictions])
    print("Confusion matrix:")
    print(conf)
    conf = conf / len(records)
    print(conf)


def main():

    # choose an input file by commenting in/out
    input_file = 'moods_unbalanced_subset_15615recs.tfrecord'
    #input_file = 'moods_balanced_subset_401recs.tfrecord'


    records = load_records(input_file)
    num_records = len(records)

    split = int((1-CV_frac) * num_records)
    train_records = records[0:split]
    validate_records = records[split:-1]

    print_label_stats("train", train_records)
    print_label_stats("validate", validate_records)

    # train, or load model...comment out/in as desired.
    train = True
    if train:
        # pick a model by changing the function on the next line
        #(model, title) = logistic_regression_model() 
        #(model, title) = lstm_1layer_model() 
        (model, title) = lstm_3layer_model() 
        #(model, title) = nn_model() 
        (model, history) = train_model(model, train_records, validate_records, 32, min(len(validate_records), 128), 60)
        model.save(title + '_most_recent.h5')
    else:
        infile = '3-layer LSTM_most_recent.h5'
        print("Loading model: " + infile)
        model = keras.models.load_model(infile)

    show_confusion_matrix(model, validate_records)

    if train:
        plot_epochs(history, title)

    # Un-comment the model you are interested in...

    # logistic regression, small balanced data set:
    #train_model(logistic_regression_model(), 'moods_balanced_subset_401recs.tfrecord', 32, 10, 100, 'Models/LogReg_401_20_20_100.h5')
    # logistic regression, large unbalanced data set:
    #train_model(LogisticRegressionModel(), 'moods_unbalanced_subset_15615recs.tfrecord', 32, 128, 100, 'Models/LogReg_15615_32_128_30.h5')

    # LSTM 1-layer, small balanced data set:
    #train_model(lstm_1layer_model(), 'moods_balanced_subset_401recs.tfrecord', 20, 10, 100, 'Models/LSTM1_401_20_20_100.h5')
    # LSTM 1-layer, large unbalanced data set:
    #train_model(lstm_1layer_model(), 'moods_unbalanced_subset_15615recs.tfrecord', 32, 128, 100, 'Models/LSTM1_15615_32_128_30.h5')

    # LSTM 3-layer, small balanced data set:
    #train_model(lstm_3layer_model(), 'moods_balanced_subset_401recs.tfrecord', 20, 10, 200, 'Models/LSTM3_401_20_20_100.h5')
    # LSTM 3-layer, large unbalanced data set:
    #train_model(lstm_3layer_model(), 'moods_unbalanced_subset_15615recs.tfrecord', 32, 128, 100, 'Models/LSTM3_15615_32_128_30.h5')

    # NN, small balanced data set:
    #train_model(nn_model(), 'moods_balanced_subset_401recs.tfrecord', 32, 10, 250, 'Models/NN_401_20_20_100.h5')
    # NN, large unbalanced data set:
    #train_model(nn_model(), 'moods_unbalanced_subset_15615recs.tfrecord', 32, 128, 100, 'Models/NN_15615_32_128_30.h5')

if __name__ == "__main__":
    main()

