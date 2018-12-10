import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Dropout
from keras.layers import LSTM, MaxPooling1D, Conv1D
from keras import regularizers
from keras.utils import plot_model
import sklearn.metrics

# validation set will be 10% of total set
CV_frac = 0.1
# comment out/in the labels you'd like to be filtered from the input and the#
# classification
# comment out/in the labels you'd like to be filtered from the input and the
# classification
mood_labels = {
    276: "Happy music",
    278: "Sad music",
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

def logistic_regression_model():
    ''' Creates a logistic regression model. Used by train_model '''
    lr_model = Sequential()
    lr_model.add(BatchNormalization(input_shape=(10, 128)))
    lr_model.add(Flatten())
    lr_model.add(Dense(len(mood_labels), activation='sigmoid'))
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
    # if interested, try using different optimizers and different optimizer
    # configs
    lstm_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return (lstm_model, '1-Layer LSTM')

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
    # if interested, try using different optimizers and different optimizer
    # configs
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
    # if interested, try using different optimizers and different optimizer #
    # configs
    # if interested, try using different optimizers and different optimizer
    # configs
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
    return history

def plot_epochs(history, title):
    # plot the results
    #plt.rcParams['figure.figsize'] = (12,8)
    sp = 131
    for series in zip(history,title):
        plt.subplot(sp)
        plt.plot(series[0].history['acc'], label='training accuracy')
        plt.plot(series[0].history['val_acc'], label='validation accuracy')
        plt.legend()
        plt.title(series[1][1])
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim([0,1])
        sp += 1
    plt.suptitle("Baseline Evaluation, 360 train/40 validate")
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
    # split records into train and test in a stratified way.
    classes = dict()
    for i in range(4):
        classes[i] = []
    for r in records:
        classes[r[1]].append(r)

    train = []
    test = []

    for i in range(4):
        sz = len(classes[i])
        split = int((1 - CV_frac) * sz)
        train += classes[i][0:split]
        test += classes[i][split:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    return [train, test]

def show_confusion_matrix(records, model, title):
    predictions = model.predict_on_batch(np.asarray([r[0] for r in records]))
    actual_class = [r[1] for r in records]
    predicted_class = [np.argmax(p) for p in predictions]
    correct = np.sum([p == a for (p,a) in zip(actual_class, predicted_class)])
    accuracy = correct / len(actual_class)
    conf = sklearn.metrics.confusion_matrix([r[1] for r in records], predicted_class)
    print("Test Results on model '" + title + "':")
    print("Accuracy: %.3f" % accuracy)
    print(" Confusion matrix:")
    print(conf)
    np.savetxt('confusion_baseline_' + title + '_raw.txt', conf)
    conf = conf / len(records)
    print(" Confusion matrix by percentage:")
    print(conf)
    np.savetxt('confusion_baseline_' + title + '_percent.txt', conf)
    print()

def main():

    # choose an input file by commenting in/out
    input_file = 'moods_unbalanced_100each.tfrecord'

    (train_records, validate_records) = load_records(input_file)
    num_records = len(train_records) + len(validate_records)

    print_label_stats("train", train_records)
    print_label_stats("validate", validate_records)

    # train 3 models
    models = [logistic_regression_model(), lstm_1layer_model(), lstm_3layer_model()]
    epochs = [100, 100, 200]
    results = []
    for (m,e) in zip(models, epochs):
        results.append(train_model(m[0], train_records, validate_records, 40, 40, e))

    plot_epochs(results, models)

    # test against eval 223 set
    (t1, t2) = load_records("moods_eval_223.tfrecord")
    test_records = t1 + t2
    for m in models:
        show_confusion_matrix(test_records, m[0], m[1])

if __name__ == "__main__":
    main()