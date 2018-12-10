import csv
import os
import numpy as np
import tensorflow as tf

moods = ["Happy music",
          "Sad music",
          "Angry music",
          "Scary music"]

def get_record_from(filename, label):
    if not os.path.isfile(filename):
        print("File not found (skipping): ", filename)
        return []

    ri = tf.python_io.tf_record_iterator(path=filename)
    n = 1
    ret = []
    for string_record in ri: 
        n += 1

        tf_example = tf.train.SequenceExample.FromString(string_record)
        vid_id = tf_example.context.feature['video_id'].bytes_list.value[0].decode(encoding = 'UTF-8')
        if vid_id == label:
            print('found record ', vid_id, ' in file ', filename)
            return tf_example
        
    return None

def build_tfrecords_for_moods(infile, outfile, dataloc):
    print("Building tfrecords...")
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        # get the index and mid for each mood we're interested in
        labels = [r[0] for r in reader]

    #labels = find_labels(infile)
    # for each mood, open the files and extract the features with the indices
    # matching that mood.
    # write all the features for the current mood to one file.
    sess = tf.Session()
    prefix = dataloc
    cnt = 0
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for vid in labels:
            fname = prefix + vid[0:2] + ".tfrecord"
            rec = get_record_from(fname, vid)
            if rec:
                cnt += 1
                writer.write(rec.SerializeToString())
    print("Saved %d records into '%s'" % (cnt, outfile))

def do_it():
    print(moods)
    #build_tfrecords_for_moods('balanced_train_segments.csv', 'moods_balanced_subset.tfrecord', 'audioset_v1_embeddings/bal_train/')
    #build_tfrecords_for_moods('unbalanced_train_segments.csv', 'moods_unbalanced_subset.tfrecord', 'audioset_v1_embeddings/unbal_train/')
    build_tfrecords_for_moods('moods_unbalanced_100each.csv', 'moods_unbalanced_100each.tfrecord', 'audioset_v1_embeddings/unbal_train/')
    build_tfrecords_for_moods('moods_balanced.csv', 'moods_balanced_220.tfrecord', 'audioset_v1_embeddings/bal_train/')

do_it()