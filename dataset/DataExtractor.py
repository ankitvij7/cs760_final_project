import csv
import os
import numpy as np
import tensorflow as tf

moods = ["Happy music",
          "Funny music",
          "Sad music",
          "Tender music",
          "Exciting music",
          "Angry music",
          "Scary music"]


def find_labels(segment_file):
    with open('class_labels_indices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        # get the index and mid for each mood we're interested in
        labels = {}
        for row in reader:
            if (row[2].strip('"') in moods):
                labels[row[2]] = [int(row[0]), row[1], []] 

    # get which files we need to look for for each label
    ln = 0
    with open(segment_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ln += 1
            if ln == 137:
                n = 3
            if len(row) < 4:
                continue
            for lbl in moods:
                for n in range(3,len(row)):
                    if row[n].find(labels[lbl][1]) >= 0:
                        labels[lbl][2].append(row[0])
    for m in moods:
        print("'%s' is id %d and is found in %d files" % (m, int(labels[m][0]), len(labels[m][2])))
    return labels

def get_records_from(filename, label):
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
        example_label = list(np.asarray(tf_example.context.feature['labels'].int64_list.value))
        if (label in example_label):
            ret.append(tf_example)
        
    return ret

def build_tfrecords_for_moods(infile, outfile, dataloc):
    print("Building tfrecords...")
    labels = find_labels(infile)
    # for each mood, open the files and extract the features with the indices
    # matching that mood.
    # write all the features for the current mood to one file.
    sess = tf.Session()
    prefix = dataloc
    cnt = 0
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for m in moods:
            fnames = set()
            for vid in labels[m][2]: # all the video ids for this mood
                fnames.add(vid[0:2] + ".tfrecord") # don't duplicate filenames if file has 2 features of this mood
            for fname in fnames:
                recs = get_records_from(prefix + fname, labels[m][0])
                cnt += len(recs)
                for r in recs:
                    writer.write(r.SerializeToString())
    print("Saved %d records into '%s'" % (cnt, outfile))

def do_it():
    print(moods)
    build_tfrecords_for_moods('balanced_train_segments.csv', 'moods_balanced_subset.tfrecord', 'audioset_v1_embeddings/bal_train/')
    build_tfrecords_for_moods('unbalanced_train_segments.csv', 'moods_unbalanced_subset.tfrecord', 'audioset_v1_embeddings/unbal_train/')
