# cs760_final_project
Final Project for CS-760 at UW Madison.

#Initial DataSet
Fetched from Googles AudioSet:
* https://research.google.com/audioset/download.html

Features where extracted from raw audio to modified VGG:
* https://github.com/tensorflow/models/tree/master/research/audioset

The following files are used:
* `dataset/eval_segments.csv`
* `dataset/balanced_train_segments.csv`
* `dataset/unbalanced_train_segments.csv`
* Pre-VGG processed features `dataset/features.tar.gz` extract to:
  * `dataset/audioset_v1_embeddings/eval/*.tfrecord`
  * `dataset/audioset_v1_embeddings/bal_train/*.tfrecord`
  * `dataset/audioset_v1_embeddings/unbal_train/*tfrecord`

This data is quite large so download them to your local disk and place them in the above directories.


# Tools:
Major packages have sub-dependencies ignored for brevity:

Package | Version
--- | ---
python | **3.6.7**
tensorflow | **1.12.0**
numpy | **1.15.4**
keras | **2.2.4**
matplotlib | **3.0.2**
soundfile | **0.10.2**
resampy | **0.2.1**

Instructions on how to install TensorFlow using pip:
* https://www.tensorflow.org/install/pip
