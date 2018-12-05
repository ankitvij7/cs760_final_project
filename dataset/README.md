# CS760 Final Project - Dataset
Running DataExtractor.py  Will generate two files.

 # Google's AudioSet 
Google's already cleaned and separated DataSet:
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

Using `DataExtractor.py` will create two sets of filtered files:

Type `.param`: is the list of filtered youtube videos to fetch from, used with `download.sh`  
* `dataset/moods_balanced_subset.param`
* `dataset/moods_unbalanced_subset.param`
* Example: `$ cat moods_balanced_subset.param | ./download.sh 2> moods_balanced_subset_error.txt` 
* `./download.sh` needs to have two external tools, `youtube-dl` & `ffmpeg.exe / ffprobe.exe`
  * `youtube-dl`: can be installed via `pip install -U youtube-dl`
  * `ffmpeg / ffprobe`: install for your OS.  


Type `.tfrecord`: is the list of filtered already organized features. 
* `dataset/moods_balanced_subset.tfrecord`
* `dataset/moods_unbalanced_subset.tfrecord`


