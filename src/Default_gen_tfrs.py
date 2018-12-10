from vggish_inference import vggish_inference
import time
import csv

#input_file = '../dataset/moods_balanced.csv'
input_file = '../dataset/moods_unbalanced_100each.csv'
#input_wav_prefix = '../dataset/wav_outputs_balanced/'
input_wav_prefix = '../dataset/wav_outputs_unbalanced/'
input_wav_suffix = '.wav'
output_tfr_prefix = 'gen_dataset/'
output_tfr_suffix = '.tfrecord'

start_time = time.time()

with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile, quotechar='"', delimiter=' ', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    num_rows = len(list(reader))
    row_cnt = 0;
    # reset the reader.
    csvfile.seek(0)
    for row in reader:
        row_cnt = row_cnt + 1
        print(row[3])
        input_wav = input_wav_prefix + row[3] + input_wav_suffix
        output_tfr = output_tfr_prefix + row[3] + output_tfr_suffix
        print('Start', input_wav, ', ', output_tfr)
        # Run this file, with this model.
        vggish_inference(input_wav, output_tfr)

        print("Finish, %d / %d,  %s seconds ---" % (row_cnt, num_rows, (time.time() - start_time)))

# Take 1
# import tensorflow as tf
# import vggish_postprocess
# import vggish_slim
# import vggish_params
# from vggish_inference import vggish_inference_fn
#
#
# # Prepare a postprocessor to munge the model embeddings.
# pproc = vggish_postprocess.Postprocessor(pca_params)
#
# with tf.Graph().as_default(), tf.Session() as sess:
#
#     # Define the model in inference mode, load the checkpoint
#     vggish_slim.define_vggish_slim(training=False)
#     vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
#
#     # Define our input and output tensors.
#     features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
#     embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
#
#
#     input_file = '../dataset/wav_outputs_balanced/_6uP5AkskmE_100.000.wav'
#     output_file = '_6uP5AkskmE.tfrecord'
#
#     # Run this file, with this model.
#     vggish_inference_fn(input_file, output_file, pproc, sess, features_tensor, embedding_tensor)
