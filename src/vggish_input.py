# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy

import mel_features
import vggish_params

import soundfile as sf


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
      #IKN - num_examples = Length of audio input / (NUM_FRAMES * STFT_HOP_LENGTH_SECONDS) (def 10/(96*0.01) = 10.5 = 11)
      #IKN - num_frames = NUM_FRAMES (def 96)
      #IKN - num_bands  = NUM_BANDS (def 64)
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # IKN - (2 x n) -> (1 x n)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)  # IKN - (1 x n) -> (1 x smaller then n = q)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    # IKN - 10s of (STFT_WINDOW_LENGTH_SECONDS(25ms) frames with STFT_HOP_LENGTH_SECONDS(10ms) hops) x NUM_BANDS(65) spectrograph

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS  # IKN - 100
    example_window_length = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))  # IKN - 96
    example_hop_length = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))  # IKN - 96
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    # IKN - 3d- (10s/0.96 x 0.96 second of frames x NUM_MEL_BINS)

    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)
