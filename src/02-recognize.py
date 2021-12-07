import collections
import subprocess
import os
import sys
import numpy as np
import time
from numpy.core.fromnumeric import argmax 

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import io_ops

import params
from spectrograms import SpectrogramLoader

from data import params as aaa

from audio import params as apm
from ml import siamese

dry_run = True 

# Taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/recognize_commands.py
class RecognizeResult(object):
    """Save recognition result temporarily.
    Attributes:
        founded_command: A string indicating the word just founded. Default value
        is '_silence_'
        score: A float representing the confidence of founded word. Default
        value is zero.
        is_new_command: A boolean indicating if the founded command is a new one
        against the last one. Default value is False.
    """
    def __init__(self, founded_command="_silence_", score=0.0, is_new_command=False):
        """Construct a recognition result.
        Args:
            founded_command: A string indicating the word just founded.
            score: A float representing the confidence of founded word.
            is_new_command: A boolean indicating if the founded command is a new one
            against the last one.
        """
        self._founded_command = founded_command
        self._score = score
        self._is_new_command = is_new_command
        

    @property
    def founded_command(self):
        return self._founded_command

    @founded_command.setter
    def founded_command(self, value):
        self._founded_command = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def is_new_command(self):
        return self._is_new_command

    @is_new_command.setter
    def is_new_command(self, value):
        self._is_new_command = value

class Recognizer(): 
    def __init__(self, 
                labels, 
                average_window_duration_ms, 
                detection_threshold, 
                suppression_ms, 
                minimum_count): 
        
        self._labels = labels
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count

        self._previous_results = collections.deque()
        self._label_count = len(labels)
        self._previous_top_label = "_silence_"
        self._previous_top_time = -np.inf
        
        # self.transmission_pid = ""
        self.receiver_pid = "" 
        self.setup()
    
    def detection_threshold(self): 
        return self._detection_threshold
        
    # def start_transmission(self):
    #     if not dry_run: 
    #         print("Starting transmission")
    #         self.transmission_pid = subprocess.Popen(["python3", "transmit.py"])

    def start_recording(self): 
        if not dry_run: 
            print("Starting recording")
            self.receiver_pid = subprocess.Popen(["python3", "receive.py"])

    def setup(self):     
        # self.start_transmission()
        self.start_recording() 
        print("sleeping")
        time.sleep(11)
    
    def stop(self): 
        self.transmission_pid.kill() 
        self.receiver_pid.kill() 
        
    def process_latest_result(self, latest_results, current_time_ms, recognize_element): 
        if latest_results.shape[0] != self._label_count:
            raise ValueError("The results for recognition should contain {} "
                    "elements, but there are {} produced".format(
                        self._label_count, latest_results.shape[0]))
        if (self._previous_results.__len__() != 0 and
            current_time_ms < self._previous_results[0][0]):
            raise ValueError("Results must be fed in increasing time order, "
                            "but receive a timestamp of {}, which was earlier "
                            "than the previous one of {}".format(
                                current_time_ms, self._previous_results[0][0]))

        # Add the latest result to the head of the deque.
        self._previous_results.append([current_time_ms, latest_results])

        # Prune any earlier results that are too old for the averaging window.
        time_limit = current_time_ms - self._average_window_duration_ms
        while time_limit > self._previous_results[0][0]:
            self._previous_results.popleft()
        
        # If there are too few results, the result will be unreliable and bail.
        how_many_results = self._previous_results.__len__()
        earliest_time = self._previous_results[0][0]
        sample_duration = current_time_ms - earliest_time
        if (how_many_results < self._minimum_count or
            sample_duration < self._average_window_duration_ms / 4):
            recognize_element.founded_command = self._previous_top_label
            recognize_element.score = 0.0
            recognize_element.is_new_command = False
            return
    
        # Calculate the average score across all the results in the window.
        average_scores = np.zeros(self._label_count)
        for item in self._previous_results:
            score = item[1]
            for i in range(score.size):
                average_scores[i] += score[i] / how_many_results

        # Sort the averaged results in descending score order.
        sorted_averaged_index_score = []
        for i in range(self._label_count):
            sorted_averaged_index_score.append([i, average_scores[i]])
        
        sorted_averaged_index_score = sorted(
            sorted_averaged_index_score, key=lambda p: p[1], reverse=True)
        
        # Use the information of previous result to get current result
        current_top_index = sorted_averaged_index_score[0][0]
        current_top_label = self._labels[current_top_index]
        current_top_score = sorted_averaged_index_score[0][1]
        time_since_last_top = 0
        
        if (self._previous_top_label == "_silence_" or
            self._previous_top_time == -np.inf):
            time_since_last_top = np.inf
        else:
            time_since_last_top = current_time_ms - self._previous_top_time
        
        if (current_top_score > self._detection_threshold and
            current_top_label != self._previous_top_label and
            time_since_last_top > self._suppression_ms):
            self._previous_top_label = current_top_label
            self._previous_top_time = current_time_ms
            recognize_element.is_new_command = True
        else:
            recognize_element.is_new_command = False
            
        recognize_element.founded_command = current_top_label
        recognize_element.score = current_top_score
    

def predict(i, model, recognizer, recognize_element, loader):     
    compare_set = "val"
    current_wav_path = apm.live_save_directory + "/" + apm.live_wav_filename + "_" + str(i) + ".wav"
    spectrogram_loader = SpectrogramLoader("", "", current_wav_path)
    
    spectrogram_save_path = apm.live_save_directory + "/" + str(i) + aaa.data_type
    spectrogram_loader.get_spectrogram(spectrogram_save_path)
    
    keys = loader.keys[compare_set]
    all_probs = loader.test_live(model, 
                                len(params.default_gestures), 
                                params.n_prediction_per, 
                                spectrogram_save_path, 
                                compare_set)
    
    zipped = zip(all_probs, keys)
    zipped_sorted = sorted(zipped)
    print(zipped_sorted)
    
    s = sorted(all_probs, reverse=True)
    if (s[0] > recognizer.detection_threshold()):
        idx = np.argmax(all_probs)
        label = keys[idx]
        print("predicted gesture: ", label, " with probability: ", all_probs[idx])
        
    
    # audio_data_offset = i
    # sample_rate = spectrogram_loader.sample_rate()
    # current_time_ms = int(audio_data_offset * 1000 / sample_rate)
    # try:
    #     recognizer.process_latest_result([ all_probs ], current_time_ms, recognize_element)
    
    # except ValueError as e:
    #     tf.compat.v1.logging.error('Recognition processing failed: {}' % e)
    #     return
    
    
def start(): 
    recognizer = Recognizer(
        params.default_gestures,
        params.average_window_ms, 
        params.detection_threshold, 
        params.suppression_ms, 
        params.minimum_count
    )
    
    data_path = os.path.join(params.predict_run_path, "images", params.predict_spectrogram_type) + "/"
    loader = siamese.OneshotLoader(data_path, params.predict_spectrogram_type, params.predict_run_path)
    
    model = keras.models.load_model(loader.model_path)
    print(model.summary())

    recognize_element = RecognizeResult()
    
    print("Predicting")

    i = 0
    # Inference along audio stream.
    while True: 
        try: 
            print("Iteration: ", i)
            predict(i, model, recognizer, recognize_element, loader)

        except KeyboardInterrupt:
            recognizer.stop()
            sys.exit(0)
        
        i += 1 
        time.sleep(1)
    
def main():  
    start()
    
main()
            