#!/usr/bin/env python

import os
import io
import csv
import shutil
from datetime import datetime
import time
from struct import pack
from threading import Thread

import wave
import pyaudio
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
#from google.gax.errors import RetryError

#import rospy
#from std_msgs.msg import String, Header
#from ros_speech2text.msg import transcript, event

from speech_detection import SpeechDetector


FORMAT = pyaudio.paInt16


def list_audio_devices(pyaudio_handler):
    device_list = [pyaudio_handler.get_device_info_by_index(i)['name']
                   for i in range(pyaudio_handler.get_device_count())]
                   
    #for i in range(pyaudio_handler.get_device_count()):
    #   dev = pyaudio_handler.get_device_info_by_index(i)
    #   print((i,dev['name'],dev['maxInputChannels']))
    #rospy.logdebug('Available devices:' + ''.join(
    #    ['\n  - [%d]: %s' % d for d in enumerate(device_list)]))
    #rospy.set_param('/ros_speech2text/available_audio_device', device_list)
    return device_list


class SpeechRecognizer(object):

    #TOPIC_BASE = '/speech_to_text'

    class InvalidDevice(ValueError):
        pass

    def __init__(self):
        self._init_history_directory()
        #self.node_name = rospy.get_name()
        self.print_level = 2
        # If do_transcription = False, do not transcribe audio (and send a dummy transcript)
        #self.do_transcription =  rospy.get_param(self.node_name + '/do_transcription', True)
        self.do_transcription = True
        #self.pub_transcript = rospy.Publisher(
        #    self.TOPIC_BASE + '/transcript', transcript, queue_size=10)
        #self.pub_text = rospy.Publisher(
        #    self.TOPIC_BASE + '/text', String, queue_size=10)
        #self.pub_event = rospy.Publisher(
        #    self.TOPIC_BASE + '/log', event, queue_size=10)
        self.bedtime_recording_only = False
        self.sample_rate = 48000
        self.async = False
        dynamic_thresholding = False
        if not dynamic_thresholding:
            threshold = 9000
        else:
            threshold = 50
        self.speech_detector = SpeechDetector(
            self.sample_rate,
            threshold,
            dynamic_threshold=dynamic_thresholding,
            dynamic_threshold_frame=3,
            min_average_volume=100,
            n_silent=10,
        )
        #rospy.loginfo('Print level: {}'.format(self.print_level))
        #if self.print_level > 0:
        #    rospy.loginfo('Sample Rate: {}'.format(self.sample_rate))

        self._init_stream()
        self._init_csv()
        #self.speech_client = speech.Client()
        self.speech_client = speech.SpeechClient()
        self.run()

    def _init_history_directory(self):
        path_for_speech_history = "~/Documents/sleep-talking-detection/speech_files"
        date_and_time_folder_name = datetime.now().isoformat()
        #self.history_dir = os.path.expanduser(os.path.join(path_for_speech_history, str(os.getpid())))
        self.history_dir = os.path.expanduser(os.path.join(path_for_speech_history, date_and_time_folder_name))
        if not os.path.isdir(self.history_dir):
            os.makedirs(self.history_dir)

    def _init_stream(self):
        self.pa_handler = pyaudio.PyAudio()
        device_list = list_audio_devices(self.pa_handler)
        print("Device List:")
        print(device_list)
        print()
        #input_idx = rospy.get_param(self.node_name + '/audio_device_idx', None)
        #input_name = rospy.get_param(self.node_name + '/audio_device_name', None)
        input_idx = 2
        input_name = "USB Audio Device"
        if input_idx is None:
            input_idx = self.pa_handler.get_default_input_device_info()['index']
            if input_name is not None:
                try:
                    # use first found for name
                    input_idx = [input_name.lower() in d.lower()
                                 for d in device_list
                                 ].index(True)
                except ValueError:
                    self.terminate()
                    raise self.InvalidDevice(
                        "No device found for name '%s'." % input_name)
        try:
            #rospy.loginfo("{} using device: {}".format(
            #    self.node_name,
            #    self.pa_handler.get_device_info_by_index(input_idx)['name'])
            #)
            devinfo = self.pa_handler.get_device_info_by_index(2)
            #audio_sample_rate_to_test = 48000
            #~ if (self.pa_handler.is_format_supported(audio_sample_rate_to_test ,
                        #~ input_device=devinfo['index'],
                        #~ input_channels=devinfo['maxInputChannels'],
                        #~ input_format=pyaudio.paInt16)):
                #~ print("Audio sample rate of %i supported" % audio_sample_rate_to_test)
            #~ else:
                #~ print("Audio sample rate of %i NOT supported" % audio_sample_rate_to_test)
            self.stream = self.pa_handler.open(
                format=FORMAT, channels=1, rate=self.sample_rate, input=True,
                start=False, input_device_index=input_idx, output=False,
                frames_per_buffer=self.speech_detector.chunk_size)
        except IOError as e:
            self.terminate()
            print(e)
            raise self.InvalidDevice(
                'Invalid device ID: {}.'.format(input_idx))
        self.sample_width = self.pa_handler.get_sample_size(FORMAT)

    def _init_csv(self):
        # self.csv_file = open(os.path.join(self.history_dir, 'transcript'), 'wb')
        self.csv_file = open(os.path.join(self.history_dir, 'transcript'), 'w')
        self.csv_writer = csv.writer(self.csv_file, delimiter=' ',)
        self.csv_writer.writerow(['utterance_id', 'start', 'end', 'duration', 'transcript', 'confidence'])

    def run(self):
        sn = 0
        if self.async:
            self.operation_queue = []
            thread = Thread(target=self.check_operation)
            thread.start()
        #while not rospy.is_shutdown():
        # put the timing constraints here??? 
        try:
            while (True):
                is_past_bedtime = (datetime.now().hour > 22)
                is_before_wake_time = (datetime.now().hour < 5)
                if (((is_past_bedtime or is_before_wake_time) and self.bedtime_recording_only) or
                    not self.bedtime_recording_only):
                    aud_data, start_time, end_time = self.speech_detector.get_next_utter(
                        self.stream, *self.get_utterance_start_end_callbacks(sn))
                    if aud_data is None:
                        #rospy.loginfo("No more data, exiting...")
                        break
                    self.record_to_file(aud_data, sn)
                    if self.async:
                        operation = self.recog(sn)
                        if operation is not None:  # TODO: Improve
                            self.operation_queue.append([sn, operation, start_time, end_time])
                    else:
                        # Send only that you received speech if you don't want transcriptions.
                        if not self.do_transcription:
                            transc, confidence = ("dummy_transcript with no confidence", 0.0)
                        else:
                            recog_thread = Thread(target=self.recog, args=(sn, start_time, end_time, ))
                            recog_thread.start()
                            #~ transc, confidence = self.recog(sn, start_time, end_time)
                            #~ if (transc is not None):
                                #~ self.utterance_decoded(sn, transc, confidence, start_time, end_time)
                            #~ else:
                                #~ self.utterance_decoded(sn, "## speech not recognized ##", 0.0, start_time, end_time)
                    sn += 1
                else:
                    print(datetime.now().isoformat())
                    time.sleep(60)
        except KeyboardInterrupt:
            print("W: interrupt received, stopping")
        finally:
            self.terminate()

    def terminate(self):
        if hasattr(self, "stream"):
            self.stream.close()
        if hasattr(self, "pa_handler"):
            self.pa_handler.terminate()
        if hasattr(self, "csv_file"):
            self.csv_file.close()
        #~ if (hasattr(self, "history_dir")):
            #~ shutil.rmtree(self.history_dir)

    def utterance_start(self, utterance_id):
        if self.print_level > 1:
            #rospy.loginfo('Utterance started')
            print("Utterance " + str(utterance_id) + " started")
        #self.pub_event.publish(
            #self.get_event_base_message(event.STARTED, utterance_id))

    def utterance_end(self, utterance_id):
        if self.print_level > 1:
            #rospy.loginfo('Utterance completed')
            print("Utterance " + str(utterance_id) + " completed")
        #self.pub_event.publish(
            #self.get_event_base_message(event.STOPPED, utterance_id))

    def get_utterance_start_end_callbacks(self, utterance_id):
        def start():
            self.utterance_start(utterance_id)

        def end():
            self.utterance_end(utterance_id)

        return start, end

    def utterance_decoded(self, utterance_id, transcription, confidence,
                          start_time, end_time):
        #~ transcript_msg = self.get_transcript_message(transcription, confidence,
                                                     #~ start_time, end_time)
        #event_msg = self.get_event_base_message(event.DECODED, utterance_id)
        #event_msg.transcript = transcript_msg
        if self.print_level > 0:
            print('=' * 20)
            print("transcript: {}".format(transcription))
            print("confidence: {}".format(confidence))
        #    rospy.loginfo("{} [confidence: {}]".format(transcription, confidence))
        #self.pub_transcript.publish(transcript_msg)
        #self.pub_text.publish(transcription)
        #self.pub_event.publish(event_msg)
        self.csv_writer.writerow([
            utterance_id, start_time, end_time, (end_time - start_time),
            transcription, confidence])

    def utterance_failed(self, utterance_id, start_time, end_time):
        if self.print_level > 1:
            #rospy.loginfo("No good results returned!")
            print("No good results resturned")
        #transcript_msg = self.get_transcript_message("", 0., start_time, end_time)
        #event_msg = self.get_event_base_message(event.FAILED, utterance_id)
        #event_msg.transcript = transcript_msg
        #self.pub_event.publish(event_msg)

    #~ def get_transcript_message(self, transcription, confidence, start_time,
                               #~ end_time):
        #~ msg = transcript()
        #~ msg.start_time = start_time
        #~ msg.end_time = end_time
        #~ msg.speech_duration = end_time - start_time
        #~ msg.received_time = rospy.get_rostime()
        #~ msg.transcript = transcription
        #~ msg.confidence = confidence
        #~ return msg

#    def get_event_base_message(self, evt, utterance_id):
#        msg = event()
#        msg.header = Header()
#        msg.header.stamp = rospy.Time.now()
#        msg.event = evt
#        msg.utterance_id = utterance_id
#        msg.audio_path = self.utterance_file(utterance_id)
#        return msg

    def utterance_file(self, utterance_id):
        file_name = 'utterance_{}.wav'.format(utterance_id)
        return os.path.join(self.history_dir, file_name)

    def record_to_file(self, data, utterance_id):
        """Saves audio data to a file"""
        data = pack('<' + ('h' * len(data)), *data)
        path = self.utterance_file(utterance_id)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()
        #rospy.logdebug('File saved to {}'.format(path))

    def recog(self, utterance_id, utterance_start_time, utterance_end_time):
        """
        Constructs a recog operation with the audio file specified by sn
        The operation is an asynchronous api call
        """
        #~ context = rospy.get_param(self.node_name + '/speech_context', [])
        #~ path = self.utterance_file(utterance_id)

        #~ with io.open(path, 'rb') as audio_file:
            #~ content = audio_file.read()
            #~ audio_sample = self.speech_client.sample(
                #~ content,
                #~ source_uri=None,
                #~ encoding='LINEAR16',
                #~ sample_rate=self.sample_rate)

        #~ if self.async:
            #~ try:
                #~ operation = self.speech_client.speech_api.async_recognize(
                    #~ sample=audio_sample, speech_context=context)
                #~ return operation
            #~ except (ValueError, RetryError) as e:
                #~ rospy.logerr(e)
                #~ rospy.logerr("Audio Segment too long. Unable to recognize")
        #~ else:
            #~ alternatives = self.speech_client.speech_api.sync_recognize(
                #~ sample=audio_sample, speech_context=context)
            #~ for alternative in alternatives:
                #~ return alternative.transcript, alternative.confidence
                
        path = self.utterance_file(utterance_id)
        
        with io.open(path, 'rb') as audio_file:
            content = audio_file.read()
            
        audio = speech.types.RecognitionAudio(content=content)
            
        config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code='en-US',
            sample_rate_hertz=self.sample_rate)
            
        if self.async:
            try: 
                operation = self.speech_client.long_running_recognize(config=config, audio=audio)
                return operation
            except (ValueError, RetryError) as e:
                print(e)
        else:
            response = self.speech_client.recognize(config=config, audio=audio)
            for result in response.results:
                self.utterance_decoded(utterance_id, result.alternatives[0].transcript, 
                    result.alternatives[0].confidence, utterance_start_time, utterance_end_time)
                return
                #~ return (result.alternatives[0].transcript, result.alternatives[0].confidence)
            #~ return (None, None)
            self.utterance_decoded(utterance_id, "## speech not recognized ##", 
                0.0, utterance_start_time, utterance_end_time)
            return
                

 #   def check_operation(self):
 #       """
 #       This function is intended to be run as a seperate thread that repeatedly
 #       checks if any recog operation has finished.
 #       The transcript returned is then published on screen of baxter and sent
 #       to the ros topic with the custom message type 'transcript'.
 #       """
 #       while not rospy.is_shutdown():
 #           try:
 #               for op in self.operation_queue[:]:
 #                   utterance_id, operation, start_time, end_time = op
 #                   if operation.complete and operation.results is not None:
 #                       for result in operation.results:
 #                           self.utterance_decoded(
 #                               utterance_id, result.transcript, result.confidence,
 #                               start_time, end_time)
 #                       self.operation_queue.remove(op)
 #                   else:
 #                       try:
 #                           operation.poll()
 #                       except ValueError:
 #                           self.utterance_failed(utterance_id, start_time, end_time)
 #                           self.operation_queue.remove(op)
 #           except Exception as e:
 #               rospy.logerr("Error in speech recognition thread: {}".format(e))
 #               self.operation_queue = []
 #           rospy.sleep(1)
