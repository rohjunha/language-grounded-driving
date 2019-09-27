#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
import json
import shutil
import wave
from multiprocessing import Queue, Process, Event
from operator import attrgetter, itemgetter
from pathlib import Path
from time import sleep
from typing import Dict

import cv2
import pyaudio
from moviepy import editor
from six.moves import queue

from data.types import DriveDataFrame
from environment import set_world_asynchronous, set_world_synchronous, GameEnvironment
from evaluator import ExperimentArgument, load_param_and_evaluator, load_evaluation_dataset, \
    get_random_sentence_from_keyword, listen_keyboard
from util.common import add_carla_module, get_logger, get_current_time, unique_with_islices
from util.directory import EvaluationDirectory, mkdir_if_not_exists

add_carla_module()
logger = get_logger(__name__)
import carla

import numpy as np
import pygame
import torch
import sys
import re

from config import EVAL_FRAMERATE_SCALE
from data.dataset import generate_templated_sentence_dict
from util.road_option import fetch_high_level_command_from_index
from util.image import video_from_files

__keyword_from_input__ = {
    'j': 'left',
    'k': 'straight',
    'l': 'right',
    'u': 'left,left',
    'i': 'left,straight',
    'o': 'left,right',
    'm': 'right,left',
    ',': 'right,straight',
    '.': 'right,right',
    '1': 'straight,straight',
    '2': 'firstleft',
    '3': 'firstright',
    '4': 'secondleft',
    '5': 'secondright'
}
__input_from_keyword__ = {v: k for k, v in __keyword_from_input__.items()}

__sentence_library_dict__ = generate_templated_sentence_dict()


RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
STREAMING_LIMIT = 10000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
from google.cloud import speech_v1p1beta1 as speech
from multiprocessing import Manager


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size, audio_path_func, audio_dict, event):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
        self.audio_path_func = audio_path_func
        self.event = event
        self.audio_with_timing = audio_dict

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
        self.event.set()

    def _save_audio(self, timestamp, in_data):
        # timestamp = get_current_time()
        waveFile = wave.open(str(self.audio_path_func(timestamp)), 'wb')
        waveFile.setnchannels(self._num_channels)
        waveFile.setsampwidth(self._audio_interface.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(self._rate)
        waveFile.writeframes(in_data)
        waveFile.close()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        # for timestamp, audio in mic_manager.audio_with_timing.items():
        # self.audio_with_timing[timestamp] = in_data
        # self._save_audio(in_data)
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed and not self.event.is_set():
            data = []
            timestamp = get_current_time()
            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                                                  * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            final_data = b''.join(data)
            self._save_audio(timestamp, final_data)
            yield final_data


def listen_print_loop(responses, stream, queue):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream.result_end_time = int((result_seconds * 1000)
                                     + (result_nanos / 1000000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')
            queue.put(transcript)

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write('Exiting...\n')
                stream.closed = True
                break

        else:
            sys.stdout.write(RED)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

            stream.last_transcript_was_final = False


def generate_video_with_audio(data):
    root_dir, traj_index = data
    # root_dir = Path.home() / 'projects/language-grounded-driving/.carla/evaluations/exp40/ls-town2/step072500/online'
    # traj_index = 0
    video_dir = root_dir / 'videos'
    audio_dir = root_dir / 'audios'
    state_dir = root_dir / 'states'
    image_dir = root_dir / 'images'
    timing_path = audio_dir / 'timing{:02d}.json'.format(traj_index)
    state_path = state_dir / 'traj{:02d}.json'.format(traj_index)
    out_vid_path = video_dir / 'traj{:02d}r.mp4'.format(traj_index)

    def image_path_func(frame: int):
        return image_dir / '{:08d}e.png'.format(frame)

    def generate_text_clips(state_path: Path, frame_timing_dict):
        def fetch_sequences_online(seq_path: Path):
            with open(str(seq_path), 'r') as file:
                data = json.load(file)
            xs, ys = zip(*list(map(lambda x: tuple([float(v) for v in x.split(',')[:2]]), data['data_frames'])))
            frame_range = data['frame_range']
            sentences = data['sentences']
            subgoals = list(map(itemgetter(1), data['stop_frames']))
            return xs, ys, frame_range, sentences, subgoals

        xs, ys, frame_range, sentences, subgoals = fetch_sequences_online(state_path)
        sentence_index_list = unique_with_islices(sentences)
        subtask_index_list = unique_with_islices(subgoals)

        def _generate_text_clips(text_index_list, prefix, fontsize, pos, color='white'):
            clips = []
            for sentence, local_frame_range in text_index_list:
                local_frame_range = [v + frame_range[0] for v in local_frame_range]
                start_time, end_time = None, None
                for frame in range(*local_frame_range):
                    if frame not in frame_timing_dict:
                        continue
                    timing = frame_timing_dict[frame]
                    if start_time is None or timing < start_time:
                        start_time = timing
                    if end_time is None or timing > end_time:
                        end_time = timing
                if start_time is None or end_time is None or start_time > end_time:
                    print('failed to find proper timing from {}, {}'.format(sentence, local_frame_range))
                text_clip = editor.TextClip(
                    prefix + sentence + ' ', fontsize=fontsize, color=color, bg_color='black', font='Lato-Medium')
                text_clip = text_clip.set_start(start_time, False).set_end(end_time)
                text_clip = text_clip.set_position(pos)
                clips.append(text_clip)
            return clips

        clips = _generate_text_clips(sentence_index_list, ' sentence: ', 27, (30, 360)) + \
                _generate_text_clips(subtask_index_list, ' sub-task : ', 27, (30, 393), 'yellow')
        return clips

    with open(str(timing_path), 'r') as file:
        data = json.load(file)
    frame_timing_dict = {int(key): data[key] for key in data.keys()}
    frame_list = sorted(frame_timing_dict.keys())
    frame_list = list(filter(lambda x: image_path_func(x).exists(), frame_list))
    raw_timing_list = [frame_timing_dict[f] for f in frame_list]
    t1, t2 = raw_timing_list[0], raw_timing_list[-1]

    audio_file_list = sorted(audio_dir.glob('*.wav'))
    audio_file_list = list(filter(lambda x: t1 < int(x.stem) < t2, audio_file_list))
    audio_timing_list = list(map(lambda x: int(x.stem), audio_file_list))
    print(audio_timing_list[0], audio_timing_list[-1])

    timing_offset = frame_timing_dict[frame_list[0]]
    image_timing_list = [(frame_timing_dict[f] - timing_offset) / 1e3 for f in frame_list]
    audio_timing_list = [(t - timing_offset) / 1e3 for t in audio_timing_list]
    for key in frame_timing_dict.keys():
        frame_timing_dict[key] = (frame_timing_dict[key] - timing_offset) / 1e3

    image_duration_list = [t2 - t1 for t1, t2 in zip(image_timing_list[:-1], image_timing_list[1:])]
    image_duration_list.append(image_duration_list[-1])
    image_clip_list = []
    for frame, image_timing, image_duration in zip(frame_list, image_timing_list, image_duration_list):
        clip = editor.ImageClip(str(image_path_func(frame)))
        clip = clip.set_duration(image_duration)
        clip = clip.set_start(image_timing, True)
        image_clip_list.append(clip)
    audio_clip_list = []
    for audio_file, audio_timing in zip(audio_file_list, audio_timing_list):
        try:
            clip = editor.AudioFileClip(str(audio_file))
            clip = clip.set_start(audio_timing, True)
            audio_clip_list.append(clip)
        except:
            print('failed in {}'.format(audio_file))
            continue

    text_clips = generate_text_clips(state_path, frame_timing_dict)
    cvc = editor.CompositeVideoClip(image_clip_list + text_clips)
    cac = editor.CompositeAudioClip(audio_clip_list)
    cvc = cvc.set_fps(30)
    cvc = cvc.set_audio(cac)
    cvc.write_videofile(str(out_vid_path))
    logger.info('wrote a video file {}'.format(out_vid_path))


def generate_video_from_clips(queue, event):
    while not event.is_set():
        if not queue.empty():
            data_tuple = queue.get()
            if data_tuple is None:
                break
            generate_video_with_audio(data_tuple)
            # cvc = editor.CompositeVideoClip(data_dict['video'] + data_dict['text'])
            # cvc = cvc.set_fps(30)
            # cac = editor.CompositeAudioClip(data_dict['audio'])
            # cvc = cvc.set_audio(cac)
            # cvc.write_videofile(str(data_dict['path']))
            # logger.info('write video file {}'.format(data_dict['path']))
    event.set()


def launch_recognizer(queue, event, audio_path_func, audio_dict, audio_setup_dict):
    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE, audio_path_func, audio_dict, event)
    audio_setup_dict['setnchannels'] = mic_manager._num_channels
    audio_setup_dict['setsampwidth'] = mic_manager._audio_interface.get_sample_size(pyaudio.paInt16)
    audio_setup_dict['setframerate'] = mic_manager._rate

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:
        while not stream.closed and not event.is_set():
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(
                STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content) for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream, queue)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True
    event.set()

    # for timestamp, audio in mic_manager.audio_with_timing.items():
    #     waveFile = wave.open(str(audio_path_func(timestamp)), 'wb')
    #     waveFile.setnchannels(mic_manager._num_channels)
    #     waveFile.setsampwidth(mic_manager._audio_interface.get_sample_size(pyaudio.paInt16))
    #     waveFile.setframerate(mic_manager._rate)
    #     waveFile.writeframes(audio)
    #     waveFile.close()


class SpeechEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, language_queue, video_queue, event, audio_dict, audio_setup_dict, args):
        self.eval_keyword = eval_keyword
        args.show_game = True
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')

        self.event = event
        self.language_queue = language_queue
        self.video_queue = video_queue

        # load params and evaluators
        self.eval_name = args.eval_name
        self.control_param, self.control_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='control')
        self.stop_param, self.stop_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='stop')
        self.high_param, self.high_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='high')

        # set image type
        self.image_type = self.high_param.image_type
        if 'd' in self.image_type:
            from model import DeepLabModel, prepare_deeplab_model
            self.deeplab_model: DeepLabModel = prepare_deeplab_model()

        self.final_images = []
        self.eval_dataset, self.eval_sentences = load_evaluation_dataset(self.high_param)
        self.eval_transforms = list(map(lambda x: x[0].state.transform, self.eval_dataset))
        self.high_sentences = self.eval_sentences
        self.softmax = torch.nn.Softmax(dim=1)
        EvaluationDirectory.__init__(self, *self.eval_info)
        self.high_data_dict = dict()
        self.audio_dict = audio_dict
        self.audio_setup_dict = audio_setup_dict

    @property
    def eval_info(self):
        return self.control_param.exp_index, self.eval_name, \
               self.control_evaluator.step, 'online'

    @property
    def segment_image(self):
        return np.reshape(((self.agent.segment_frame[:, :, 2] == 7).astype(dtype=np.uint8) * 255), (88, 200, 1))

    @property
    def custom_segment_image(self):
        return np.reshape(self.deeplab_model.run(self.agent.image_frame), (88, 200, 1))

    @property
    def final_image(self):
        if self.image_type == 's':
            return self.segment_image
        elif self.image_type == 'd':
            return self.custom_segment_image
        elif self.image_type == 'bgr':
            return self.agent.image_frame
        elif self.image_type == 'bgrs':
            return np.concatenate((self.agent.image_frame, self.segment_image), axis=-1)
        elif self.image_type == 'bgrd':
            return np.concatenate((self.agent.image_frame, self.custom_segment_image), axis=-1)
        else:
            raise TypeError('invalid image type {}'.format(self.image_type))

    def export_video(self, t: int, camera_keyword: str, curr_eval_data: dict):
        _, sub_goals = zip(*curr_eval_data['stop_frames'])
        texts = ['sentence: {}\nsub-task: {}'.format(s, g)
                 for g, s in zip(sub_goals, curr_eval_data['sentences'])]
        text_dict = {i: t for i, t in zip(range(*curr_eval_data['frame_range']), texts)}
        src_image_files = [self.agent.image_path(f, camera_keyword) for f in range(*curr_eval_data['frame_range'])]
        src_image_files = list(filter(lambda x: x.exists(), src_image_files))
        image_frames = set([int(s.stem[:-1]) for s in src_image_files])
        drive_frames = set(text_dict.keys())
        common_frames = sorted(list(image_frames.intersection(drive_frames)))
        src_image_files = [self.agent.image_path(f, camera_keyword) for f in common_frames]
        dst_image_files = [self.image_dir / p.name for p in src_image_files]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]
        text_list = [text_dict[f] for f in common_frames]
        video_from_files(src_image_files, self.video_path(t, camera_keyword),
                         texts=text_list, framerate=30, revert=True)

    def export_segment_video(self, t: int):
        final_image_files = [self.segment_dir / '{:08d}.png'.format(i) for i in range(len(self.final_images))]
        logger.info('final_image_files {}'.format(len(final_image_files)))
        for p, s in zip(final_image_files, self.final_images):
            cv2.imwrite(str(p), s)
        video_from_files(final_image_files, self.video_dir / 'segment{:02d}.mp4'.format(t),
                         texts=[], framerate=30, revert=False)

    def export_evaluation_data(self, t: int, curr_eval_data: dict) -> bool:
        with open(str(self.state_path(t)), 'w') as file:
            json.dump(curr_eval_data, file, indent=2)

        data_frames = [DriveDataFrame.load_from_str(s) for s in curr_eval_data['data_frames']]
        controls = list(map(attrgetter('control'), data_frames))
        stops, sub_goals = zip(*curr_eval_data['stop_frames'])
        logger.info('controls, stops, goals {}, {}, {}'.format(len(controls), len(stops), len(sub_goals)))

        timing_dict = self.export_timing_dict(t)
        self.export_video(t, 'center', curr_eval_data)
        self.export_video(t, 'extra', curr_eval_data)
        self.export_segment_video(t)
        self.export_video_with_audio(t, timing_dict, curr_eval_data)

        return self.state_path(t).exists()

    def export_timing_dict(self, t: int):
        target_sensor = None
        if 'extra' in self.agent.camera_sensor_dict:
            target_sensor = self.agent.camera_sensor_dict['extra']
        elif 'center' in self.agent.camera_sensor_dict:
            target_sensor = self.agent.camera_sensor_dict['center']
        if target_sensor is None:
            return dict()
        with open(str(self.timing_path(t)), 'w') as file:
            json.dump(target_sensor.timing_dict, file, indent=4)
        return target_sensor.timing_dict

    def export_video_with_audio(self, t: int, frame_timing_dict, curr_eval_data):
        out_vid_path = self.video_dir / 'traj{:02d}r.mp4'.format(t)

        def image_path_func(frame: int):
            return self.image_dir / '{:08d}e.png'.format(frame)

        frame_list = sorted(frame_timing_dict.keys())
        frame_list = list(filter(lambda x: image_path_func(x).exists(), frame_list))
        raw_timing_list = [frame_timing_dict[f] for f in frame_list]
        t1, t2 = raw_timing_list[0], raw_timing_list[-1]

        # audio_timing_list = sorted(self.audio_dict.keys())
        # audio_timing_list = list(filter(lambda x: t1 < x < t2, audio_timing_list))
        # for audio_timing in audio_timing_list:
        #     audio_value = self.audio_dict[audio_timing]
        #     waveFile = wave.open(str(self.audio_path(audio_timing)), 'wb')
        #     waveFile.setnchannels(self.audio_setup_dict['setnchannels'])
        #     waveFile.setsampwidth(self.audio_setup_dict['setsampwidth'])
        #     waveFile.setframerate(self.audio_setup_dict['setframerate'])
        #     waveFile.writeframes(audio_value)
        #     waveFile.close()

        self.video_queue.put((self.root_dir, t))

        # cvc = editor.CompositeVideoClip(image_clip_list + text_clips)
        # cvc = cvc.set_fps(30)
        #
        # cac = editor.CompositeAudioClip(audio_clip_list)
        # logger.info('generated audio clips {}'.format(len(audio_clip_list)))
        #
        # cvc = cvc.set_audio(cac)
        # cvc.write_videofile(str(out_vid_path))

        # self.audio_dict = dict()

    # def export_video_with_audio(self, t: int, frame_timing_dict, curr_eval_data):
    #     out_vid_path = self.video_dir / 'traj{:02d}r.mp4'.format(t)
    #
    #     def image_path_func(frame: int):
    #         return self.image_dir / '{:08d}e.png'.format(frame)
    #
    #     def generate_text_clips():
    #         frame_range = curr_eval_data['frame_range']
    #         sentences = curr_eval_data['sentences']
    #         subgoals = list(map(itemgetter(1), curr_eval_data['stop_frames']))
    #         sentence_index_list = unique_with_islices(sentences)
    #         subtask_index_list = unique_with_islices(subgoals)
    #
    #         def _generate_text_clips(text_index_list, prefix, fontsize, pos, color='white'):
    #             clips = []
    #             for sentence, local_frame_range in text_index_list:
    #                 local_frame_range = [v + frame_range[0] for v in local_frame_range]
    #                 start_time, end_time = None, None
    #                 for frame in range(*local_frame_range):
    #                     if frame not in frame_timing_dict:
    #                         continue
    #                     timing = frame_timing_dict[frame]
    #                     if start_time is None or timing < start_time:
    #                         start_time = timing
    #                     if end_time is None or timing > end_time:
    #                         end_time = timing
    #                 if start_time is None or end_time is None or start_time > end_time:
    #                     print('failed to find proper timing from {}, {}'.format(sentence, local_frame_range))
    #                 text_clip = editor.TextClip(
    #                     prefix + sentence + ' ', fontsize=fontsize, color=color, bg_color='black',
    #                     font='Lato-Medium')
    #                 text_clip = text_clip.set_start(start_time, False).set_end(end_time)
    #                 text_clip = text_clip.set_position(pos)
    #                 clips.append(text_clip)
    #             return clips
    #
    #         clips = _generate_text_clips(sentence_index_list, ' sentence: ', 27, (30, 360)) + \
    #                 _generate_text_clips(subtask_index_list, ' sub-task : ', 27, (30, 393), 'yellow')
    #         return clips
    #
    #     text_clips = generate_text_clips()
    #     logger.info('generated text clips {}'.format(len(text_clips)))
    #
    #     frame_list = sorted(frame_timing_dict.keys())
    #     frame_list = list(filter(lambda x: image_path_func(x).exists(), frame_list))
    #     raw_timing_list = [frame_timing_dict[f] for f in frame_list]
    #     t1, t2 = raw_timing_list[0], raw_timing_list[-1]
    #
    #     audio_timing_list = sorted(self.audio_dict.keys())
    #     audio_timing_list = list(filter(lambda x: t1 < x < t2, audio_timing_list))
    #     for audio_timing in audio_timing_list:
    #         audio_value = self.audio_dict[audio_timing]
    #         waveFile = wave.open(str(self.audio_path(audio_timing)), 'wb')
    #         waveFile.setnchannels(self.audio_setup_dict['setnchannels'])
    #         waveFile.setsampwidth(self.audio_setup_dict['setsampwidth'])
    #         waveFile.setframerate(self.audio_setup_dict['setframerate'])
    #         waveFile.writeframes(audio_value)
    #         waveFile.close()
    #     audio_file_list = [self.audio_path(t) for t in audio_timing_list]
    #     logger.info('generated audio files {}, {}'.format(audio_timing_list[0], audio_timing_list[-1]))
    #
    #     timing_offset = frame_timing_dict[frame_list[0]]
    #     image_timing_list = [(frame_timing_dict[f] - timing_offset) / 1e3 for f in frame_list]
    #     audio_timing_list = [(t - timing_offset) / 1e3 for t in audio_timing_list]
    #     for key in frame_timing_dict.keys():
    #         frame_timing_dict[key] = (frame_timing_dict[key] - timing_offset) / 1e3
    #
    #     image_duration_list = [t2 - t1 for t1, t2 in zip(image_timing_list[:-1], image_timing_list[1:])]
    #     image_duration_list.append(image_duration_list[-1])
    #     image_clip_list = []
    #     for frame, image_timing, image_duration in zip(frame_list, image_timing_list, image_duration_list):
    #         clip = editor.ImageClip(str(image_path_func(frame)))
    #         clip = clip.set_duration(image_duration)
    #         clip = clip.set_start(image_timing, True)
    #         image_clip_list.append(clip)
    #     logger.info('generated image clips {}'.format(len(image_clip_list)))
    #
    #     audio_clip_list = []
    #     for audio_file, audio_timing in zip(audio_file_list, audio_timing_list):
    #         try:
    #             clip = editor.AudioFileClip(str(audio_file))
    #             clip = clip.set_start(audio_timing, True)
    #             audio_clip_list.append(clip)
    #         except:
    #             print('failed in {}'.format(audio_file))
    #             continue
    #
    #     self.video_queue.put({
    #         'text': text_clips,
    #         'audio': audio_clip_list,
    #         'video': image_clip_list,
    #         'path': out_vid_path
    #     })
    #
    #     # cvc = editor.CompositeVideoClip(image_clip_list + text_clips)
    #     # cvc = cvc.set_fps(30)
    #     #
    #     # cac = editor.CompositeAudioClip(audio_clip_list)
    #     # logger.info('generated audio clips {}'.format(len(audio_clip_list)))
    #     #
    #     # cvc = cvc.set_audio(cac)
    #     # cvc.write_videofile(str(out_vid_path))
    #
    #     self.audio_dict = dict()

    def run_single_trajectory(self, t: int, transform: carla.Transform) -> Dict[str, bool]:
        status = {
            'exited': False,  # has to finish the entire loop
            'finished': False,  # this procedure has been finished successfully
            'saved': False,  # successfully saved the evaluation data
            'collided': False,  # the agent has collided
            'restart': False,  # this has to be restarted
            'stopped': True  # low-level controller returns stop
        }
        self.agent.reset()
        self.agent.move_vehicle(transform)
        self.control_evaluator.initialize()
        self.stop_evaluator.initialize()
        self.high_evaluator.initialize()
        self.high_data_dict[t] = []
        self.final_images = []
        self.sentence = get_random_sentence_from_keyword(self.eval_keyword)
        logger.info('moved the vehicle to the position {}'.format(t))

        count = 0
        frame = None
        clock = pygame.time.Clock()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)

        stop_buffer = []

        while not status['exited'] or not status['collided']:
            keyboard_input = listen_keyboard()
            if keyboard_input == 'q':
                status['exited'] = True
                self.event.set()
                logger.info('event was triggered')
                break

            if not self.language_queue.empty():
                self.sentence = self.language_queue.get()
                if self.sentence is None:
                    status['finished'] = True
                    break
                self.control_evaluator.initialize()
                self.stop_evaluator.initialize()
                self.high_evaluator.initialize()

            if frame is not None and self.agent.collision_sensor.has_collided(frame):
                logger.info('collision was detected at frame #{}'.format(frame))
                status['collided'] = True
                break

            clock.tick()
            self.world.tick()
            try:
                ts = self.world.wait_for_tick()
            except RuntimeError as e:
                logger.error('runtime error: {}'.format(e))
                status['restart'] = True
                return status

            if frame is not None:
                if ts.frame_count != frame + 1:
                    logger.info('frame skip!')
            frame = ts.frame_count

            if self.agent.image_frame is None:
                continue
            if self.agent.segment_frame is None:
                continue

            # run high-level evaluator when stopped was triggered by the low-level controller
            final_image = self.final_image
            if status['stopped']:
                action = self.high_evaluator.run_step(final_image, self.sentence)
                action = self.softmax(action)
                action_index = torch.argmax(action[-1], dim=0).item()
                location = self.agent.fetch_car_state().transform.location
                self.high_data_dict[t].append((final_image, {
                    'sentence': self.sentence,
                    'location': (location.x, location.y),
                    'action_index': action_index}))
                if action_index < 4:
                    self.control_evaluator.cmd = action_index
                    self.stop_evaluator.cmd = action_index
                    stop_buffer = []
                else:
                    logger.info('the task was finished by "finish"')
                    status['finished'] = True
                    break

            # run low-level evaluator to apply control and update stopped status
            if count % EVAL_FRAMERATE_SCALE == 0:
                control: carla.VehicleControl = self.control_evaluator.run_step(final_image)
                stop: float = self.stop_evaluator.run_step(final_image)
                sub_goal = fetch_high_level_command_from_index(self.control_evaluator.cmd).lower()
                logger.info('throttle {:+6.4f}, steer {:+6.4f}, delayed {}, current {:d}, stop {:+6.4f}'.
                            format(control.throttle, control.steer, frame - self.agent.image_frame_number, action_index,
                                   stop))
                self.agent.step_from_control(frame, control)
                self.agent.save_stop(frame, stop, sub_goal)
                self.agent.save_cmd(frame, self.sentence)
                stop_buffer.append(stop)
                recent_buffer = stop_buffer[-3:]
                status['stopped'] = len(recent_buffer) > 2 and sum(list(map(lambda x: x > 0.0, recent_buffer))) > 1

            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock, extra_str=self.sentence)

            self.final_images.append(final_image)

            count += 1
        logger.info('saving information')
        curr_eval_data = self.agent.export_eval_data(status['collided'], self.sentence)
        if curr_eval_data is not None:
            status['saved'] = self.export_evaluation_data(t, curr_eval_data)
        return status

    def run(self) -> bool:
        assert self.evaluation
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        if self.control_evaluator is None or self.stop_evaluator is None:
            raise ValueError('evaluation call function was not set')

        old_indices = self.traj_indices_from_state_dir()
        exited = False
        while len(old_indices) < len(self.eval_transforms) and not exited and not self.event.is_set():
            try:
                t = 0
                while t < len(self.eval_transforms):
                    if t in old_indices:
                        t += 1
                        continue
                    transform = self.eval_transforms[t]
                    run_status = self.run_single_trajectory(t, transform)
                    if run_status['exited']:
                        exited = True
                        break
                    if run_status['finished']:
                        break
                    if run_status['restart']:
                        continue
                    if run_status['saved']:
                        old_indices.add(t)
                    t += 1
            finally:
                old_indices = self.traj_indices_from_state_dir()
        set_world_asynchronous(self.world)
        if self.agent is not None:
            self.agent.destroy()
        self.event.set()
        return not exited


def main():
    argparser = argparse.ArgumentParser(description='Evaluation of trained models')
    argparser.add_argument('exp_name', type=str)
    args = argparser.parse_args()
    exp_name = args.exp_name

    conf_dir = Path.cwd() / '.carla/settings/experiments'
    conf_path = conf_dir / '{}.json'.format(exp_name)
    if not conf_path.exists():
        raise FileNotFoundError('configuration file does not exist {}'.format(conf_path))

    with open(str(conf_path), 'r') as file:
        data = json.load(file)

    def prepare_model(info_dict: dict):
        index, name, step = info_dict['index'], info_dict['name'], info_dict['step']
        rel_checkpoint_dir = '.carla/checkpoints/exp{}/{}'.format(index, name)
        rel_param_dir = '.carla/params/exp{}'.format(index)
        checkpoint_pth_name = 'step{:06d}.pth'.format(step)
        checkpoint_json_name = 'step{:06d}.json'.format(step)
        param_name = '{}.json'.format(name)
        model_dir = Path.cwd() / rel_checkpoint_dir
        param_dir = Path.cwd() / rel_param_dir
        if not model_dir.exists():
            mkdir_if_not_exists(model_dir)
        if not param_dir.exists():
            mkdir_if_not_exists(param_dir)
        checkpoint_model_path = Path.cwd() / '{}/{}'.format(rel_checkpoint_dir, checkpoint_pth_name)
        checkpoint_json_path = Path.cwd() / '{}/{}'.format(rel_checkpoint_dir, checkpoint_json_name)
        param_path = Path.cwd() / '{}/{}'.format(rel_param_dir, param_name)

        error_messages = []
        if not checkpoint_model_path.exists() or not checkpoint_json_path.exists() or not param_path.exists():
            servers = ['dgx:/raid/rohjunha', 'grta:/home/rohjunha']
            from subprocess import run
            for server in servers:
                try:
                    run(['scp', '{}/{}/{}'.format(server, rel_checkpoint_dir, checkpoint_pth_name),
                         checkpoint_model_path])
                    run(['scp', '{}/{}/{}'.format(server, rel_checkpoint_dir, checkpoint_json_name),
                         checkpoint_json_path])
                    run(['scp', '{}/{}/{}'.format(server, rel_param_dir, param_name), param_path])
                except:
                    error_messages.append('file not found in {}'.format(server))
                finally:
                    pass

        if not checkpoint_model_path.exists() or not checkpoint_json_path.exists() or not param_path.exists():
            logger.error(error_messages)
            raise FileNotFoundError('failed to fetch files from other servers')

    model_keys = ['control', 'stop', 'high', 'single']
    for key in model_keys:
        if key in data:
            prepare_model(data[key])

    args = ExperimentArgument(exp_name, data)
    param, evaluator = load_param_and_evaluator(eval_keyword='left,right', args=args, model_type='control')
    dir = EvaluationDirectory(param.exp_index, args.eval_name, evaluator.step, 'online')

    language_queue = Queue()
    video_queue = Queue()
    event = Event()
    manager = Manager()
    audio_dict = manager.dict()
    audio_setup_dict = manager.dict()

    processes = [
        Process(target=launch_recognizer, args=(language_queue, event, dir.audio_path, audio_dict, audio_setup_dict)),
        # Process(target=generate_video_from_clips, args=(video_queue, event))
    ]
    for p in processes:
        p.start()
    for keyword in args.eval_keywords:
        env = SpeechEvaluationEnvironment(keyword, language_queue, video_queue, event, audio_dict, audio_setup_dict, args)
        if not env.run():
            event.set()
            break
    language_queue.put(None)
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
