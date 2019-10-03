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
from copy import deepcopy
from functools import partial
from multiprocessing import Queue, Process, Event
from operator import attrgetter, itemgetter
from pathlib import Path
from subprocess import run
from time import sleep
from typing import Dict, Tuple, List
from PIL import Image, ImageDraw, ImageFont

import cv2
import pyaudio
from moviepy import editor
from six.moves import queue

from data.types import DriveDataFrame, LengthComputer
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
from util.road_option import fetch_high_level_command_from_index, HIGH_LEVEL_COMMAND_NAMES
from util.image import video_from_files, video_from_memory

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
SAMPLE_WIDTH = 2
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
from google.cloud import speech_v1p1beta1 as speech
from multiprocessing import Manager


TARGET_FPS = 15


def save_audio(audio_path, audio_data):
    waveFile = wave.open(str(audio_path), 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(SAMPLE_WIDTH)
    waveFile.setframerate(SAMPLE_RATE)
    waveFile.writeframes(audio_data)
    waveFile.close()


class AudioManager:
    def __init__(self, directory: EvaluationDirectory):
        self.directory = directory

    def load_audio_info(self) -> Dict[int, int]:
        if self.directory.audio_info_path.exists():
            with open(str(self.directory.audio_info_path), 'r') as file:
                info = json.load(file)
        else:
            info = dict()
        info = {int(k): int(v) for k, v in info.items()}
        return info

    def save_audio_info(self, info: Dict[int, int]):
        with open(str(self.directory.audio_info_path), 'w') as file:
            json.dump(info, file, indent=4)

    def save_audio_with_timing(self, audio_data, timing: int, traj_index: int):
        info = self.load_audio_info()
        info[traj_index] = timing
        save_audio(self.directory.audio_path(timing), audio_data)
        self.save_audio_info(info)

    def save_audio(self, audio_path, audio_data):
        save_audio(audio_path, audio_data)

    def load_audio_with_timing(self, traj_index: int):
        info = self.load_audio_info()
        if traj_index not in info:
            raise IndexError('invalid trajectory index {}'.format(traj_index))
        timing = info[traj_index]
        audio_path = self.directory.audio_path(timing)
        if not audio_path.exists():
            raise FileNotFoundError('could not find audio file {}'.format(audio_path))
        audio_read = wave.open(str(audio_path), 'rb')
        audio_data = audio_read.readframes(audio_read.getnframes())
        return audio_data, timing


def generate_video_with_audio(directory: EvaluationDirectory, traj_index: int):
    root_dir = directory.root_dir
    timing_path = root_dir / 'audios/timing{:02d}.json'.format(traj_index)
    state_path = root_dir / 'states/traj{:02d}.json'.format(traj_index)
    tmp_video_path = root_dir / 'tmp{:02d}.mp4'.format(traj_index)
    out_audio_path = root_dir / 'audio{:02d}.wav'.format(traj_index)
    out_video_path = root_dir / 'video{:02d}.mp4'.format(traj_index)
    out_sub_path = root_dir / 'video{:02d}.vtt'.format(traj_index)
    audio_manager = AudioManager(directory)
    export_subtitle = False

    if out_video_path.exists():
        return True

    if not state_path.exists():
        return False

    def image_path_func(frame: int):
        return root_dir / 'images/{:08d}e.png'.format(frame)

    # read audio
    audio_data, audio_start_ts = audio_manager.load_audio_with_timing(traj_index)

    # read state
    with open(str(state_path), 'r') as file:
        state_dict = json.load(file)
    frame_range = state_dict['frame_range']
    sentence_dict = {i: s for i, s in zip(range(*frame_range), state_dict['sentences'])}
    subtask_dict = {i: s for i, s in zip(range(*frame_range), map(itemgetter(1), state_dict['stop_frames']))}

    # read timing dict and prune with images
    def get_frame_from_image_path(image_path):
        return int(image_path.stem[:-1])

    frame_from_images = [get_frame_from_image_path(p) for p in sorted((root_dir / 'images').glob('*e.png'))]
    with open(str(timing_path), 'r') as file:
        raw_timing_dict = json.load(file)
    timing_dict = dict()
    for k, v in raw_timing_dict.items():
        if int(k) in frame_from_images:
            timing_dict[int(k)] = v
    frame_by_timing = {v: k for k, v in timing_dict.items()}
    sorted_frames = sorted(timing_dict.keys())
    sorted_timings = sorted(frame_by_timing.keys())
    interpolate = partial(
        interpolate_frame_by_timing, frame_by_timing=frame_by_timing, sorted_timings=sorted_timings)

    # compute the common starting timestamp
    image_start_ts = sorted_timings[0]
    state_start_ts = interpolate_timing_by_frame(frame_range[0], timing_dict, sorted_frames)
    start_ts = max(audio_start_ts, image_start_ts, state_start_ts)
    end_ts = interpolate_timing_by_frame(frame_range[1], timing_dict, sorted_frames)
    logger.info('set the interval {}, {}'.format(start_ts, end_ts))

    # cut the audio
    diff_audio_ts = start_ts - audio_start_ts
    diff_audio_len = int(round(diff_audio_ts / 1e3 * SAMPLE_WIDTH * SAMPLE_RATE))
    audio_data = audio_data[diff_audio_len:]

    duration_audio = len(audio_data) / (SAMPLE_WIDTH * SAMPLE_RATE)
    duration_image = (sorted_timings[-1] - start_ts) / 1e3
    duration_state = (end_ts - start_ts) / 1e3
    duration = min(duration_audio, duration_image, duration_state)
    num_frames = int(round(duration * TARGET_FPS))
    timestamps = [int(round(start_ts + i * 1e3 / TARGET_FPS)) for i in range(num_frames)]
    relative_timings = [(ts - start_ts) * TARGET_FPS / 1e3 for ts in timestamps]
    image_frames = [interpolate(ts) for ts in timestamps]

    image_path_list = [image_path_func(f) for f in image_frames]
    for p, f in zip(image_path_list, image_frames):
        if not p.exists():
            print(p, f)
    assert all(p.exists() for p in image_path_list)
    image_path_set = set(image_path_list)
    image_dict = dict()
    last_value = None
    for p in image_path_set:
        key = get_frame_from_image_path(p)
        try:
            value = Image.open(p)
            if value is not None:
                last_value = value
        except:
            if last_value is not None:
                image_dict[key] = last_value
            continue
        image_dict[key] = value
    images = [image_dict[f] for f in image_frames]
    sentences = [sentence_dict[f] for f in image_frames]
    subtasks = [subtask_dict[f] for f in image_frames]
    logger.info('read {} images'.format(len(image_dict.keys())))

    if export_subtitle:
        texts = ['sentence: {}\nsubtask: {}'.format(s1, s2) for s1, s2 in zip(sentences, subtasks)]
        indices = unique_with_islices(texts)

        def format_timing(timing: float):
            sec = timing % 60
            rem = int(round(timing - sec) / 60)
            min = rem % 60
            hr = int(round(rem / 60))
            print(timing, hr, min, sec)
            return '{:02d}:{:02d}:{:06.3f}'.format(hr, min, sec)

        def format_interval(t1: float, t2: float):
            return '{} --> {}'.format(format_timing(t1), format_timing(t2))

        lines = ['WEBVTT']
        for text, (i1, i2) in indices:
            f1 = image_frames[i1]
            f2 = image_frames[i2] if i2 < len(image_frames) else image_frames[i2-1]
            t1 = (interpolate_timing_by_frame(f1, timing_dict, sorted_frames) - start_ts) / 1e3
            t2 = (interpolate_timing_by_frame(f2, timing_dict, sorted_frames) - start_ts) / 1e3
            # t1 = relative_timings[i1]
            # t2 = relative_timings[i2-1]
            lines.append('\n'.join([format_interval(t1, t2), text]))
        line = '\n\n'.join(lines)
        with open(str(out_sub_path), 'w') as file:
            file.write(line)
        return True
    else:
        font_path = str(Path.cwd() / 'Roboto-Regular.ttf')
        font = ImageFont.truetype(font_path, 30)

        def draw(image, font, text, pos, color):
            draw = ImageDraw.Draw(image)
            text_size = font.getsize(text)
            xmargin, ymargin = 7, 5
            pos1 = (pos[0] - xmargin, pos[1] - ymargin)
            pos2 = (pos[0] + text_size[0] + xmargin, pos[1] + text_size[1] + ymargin)
            fill = 'rgb({}, {}, {})'.format(color[2], color[1], color[0])
            draw.rectangle([pos1, pos2], fill='rgb(0, 0, 0)')
            draw.text(pos, text, fill=fill, font=font)

        labels = ['sentence', 'sub-task']
        positions = [(50, 350), (50, 390)]
        colors = [(255, 255, 255), (0, 255, 255)]
        for i, (image, sentence, subtask) in enumerate(zip(images, sentences, subtasks)):
            size = np.array(image.size) * 2
            image = image.resize(size.astype(int), Image.ANTIALIAS)
            texts = ['{}: {}'.format(l, s) for l, s in zip(labels, [sentence, subtask])]
            sizes = [font.getsize(s) for s in texts]
            for text, size, pos, color in zip(texts, sizes, positions, colors):
                draw(image, font, text, pos, color)

            size = np.array(image.size) / 2
            image = image.resize(size.astype(int), Image.ANTIALIAS)
            images[i] = image
        logger.info('write texts on images')
    images = [np.array(i) for i in images]
    video_from_memory(images, out_video_path, framerate=TARGET_FPS, revert=False)
    logger.info('save temporary video')

    # cut out the audio again
    audio_frames = int(round(duration * SAMPLE_WIDTH * SAMPLE_RATE))
    audio_data = audio_data[:audio_frames]
    save_audio(out_audio_path, audio_data)
    logger.info('save audio')

    # merge the audio to the video
    cmd = ['ffmpeg', '-y', '-i', str(out_video_path), '-c:v', 'libx264', str(tmp_video_path)]
    run(cmd)

    video_clip = editor.VideoFileClip(str(tmp_video_path))
    audio_clip = editor.AudioFileClip(str(out_audio_path))
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(out_video_path), fps=30)
    logger.info('write the final video file {}'.format(out_video_path))
    if out_video_path.exists() and tmp_video_path.exists():
        tmp_video_path.unlink()
    return True


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size, audio_path_func, audio_queue, event, traj_index):
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
        self.audio_queue = audio_queue
        self.traj_index = traj_index

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

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self.audio_queue.put({
            'timestamp': get_current_time(),
            'traj_index': self.traj_index,
            'data': in_data,
        })
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed and not self.event.is_set():
            data = []
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
            # self._save_audio(timestamp, final_data)
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
            sys.stdout.write(str(corrected_time) + ': ' + transcript.strip() + '\n')
            queue.put(transcript.strip())

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


def launch_recognizer(language_queue, event, audio_path_func, audio_queue, audio_setup_dict, traj_index):
    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE, audio_path_func, audio_queue, event, traj_index)
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
            listen_print_loop(responses, stream, language_queue)

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


class SpeechEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, language_queue, event, audio_queue, audio_setup_dict, traj_index, args):
        self.eval_keyword = eval_keyword
        args.show_game = True
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')

        self.event = event
        self.language_queue = language_queue
        self.traj_index = traj_index

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
        self.audio_queue = audio_queue
        self.audio_setup_dict = audio_setup_dict
        self.last_sub_task = None
        self.sentence = None

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

        # while not self.event.is_set():
        #     if not self.language_queue.empty():
        #         self.sentence = self.language_queue.get()
        #     if self.sentence is not None:
        #         break

        self.sentence = get_random_sentence_from_keyword(self.eval_keyword)
        self.last_sub_task = None
        logger.info('moved the vehicle to the position {}'.format(t))

        count = 0
        frame = None
        clock = pygame.time.Clock()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)

        agent_len = LengthComputer()
        stop_buffer = []
        while not status['exited'] or not status['collided']:
            keyboard_input = listen_keyboard()
            updated = False
            if keyboard_input == 'q':
                status['exited'] = True
                self.event.set()
                logger.info('event was triggered')
                break
            elif keyboard_input == 'r':
                status['restart'] = True
                logger.info('restarted by the user')
                return status
            elif keyboard_input == 's':
                status['finished'] = True
                logger.info('finished by the user')
                break

            if not self.language_queue.empty():
                updated = True
                self.sentence = self.language_queue.get()
                logger.info('sentence was updated: {}'.format(self.sentence))
                if self.sentence is None:
                    status['finished'] = True
                    break
                keyword = 'left'
                self.eval_keyword = keyword
                self.control_param.eval_keyword = keyword
                self.stop_param.eval_keyword = keyword
                self.high_param.eval_keyword = keyword
                self.control_evaluator.param = self.control_param
                self.stop_evaluator.param = self.stop_param
                self.high_evaluator.cmd = keyword
                self.high_evaluator.param = self.high_param
                self.high_evaluator.sentence = keyword.lower()
                self.control_evaluator.initialize()
                self.stop_evaluator.initialize()
                self.high_evaluator.initialize()

            if frame is not None and self.agent.collision_sensor.has_collided(frame):
                logger.info('collision was detected at frame #{}'.format(frame))
                status['collided'] = True
                break

            if count > 50 and agent_len.length < 0.5:
                logger.info('simulation has a problem in going forward')
                status['restart'] = True
                return status

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
            if status['stopped'] or self.control_evaluator.cmd == 3 and updated:
                sentence = deepcopy(self.sentence)
                action = self.high_evaluator.run_step(final_image, sentence)
                action = self.softmax(action)
                action_index = torch.argmax(action[-1], dim=0).item()
                location = self.agent.fetch_car_state().transform.location
                self.high_data_dict[t].append((final_image, {
                    'sentence': sentence,
                    'location': (location.x, location.y),
                    'action_index': action_index}))
                sub_task = HIGH_LEVEL_COMMAND_NAMES[action_index]
                # if self.last_sub_task != sub_task:
                self.last_sub_task = sub_task
                logger.info('sentence: {}, sub-task: {}'.format(sentence, self.last_sub_task))
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
                logger.info('{} {:+6.4f}'.format(sub_goal, stop))
                # logger.info('throttle {:+6.4f}, steer {:+6.4f}, delayed {}, current {:d}, stop {:+6.4f}'.
                #             format(control.throttle, control.steer, frame - self.agent.image_frame_number, action_index,
                #                    stop))
                self.agent.step_from_control(frame, control)
                self.agent.save_stop(frame, stop, sub_goal)
                self.agent.save_cmd(frame, self.sentence)
                agent_len(self.agent.data_frame_dict[self.agent.data_frame_number].state.transform.location)
                stop_buffer.append(stop)
                recent_buffer = stop_buffer[-3:]
                status['stopped'] = len(recent_buffer) > 2 and sum(list(map(lambda x: x > 0.0, recent_buffer))) > 1

            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock, extra_str=self.sentence)

            self.final_images.append(final_image)

            count += 1

        self.audio_queue.put({
            'timestamp': get_current_time(),
            'data': None,
            'traj_index': t,
        })
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
                    self.traj_index = t
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


def interpolate_timing_by_frame(query_frame: int, timing_dict: Dict[int, int], sorted_frames: List[int]) -> int:
    if query_frame in timing_dict:
        return timing_dict[query_frame]

    n1_index, n1_frame, n1_dist = -1, -1, 1e10
    n2_index, n2_frame, n2_dist = -1, -1, 1e10
    for i, frame in enumerate(sorted_frames):
        dist = abs(query_frame - frame)
        if query_frame <= frame:
            if dist < n1_dist:
                n1_dist = dist
                n1_index = i
                n1_frame = frame
        else:
            if dist < n2_dist:
                n2_dist = dist
                n2_index = i
                n2_frame = frame

    if n1_frame >= 0 and n2_frame >= 0:
        f1, f2 = min(n1_frame, n2_frame), max(n1_frame, n2_frame)
        r1, r2 = query_frame - f1, f2 - query_frame
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
        return int(round(r1 * timing_dict[f2] + r2 * timing_dict[f1]))
    elif n1_frame < 0 and n2_frame >= 0:
        if n2_index - 1 < 0:
            raise IndexError('negative n2_index value')
        f1, f2 = sorted_frames[n2_index - 1], n2_frame
        dv = (timing_dict[f2] - timing_dict[f1]) / (f2 - f1)
        dd = query_frame - f2
        return int(round(timing_dict[f2] + dd * dv))
    elif n2_frame < 0 and n1_frame >= 0:
        if n1_index + 1 >= len(sorted_frames):
            raise IndexError('too large n1_index value {}, {}, {}'.format(query_frame, n1_index, len(sorted_frames)))
        f1, f2 = n1_frame, sorted_frames[n1_index + 1]
        dv = (timing_dict[f2] - timing_dict[f1]) / (f2 - f1)
        dd = f1 - query_frame
        return int(round(timing_dict[f1] - dd * dv))
    else:
        raise ValueError('could not find any neighboring frames')


def interpolate_frame_by_timing(query_timing: int, frame_by_timing: Dict[int, int], sorted_timings: List[int]) -> int:
    if query_timing in frame_by_timing:
        return frame_by_timing[query_timing]

    index1, timing1, dist1 = -1, -1, 1e20
    index2, timing2, dist2 = -1, -1, 1e20
    for i, timing in enumerate(sorted_timings):
        dist = abs(query_timing - timing)
        if query_timing <= timing:
            if dist < dist1:
                dist1 = dist
                index1 = i
                timing1 = timing
        else:
            if dist < dist2:
                dist2 = dist
                index2 = i
                timing2 = timing

    if timing1 >= 0 and timing2 >= 0:
        t1, t2 = min(timing1, timing2), max(timing1, timing2)
        r1, r2 = query_timing - t1, t2 - query_timing
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
        return int(round(r1 * frame_by_timing[t2] + r2 * frame_by_timing[t1]))
    elif timing1 < 0 and timing2 >= 0:
        return frame_by_timing[timing2]
        # if index2 - 1 < 0:
        #     raise IndexError('negative n2_index value')
        # t1, t2 = sorted_timings[index2 - 1], timing2
        # dv = (frame_by_timing[t2] - frame_by_timing[t1]) / (t2 - t1)
        # dd = query_timing - t2
        # return int(round(frame_by_timing[t2] + dd * dv))
    elif timing2 < 0 and timing1 >= 0:
        return frame_by_timing[timing1]
        # if index1 + 1 >= len(sorted_timings):
        #     raise IndexError('too large n1_index value {}, {}, {}'.format(query_timing, index1, len(sorted_timings)))
        # t1, t2 = timing1, sorted_timings[index1 + 1]
        # dv = (frame_by_timing[t2] - frame_by_timing[t1]) / (t2 - t1)
        # dd = t1 - query_timing
        # return int(round(frame_by_timing[t1] - dd * dv))
    else:
        raise ValueError('could not find any neighboring frames')


def audio_saver(directory: EvaluationDirectory, audio_queue: Queue, video_queue: Queue, event: Event):
    audio_start_ts = -1
    audio_list = []
    audio_manager = AudioManager(directory)
    while not event.is_set():
        if not audio_queue.empty():
            data_dict = audio_queue.get()
            timestamp = data_dict['timestamp']
            audio_data = data_dict['data']
            traj_index = data_dict['traj_index']
            if audio_data is None:
                final_audio_data = b''.join(audio_list)
                audio_manager.save_audio_with_timing(final_audio_data, audio_start_ts, traj_index)
                video_queue.put(traj_index)
                audio_start_ts = -1
                audio_list = []
            else:
                audio_list.append(audio_data)
            if audio_start_ts < 0:
                audio_start_ts = timestamp
    event.set()
    video_queue.put(None)


def video_saver(directory: EvaluationDirectory, video_queue: Queue, event: Event):
    max_trial = 3
    while not event.is_set():
        if not video_queue.empty():
            traj_index = video_queue.get()
            if traj_index is None:
                event.set()
                break
            generated = False
            count = 0
            while not generated and count < max_trial:
                generated = generate_video_with_audio(directory, traj_index)
                count += 1
    event.set()


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

    model_keys = ['control', 'stop', 'high', 'single']
    for key in model_keys:
        if key in data:
            prepare_model(data[key])

    args = ExperimentArgument(exp_name, data)
    param, evaluator = load_param_and_evaluator(eval_keyword='left,right', args=args, model_type='control')
    directory = EvaluationDirectory(param.exp_index, args.eval_name, evaluator.step, 'online')
    # if directory.root_dir.exists():
    #     [p.unlink() for p in directory.audio_dir.glob('*')]
    #     [p.unlink() for p in directory.state_dir.glob('*')]
    #     [p.unlink() for p in directory.image_dir.glob('*')]
    #     [p.unlink() for p in directory.segment_dir.glob('*')]

    language_queue = Queue()
    audio_queue = Queue()
    video_queue = Queue()
    event = Event()
    manager = Manager()
    traj_index = manager.Value('i', 0)
    audio_setup_dict = manager.dict()
    audio_manager = AudioManager(directory)

    processes = [
        Process(target=launch_recognizer,
                args=(language_queue, event, directory.audio_path, audio_queue, audio_setup_dict, traj_index, )),
        Process(target=audio_saver, args=(directory, audio_queue, video_queue, event,)),
        Process(target=video_saver, args=(directory, video_queue, event, ),)
    ]
    for p in processes:
        p.start()
    for keyword in args.eval_keywords:
        env = SpeechEvaluationEnvironment(
            keyword, language_queue, event, audio_queue, audio_setup_dict, traj_index, args)
        if not env.run():
            event.set()
            break
    language_queue.put(None)
    for p in processes:
        p.join()
    # except:
    #     logger.info('exception raised')
    # finally:
    #     logger.info('finished the evaluation')

    info = audio_manager.load_audio_info()
    for traj_index in info.keys():
        generate_video_with_audio(directory, traj_index)


if __name__ == '__main__':
    directory = EvaluationDirectory(40, 'ls-town2', 72500, 'online')
    audio_manager = AudioManager(directory)
    info = audio_manager.load_audio_info()
    for traj_index in info.keys():
        generate_video_with_audio(directory, traj_index)

    # main()
