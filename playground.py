import re
import sys
import wave
from itertools import chain
from typing import Tuple

import numpy as np
from pathlib import Path

import pyaudio

from speech_evaluator import ResumableMicrophoneStream

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
STREAMING_LIMIT = 10000
SAMPLE_RATE = 16000
UNIT_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
from google.cloud import speech_v1p1beta1 as speech
from multiprocessing import Manager, Queue, Event, Process
from util.common import get_current_time, get_logger

logger = get_logger(__name__)


def save_wav(wav_data, wav_path):
    waveFile = wave.open(str(wav_path), 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(16000)
    waveFile.writeframes(wav_data)
    waveFile.close()


# class Voice:
#     framerate = SAMPLE_RATE
#
#     def __init__(self, timestamp, raw_data):
#         super().__init__()
#         self.timestamp = timestamp
#         self.raw_data = raw_data
#
#     def __len__(self):
#         return self.raw_data.shape[0]
#
#     def __add__(self, other):
#         if self.timestamp is None or self.raw_data is None:
#             self.timestamp = other.timestamp
#             self.raw_data = other.raw_data
#         else:
#             self.timestamp = min(self.timestamp, other.timestamp)
#             self.raw_data = np.vstack([self.raw_data, other.raw_data])
#
#     @property
#     def duration(self):
#         return Voice.duration_from_length(len(self))
#
#     def save(self, filepath: Path, timestamp=False):
#         if timestamp:
#             dst_path = filepath / '{}.wav'.format(self.timestamp)
#         else:
#             dst_path = filepath
#         logger.info('save voice file: {}'.format(dst_path))
#         sf.write(str(dst_path), self.raw_data, self.framerate)
#
#     @property
#     def range(self):
#         return self.timestamp, self.timestamp + round(self.duration * 1e6)
#
#     def trim(self, timestamp_range):
#         """
#         cut the audio w.r.t the timestamp range
#         :param timestamp_range:
#         :return:
#         """
#         gts1, gts2 = timestamp_range
#         lts1, lts2 = self.timestamp, self.timestamp + Voice.timestamp_diffs_from_length(len(self))
#         if lts1 < gts1:
#             tsdiff1 = gts1 - lts1
#             fcount = Voice.length_from_timestamp_diffs(tsdiff1)
#             if fcount > 0:
#                 self.raw_data = self.raw_data[fcount:, :]
#                 self.timestamp += Voice.timestamp_diffs_from_length(fcount)
#         if lts2 > gts2:
#             tsdiff2 = lts2 - gts2
#             fcount = Voice.length_from_timestamp_diffs(tsdiff2)
#             if fcount > 0:
#                 self.raw_data = self.raw_data[:-fcount, :]
#
#     def cut(self, durations: Tuple[float, float]):
#         frames = [Voice.length_from_duration(v) for v in durations]
#         return Voice(0, self.raw_data[frames[0]:frames[1], :])
#
#         # gts1, gts2 = timestamp_range
#         # lts1, lts2 = self.timestamp, self.timestamp + Voice.timestamp_diffs_from_length(len(self))
#         # if lts1 < gts1:
#         #     tsdiff1 = gts1 - lts1
#         #     fcount = Voice.length_from_timestamp_diffs(tsdiff1)
#         #     if fcount > 0:
#         #         new_audio.raw_data = self.raw_data[fcount:, :]
#         # if lts2 > gts2:
#         #     tsdiff2 = lts2 - gts2
#         #     fcount = Voice.length_from_timestamp_diffs(tsdiff2)
#         #     if fcount > 0:
#         #         new_audio.raw_data = self.raw_data[:-fcount, :]
#         # return new_audio
#
#     @staticmethod
#     def load(filepath: Path, timestamp):
#         if not filepath.exists():
#             raise FileNotFoundError('could not find a file: {}'.format(filepath))
#         data, _ = sf.read(str(filepath), dtype='int16')
#         return Voice(timestamp, data)
#
#     @staticmethod
#     def load_dir(filepath: Path):
#         if not filepath.exists():
#             raise FileNotFoundError('could not find a directory: {}'.format(filepath))
#         filepaths = sorted(list(filepath.glob('*.wav')))
#         return Voice.collapse([Voice.load(f, int(f.stem)) for f in filepaths])
#
#     @staticmethod
#     def collapse(raw_voices: list):
#         raw_voices = sorted(raw_voices, key=lambda x: x.timestamp)
#         timestamps = [v.timestamp for v in raw_voices]
#         tds1 = [v.duration for v in raw_voices[:-1]]
#         tds2 = [ts2 - ts1 for ts1, ts2 in zip(timestamps[:-1], timestamps[1:])]
#         inds = [[i] * round(tds[1] / 1e6 / tds[0]) for i, tds in enumerate(zip(tds1, tds2))] + [[len(raw_voices) - 1]]
#         voices = [raw_voices[i] for i in chain.from_iterable(inds)]
#         timestamp = min([v.timestamp for v in voices])
#         raw_data = np.vstack([v.raw_data for v in voices])
#         return Voice(timestamp, raw_data)
#
#     @staticmethod
#     def timestamp_diffs_from_length(length):
#         return Voice.timestamp_diffs_from_duration(Voice.duration_from_length(length))
#
#     @staticmethod
#     def timestamp_diffs_from_duration(duration):
#         return round(duration * 1e6)
#
#     @staticmethod
#     def duration_from_timestamp_diffs(timestamp_diffs):
#         return timestamp_diffs / 1e6
#
#     @staticmethod
#     def length_from_timestamp_diffs(timestamp_diffs):
#         duration = Voice.duration_from_timestamp_diffs(timestamp_diffs)
#         return Voice.length_from_duration(duration)
#
#     @staticmethod
#     def length_from_duration(duration):
#         return round(duration / 1000 * Voice.framerate)
#
#     @staticmethod
#     def duration_from_length(length):
#         return length / Voice.framerate * 1000


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


def launch_recognizer(queue, event, audio_path_func, audio_dict, audio_setup_dict):
    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE, audio_path_func, audio_dict, event)
    audio_setup_dict['setnchannels'] = mic_manager._num_channels
    audio_setup_dict['setsampwidth'] = mic_manager._audio_interface.get_sample_size(pyaudio.paInt16)
    audio_setup_dict['setframerate'] = mic_manager._rate
    logger.info('microphone stream')

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    logger.info('speech client')

    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:
        logger.info('start stream')
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


def save_audio_file_with_timing_info(audio_dict: dict, traj_index: int):
    timestamps = sorted(audio_dict.keys())
    t1, t2 = timestamps[0], timestamps[-1]
    values = [audio_dict[k] for k in timestamps]
    data = b''.join(values)
    save_wav(data, Path.home() / '.tmp/audio/{}.wav'.format(t1))

    # # diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    # # diffs.append(0.1)
    #
    # unit_length = len(values[0])
    # unit_count = int(round(unit_length / CHUNK_SIZE))
    # print(unit_length, unit_count)
    #
    # def frame_from_timestamp(time, unit_length):
    #     return int(round(time / 1e3 / UNIT_DURATION * unit_length))
    #
    # total_length = frame_from_timestamp(t2 - t1, unit_length) + unit_length
    # # data = list(b'\0' * total_length)  # 0.04987576772443264
    # # for i, (v, t) in enumerate(zip(values, timestamps)):
    # #     index = frame_from_timestamp(t - t1, unit_length)
    # #     print(i, t, index)
    # #     data[index:index+unit_length] = v
    # # data = bytearray(data)
    #
    # print((t2 - t1) / 1e3 + 0.1)
    # print(len(data) / 2 / 16000)
    # # print(total_length)
    # # print(len(v) * len(values))


def main():
    event = Event()
    queue = Queue()
    manager = Manager()
    audio_dict = manager.dict()
    audio_setup_dict = manager.dict()

    def audio_path_func(timestamp):
        root_dir = Path.home() / '.tmp/audio'
        return root_dir / '{}.wav'.format(timestamp)

    # processes = [
    #     Process(target=launch_recognizer, args=(queue, event, audio_path_func, audio_dict, audio_setup_dict)),
    #     # Process(target=generate_video_from_clips, args=(video_queue, event))
    # ]
    # for p in processes:
    #     p.start()
    # sleep(10)
    launch_recognizer(queue, event, audio_path_func, audio_dict, audio_setup_dict)
    # for p in processes:
    #     p.join()
    save_audio_file_with_timing_info(audio_dict, 0)


if __name__ == '__main__':
    main()
