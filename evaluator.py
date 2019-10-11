#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
import json
import random
import shutil
from collections import defaultdict
from functools import partial
from math import sqrt
from operator import attrgetter, itemgetter
from pathlib import Path
from time import sleep
from typing import Tuple, List, Dict, Any

import cv2

from custom_carla.agents.navigation.local_planner import RoadOption
from data.types import DriveDataFrame, LengthComputer
from environment import set_world_asynchronous, set_world_synchronous, FrameCounter, should_quit, GameEnvironment
from util.common import add_carla_module, get_logger
from util.directory import EvaluationDirectory, mkdir_if_not_exists

add_carla_module()
logger = get_logger(__name__)
import carla

import numpy as np
import pygame
import torch

from config import IMAGE_WIDTH, IMAGE_HEIGHT, EVAL_FRAMERATE_SCALE, DATASET_FRAMERATE
from data.dataset import load_index_from_word, generate_templated_sentence_dict, HighLevelDataset
from util.road_option import fetch_road_option_from_str, fetch_onehot_vector_dim, fetch_onehot_vector_from_index, \
    fetch_num_sentence_commands, fetch_onehot_vector_from_sentence_command, fetch_onehot_index_from_high_level_str, \
    fetch_high_level_command_from_index
from util.image import tensor_from_numpy_image, video_from_files
from model import init_hidden_states
from parameter import Parameter
from trainer import CheckpointBase


def canonicalize(src: str):
    return str(src).replace('\\', '/').replace('//', '/')


def onehot_from_index(cmd: int, use_low_level_segment: bool) -> torch.Tensor:
    onehot_dim = fetch_onehot_vector_dim(use_low_level_segment)
    return fetch_onehot_vector_from_index(cmd, use_low_level_segment).view(1, onehot_dim)


def _tensor_from_numpy_image(image: np.ndarray) -> torch.Tensor:
    c = image.shape[2]
    return tensor_from_numpy_image(image, False).view(1, 1, c, IMAGE_HEIGHT, IMAGE_WIDTH)


class LowLevelEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: int):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.use_low_level_segment = param.use_low_level_segment
        self.onehot_dim = fetch_onehot_vector_dim(param.use_low_level_segment)
        self.onehot_func = partial(onehot_from_index, use_low_level_segment=param.use_low_level_segment)
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray) -> Any:
        batch = self._prepare_batch(image)
        model_output = self._run_step(batch)
        self.step_elapsed += 1
        return model_output

    @property
    def onehot_vector(self):
        return fetch_onehot_vector_from_index(self.cmd, self.use_low_level_segment).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, custom_action_index: int = -1):
        # self.initialize()
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-5:]
        data_dict = {
            'onehot': self.onehot_vector,
            'action_index': [self.cmd if custom_action_index < 0 else custom_action_index],
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        model_output = self.model.forward(data, self.encoder_hidden, self.decoder_hidden)
        output = model_output['output'][0][-1]
        if output.size(-1) == 2:
            control = carla.VehicleControl()
            control.throttle = output[0].item()
            control.steer = output[1].item()
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control
        else:
            return output.item()


class HighLevelEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: str):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.onehot_dim = fetch_num_sentence_commands()
        self.onehot_func = fetch_onehot_vector_from_sentence_command
        self.index_func = fetch_onehot_index_from_high_level_str
        self.sentence = param.eval_keyword.lower()
        self.index_from_word = load_index_from_word()
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray, sentence: str) -> torch.Tensor:
        batch = self._prepare_batch(image, sentence)
        action = self._run_step(batch)
        return action

    @property
    def onehot_vector(self):
        return self.onehot_func(self.cmd).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, sentence: str):
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        length = torch.tensor([len(word_indices)], dtype=torch.long)
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-5:]
        data_dict = {
            'sentence': sentence.strip(),
            'word_indices': word_indices,
            'length': length,
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        model_output = self.model(data, self.encoder_hidden, self.decoder_hidden)
        agent_action = model_output['output'][0]
        self.step_elapsed += 1
        return agent_action


class SingleEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: int):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.use_low_level_segment = param.use_low_level_segment
        self.index_from_word = load_index_from_word()
        self.onehot_dim = fetch_onehot_vector_dim(param.use_low_level_segment)
        self.onehot_func = partial(onehot_from_index, use_low_level_segment=param.use_low_level_segment)
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray, sentence: str) -> Any:
        batch = self._prepare_batch(image, sentence)
        output = self._run_step(batch)
        self.step_elapsed += 1
        return output

    @property
    def onehot_vector(self):
        return fetch_onehot_vector_from_index(self.cmd, self.use_low_level_segment).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, sentence: str, custom_action_index: int = -1):
        # self.initialize()
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        length = torch.tensor([len(word_indices)], dtype=torch.long)
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-10:]
        data_dict = {
            'sentence': sentence,
            'word_indices': word_indices,
            'length': length,
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        model_output = self.model.forward(data, self.encoder_hidden, self.decoder_hidden)
        output = model_output['output'][0][-1]
        if output.size(-1) == 2:
            control = carla.VehicleControl()
            control.throttle = output[0].item()
            control.steer = output[1].item()
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control
        else:
            return output.item()


class EvaluationMetaInfo:
    def __init__(self, timestamp: int, road_option: RoadOption, frame_range: Tuple[int, int]):
        self.timestamp = timestamp
        self.road_option = road_option
        self.frame_range = frame_range


def str_from_road_option(road_option: RoadOption) -> str:
    return road_option.name.lower()


def road_option_from_str(road_option_str: str) -> RoadOption:
    return fetch_road_option_from_str(road_option_str)


def _prepare_evaluation_param(param: Parameter) -> Parameter:
    assert param.eval_data_name
    assert param.eval_info_name
    assert param.eval_keyword
    if param.model_level == 'low':
        param.eval_keyword = fetch_road_option_from_str(param.eval_keyword.upper())
    elif param.model_level == 'high':
        param.eval_keyword = param.eval_keyword.lower()
    else:
        logger.info(param.model_level)
        raise TypeError('invalid eval_keyword was given {}'.format(param.eval_keyword))
    param.max_data_length = -1
    param.shuffle = False
    param.batch_size = 1
    param.dataset_data_names = [param.eval_data_name]
    param.dataset_info_names = [param.eval_info_name]
    if param.model_level == 'low':
        param.use_multi_cam = False
        param.use_sequence = False
        param.has_clusters = False
    return param


def fetch_unique_data_from_high_level_dataset(dataset: HighLevelDataset, eval_keyword: str) -> \
        List[Tuple[List[DriveDataFrame], str]]:
    keywords = dataset.get_extended_keywords()
    indices = list(map(itemgetter(0), filter(lambda x: x[1].lower() == eval_keyword, enumerate(keywords))))

    def position_from_drive_data_frame(drive_frame: DriveDataFrame):
        location = drive_frame.state.transform.location
        return location.x, location.y

    positions = []
    for i in indices:
        drive_list = dataset.get_mid_drive_data(i)
        xs, ys = zip(*list(map(position_from_drive_data_frame, drive_list)))
        positions.append((list(xs), list(ys)))

    def compute_list_dist(l1, l2) -> float:
        return sqrt(sum([(v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2 for v1, v2 in zip(l1, l2)]))

    dist_threshold = 10.0
    edge_dict = defaultdict(list)
    for i, v1 in enumerate(positions):
        ivl = sorted([(j, compute_list_dist(v1, positions[j])) for j in range(i + 1, len(positions))],
                     key=itemgetter(1))
        edge_dict[i] = list(map(itemgetter(0), filter(lambda x: x[1] < dist_threshold, ivl)))

    visited = [False for _ in range(len(positions))]
    unique_indices = []
    for i in range(len(positions)):
        if visited[i]:
            continue
        visited[i] = True
        for n in edge_dict[i]:
            visited[n] = True
        unique_indices.append(i)

    indices = [indices[i] for i in unique_indices]

    data_list = []
    for i in indices:
        keyword, sentence, data_frame = dataset.get_trajectory_data_from_sequence_index(i)
        data_list.append((data_frame.drives, sentence))
    return data_list


def load_evaluation_dataset(param: Parameter) -> Tuple[List[List[DriveDataFrame]], List[str]]:
    param = _prepare_evaluation_param(param)
    data_root = Path.cwd() / '.carla/dataset/evaluation'
    if int(param.dataset_data_names[0][-1]) == 1:
        data_root = data_root / 'town1'
    else:
        data_root = data_root / 'town2'
    if not data_root.exists():
        raise FileNotFoundError('could not find {}'.format(data_root))

    data_path = data_root / '{}.json'.format(param.eval_keyword)
    with open(str(data_path), 'r') as file:
        eval_dict = json.load(file)

    drives = [[DriveDataFrame.load_from_str(d) for d in dl] for dl in eval_dict['drives']]
    sentences = eval_dict['sentences']
    return list(drives), list(sentences)


def load_param_and_evaluator(eval_keyword: str, args, model_type: str):
    param = Parameter()
    low_level = model_type in ['control', 'stop']
    if model_type == 'control':
        exp_index = args.control_model_index
        exp_name = args.control_model_name
        exp_step = args.control_model_step
    elif model_type == 'stop':
        exp_index = args.stop_model_index
        exp_name = args.stop_model_name
        exp_step = args.stop_model_step
    elif model_type == 'high':
        exp_index = args.high_level_index
        exp_name = args.high_level_name
        exp_step = args.high_level_step
    elif model_type == 'single':
        exp_index = args.single_model_index
        exp_name = args.single_model_name
        exp_step = args.single_model_step
    else:
        raise TypeError('invalid model type {}'.format(model_type))
    param.exp_name = exp_name
    param.exp_index = exp_index
    param.load()
    param.batch_size = 1
    param.eval_keyword = eval_keyword
    param.eval_data_name = args.eval_data_name
    param.eval_info_name = args.eval_info_name

    logger.info('model type: {}'.format(param.model_type))
    cls = LowLevelEvaluator if low_level else (SingleEvaluator if model_type == 'single' else HighLevelEvaluator)
    logger.info((model_type, cls, param.model_level, param.encoder_type))
    eval_arg = args.exp_cmd if low_level else eval_keyword
    evaluator = cls(param, eval_arg)
    evaluator.load(step=exp_step)
    return param, evaluator


class EvaluationEnvironmentBase(GameEnvironment, EvaluationDirectory):
    def __init__(self, args, model_type: str):
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')
        self.eval_param, self.evaluator = load_param_and_evaluator(args=args, model_type=model_type)
        self.eval_transforms = self.world.get_map().get_spawn_points()
        EvaluationDirectory.__init__(self, *self.eval_info)

    @property
    def eval_info(self):
        return self.eval_param.exp_index, self.eval_param.exp_name, self.evaluator.step, 'online'


class OfflineEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, args):
        self.eval_keyword = eval_keyword
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')
        self.eval_name = args.eval_name

        # load params and evaluators
        self.control_param, self.control_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='control')
        self.stop_param, self.stop_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='stop')
        self.high_level_param, self.high_level_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='high')

        # set image type
        self.image_type = self.high_level_param.image_type
        if 'd' in self.image_type:
            from model import DeepLabModel, prepare_deeplab_model
            self.deeplab_model: DeepLabModel = prepare_deeplab_model()

        self.final_images = []
        self.eval_dataset, self.eval_sentences = load_evaluation_dataset(self.high_level_param)
        self.eval_transforms = list(map(lambda x: x[0].state.transform, self.eval_dataset))
        self.high_level_sentences = self.eval_sentences
        logger.info('fetched {} sentences from {}'.format(
            len(self.high_level_sentences), self.high_level_param.eval_keyword.lower()))
        self.softmax = torch.nn.Softmax(dim=1)
        EvaluationDirectory.__init__(self, *self.eval_info)
        self.high_level_data_dict = dict()

    @property
    def eval_info(self):
        return self.control_param.exp_index, self.eval_name, \
               self.control_evaluator.step, self.eval_keyword

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

    def export_evaluation_data(self, t: int, curr_eval_data: dict) -> bool:
        with open(str(self.state_path(t)), 'w') as file:
            json.dump(curr_eval_data, file, indent=2)

        data_frames = [DriveDataFrame.load_from_str(s) for s in curr_eval_data['data_frames']]
        controls = list(map(attrgetter('control'), data_frames))
        stops, sub_goals = zip(*curr_eval_data['stop_frames'])
        texts = ['th{:+4.2f} st{:+4.2f} {:4s}:{:+4.2f}'.format(c.throttle, c.steer, g[:4], s)
                 for c, s, g in zip(controls, stops, sub_goals)]
        text_dict = {i: t for i, t in zip(range(*curr_eval_data['frame_range']), texts)}
        src_image_files = [self.agent.image_path(f) for f in range(*curr_eval_data['frame_range'])]
        src_image_files = list(filter(lambda x: x.exists(), src_image_files))
        if self.image_type in ['s', 'd']:
            final_image_files = [self.segment_dir / '{:08d}.png'.format(i) for i in range(len(self.final_images))]
            for p, s in zip(final_image_files, self.final_images):
                cv2.imwrite(str(p), s)
            video_from_files(final_image_files, self.video_dir / 'segment{:02d}.mp4'.format(t),
                             texts=[], framerate=EVAL_FRAMERATE_SCALE * DATASET_FRAMERATE, revert=False)
        image_frames = set([int(s.stem[:-1]) for s in src_image_files])
        drive_frames = set(text_dict.keys())
        common_frames = sorted(list(image_frames.intersection(drive_frames)))
        src_image_files = [self.agent.image_path(f) for f in common_frames]
        dst_image_files = [self.image_dir / p.name for p in src_image_files]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]
        text_list = [text_dict[f] for f in common_frames]
        video_from_files(src_image_files, self.video_path(t),
                         texts=text_list, framerate=EVAL_FRAMERATE_SCALE * DATASET_FRAMERATE, revert=True)
        return self.state_path(t).exists()

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
        self.high_level_evaluator.initialize()
        self.high_level_data_dict[t] = []
        self.final_images = []
        sentence = random.choice(self.high_level_sentences)
        logger.info('moved the vehicle to the position {}'.format(t))

        count = 0
        frame = None
        clock = pygame.time.Clock() if self.show_image else FrameCounter()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)

        agent_len, expert_len = LengthComputer(), LengthComputer()
        for l in self.eval_dataset[t]:
            expert_len(l.state.transform.location)
        criterion_len = 2.5 * expert_len.length  # 0.9 * expert_len.length
        max_iter = 10.0 * len(self.eval_dataset[t])  # 5.0 * len(self.eval_dataset[t])
        stop_buffer = []

        while agent_len.length < criterion_len and count < max_iter:
            if self.show_image and should_quit():
                status['exited'] = True
                break

            if frame is not None and self.agent.collision_sensor.has_collided(frame):
                logger.info('collision was detected at frame #{}'.format(frame))
                status['collided'] = True
                break

            if count > 30 and agent_len.length < 1:
                logger.info('simulation has a problem in going forward')
                status['exited'] = True
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
                logger.info((final_image.shape))
                action = self.high_level_evaluator.run_step(final_image, sentence)
                action = self.softmax(action)
                logger.info((action, action.shape, sentence))
                action_index = torch.argmax(action[-1], dim=0).item()
                logger.info('action {}, action_index {}'.format(action, action_index))
                location = self.agent.fetch_car_state().transform.location
                self.high_level_data_dict[t].append((final_image, {
                    'sentence': sentence,
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
                logger.info('throttle {:+6.4f}, steer {:+6.4f}, delayed {}, stop {:+6.4f}'.format(
                    control.throttle, control.steer, frame - self.agent.image_frame_number, stop))
                self.agent.step_from_control(frame, control)
                self.agent.save_stop(frame, stop, sub_goal)
                agent_len(self.agent.data_frame_dict[self.agent.data_frame_number].state.transform.location)
                stop_buffer.append(stop)
                recent_buffer = stop_buffer[-3:]
                status['stopped'] = len(recent_buffer) > 2 and sum(list(map(lambda x: x > 0.0, recent_buffer))) > 1

            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock)

            self.final_images.append(final_image)

            count += 1

            if agent_len.length >= criterion_len:
                logger.info('trajectory length is longer than the threshold')
            if count >= max_iter:
                logger.info('reached the maximum number of iterations')

        if not status['finished']:
            status['finished'] = status['collided'] or agent_len.length >= criterion_len or count >= max_iter
        if not status['finished']:
            return status
        curr_eval_data = self.agent.export_eval_data(status['collided'], sentence)
        if curr_eval_data is not None:
            status['saved'] = self.export_evaluation_data(t, curr_eval_data)
        return status

    def save_high_level_data(self):
        tmp_dir = mkdir_if_not_exists(Path.home() / '.tmp/high-level')
        for key in self.high_level_data_dict.keys():
            if len(self.high_level_data_dict[key]) != 4:
                continue
            data_dir = mkdir_if_not_exists(tmp_dir / '{:03d}'.format(key))
            dict_list = []
            for i, (image, item_dict) in enumerate(self.high_level_data_dict[key]):
                cv2.imwrite(str(data_dir / '{:03d}.png'.format(i)), image)
                dict_list.append(item_dict)
            with open(str(data_dir / 'data.json'), 'w') as file:
                json.dump(dict_list, file, indent=2)

    def run(self) -> bool:
        assert self.evaluation
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        if self.control_evaluator is None or self.stop_evaluator is None:
            raise ValueError('evluation call function was not set')

        old_indices = self.traj_indices_from_state_dir()
        exited = False
        while len(old_indices) < len(self.eval_transforms) and not exited:
            try:
                t = 0
                while t < len(self.eval_transforms):
                    if t in old_indices:
                        t += 1
                        continue
                    transform = self.eval_transforms[t]
                    run_status = self.run_single_trajectory(t, transform)
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
        self.save_high_level_data()
        return True


def listen_keyboard() -> str:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 'q'
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return 'q'
            elif event.key == pygame.K_l:
                return 'l'
            elif event.key == pygame.K_r:
                return 'r'
            elif event.key == pygame.K_s:
                return 's'
            elif event.key == pygame.K_f:
                return 'f'
            elif event.key == pygame.K_u:
                return 'u'
            elif event.key == pygame.K_i:
                return 'i'
            elif event.key == pygame.K_o:
                return 'o'
            elif event.key == pygame.K_j:
                return 'j'
            elif event.key == pygame.K_k:
                return 'k'
            elif event.key == pygame.K_m:
                return 'm'
            elif event.key == pygame.K_COMMA:
                return ','
            elif event.key == pygame.K_PERIOD:
                return '.'
            elif event.key == pygame.K_1:
                return '1'
            elif event.key == pygame.K_2:
                return '2'
            elif event.key == pygame.K_3:
                return '3'
            elif event.key == pygame.K_4:
                return '4'
            elif event.key == pygame.K_5:
                return '5'
    return ''


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


def get_random_sentence_from_keyword(keyword: str) -> str:
    def replace_word(word: str):
        if word.startswith('extrastraight'):
            return 'straight'
        elif word == 'extraleft':
            return 'left'
        elif word == 'extraright':
            return 'right'
        else:
            return word

    words = list(map(replace_word, keyword.split(',')))
    keyword = ','.join(words)
    if keyword not in __sentence_library_dict__:
        raise KeyError('invalid keyword was given {}'.format(keyword))
    sentence_group = __sentence_library_dict__[keyword]
    sentences = random.choice(sentence_group)
    sentence = random.choice(sentences)
    return sentence


class OnlineEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, args):
        self.eval_keyword = eval_keyword
        args.show_game = True
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')

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

        self.export_video(t, 'center', curr_eval_data)
        self.export_video(t, 'extra', curr_eval_data)
        self.export_segment_video(t)

        return self.state_path(t).exists()

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
                break
            elif keyboard_input in __keyword_from_input__.keys():
                keyword = __keyword_from_input__[keyboard_input]
                if keyword != self.eval_keyword:
                    self.eval_keyword = keyword
                    self.sentence = get_random_sentence_from_keyword(self.eval_keyword)
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
                    logger.info('updated sentence {}'.format(self.sentence))

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
        while len(old_indices) < len(self.eval_transforms) and not exited:
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
        return not exited


def fetch_ip_address():
    from subprocess import run
    from re import findall
    from subprocess import PIPE
    raw_lines = run(['ifconfig'], stdout=PIPE).stdout.decode()
    candidates = findall('inet addr:([\d]+.[\d]+.[\d]+.[\d]+)', raw_lines)

    def filter_out(cand: str):
        if cand == '192.168.0.1':
            return False
        if cand.startswith('127') or cand.startswith('172'):
            return False
        return True

    candidates = list(filter(filter_out, candidates))
    if candidates:
        return candidates[0]
    else:
        return '172.0.0.1'


class ExperimentArgument:
    def __init__(self, eval_name: str, info_dict: dict):
        exp_keys = ['port', 'keywords', 'data']
        for key in exp_keys:
            if key not in info_dict:
                raise KeyError('essential key was not found {}'.format(key))

        self.eval_type: str = 'offline'
        self.model_level: str = 'both'
        self.verbose: bool = False
        self.host: str = fetch_ip_address()
        self.port: int = info_dict['port']
        self.res: str = '200x88'
        self.width, self.height = [int(x) for x in self.res.split('x')]
        self.filter: str = 'vehicle.*'
        self.map: str = None
        self.speed: int = 20
        self.no_rendering: bool = False
        self.safe: bool = False
        self.show_game: bool = False
        self.eval_name: str = eval_name
        self.eval_keywords: List[str] = info_dict['keywords']
        self.exp_cmd: int = 0
        self.random_seed: int = 0
        self.position_index: int = 0
        self.eval_data_name: str = info_dict['data']['name']
        self.eval_info_name: str = '{}-v{}'.format(self.eval_data_name, info_dict['data']['version'])
        self.camera_keywords: List[str] = ['center']

        model_keys = ['control', 'stop', 'high', 'single']
        model_suffix = ['model', 'model', 'level', 'model']
        items = ['index', 'name', 'step']
        default_values = [None, '', None]
        for key, suffix in zip(model_keys, model_suffix):
            if key in info_dict:
                for item in items:
                    setattr(ExperimentArgument, '{}_{}_{}'.format(key, suffix, item), info_dict[key][item])
            else:
                for item, value in zip(items, default_values):
                    setattr(ExperimentArgument, '{}_{}_{}'.format(key, suffix, item), value)


def launch_experiment_from_json(exp_name: str, online: bool):
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
    cls = OnlineEvaluationEnvironment if online else OfflineEvaluationEnvironment
    for keyword in args.eval_keywords:
        eval_env = cls(eval_keyword=keyword, args=args)
        if not eval_env.run():
            break


def main():
    argparser = argparse.ArgumentParser(description='Evaluation of trained models')
    argparser.add_argument('exp_name', type=str)
    argparser.add_argument('--online', action='store_true')
    args = argparser.parse_args()
    launch_experiment_from_json(args.exp_name, args.online)


if __name__ == '__main__':
    main()
