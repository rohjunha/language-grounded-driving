import argparse
import json
import shutil
from math import sqrt
from operator import attrgetter
from pathlib import Path
from time import sleep
from typing import List, Dict

import numpy as np
import pygame

from data.types import DriveDataFrame
from game.environment import GameEnvironment, set_world_asynchronous, set_world_synchronous
from evaluator import listen_keyboard, ExperimentArgument
from speech_evaluator import generate_video_with_audio
from util.common import get_logger, add_carla_module
from util.directory import EvaluationDirectory
from util.serialize import list_from_vector

add_carla_module()
logger = get_logger(__name__)
import carla


class ReplayEnvironment(GameEnvironment):
    def __init__(self, transform_dict: Dict[int, List[carla.Transform]], frame_range_dict, directory: EvaluationDirectory, args):
        args.show_game = True
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')
        self.transform_dict = transform_dict
        self.frame_range_dict = frame_range_dict
        self.image_type = 'bgr'
        self.directory = directory

    def transform_path(self, traj_index: int) -> Path:
        return self.directory.transform_path(traj_index)

    @property
    def segment_image(self):
        return np.reshape(((self.agent.segment_frame[:, :, 2] == 7).astype(dtype=np.uint8) * 255), (88, 200, 1))

    @property
    def final_image(self):
        if self.image_type == 's':
            return self.segment_image
        elif self.image_type == 'd':
            return self.segment_image
        elif self.image_type == 'bgr':
            return self.agent.image_frame
        elif self.image_type == 'bgrs':
            return np.concatenate((self.agent.image_frame, self.segment_image), axis=-1)
        elif self.image_type == 'bgrd':
            return np.concatenate((self.agent.image_frame, self.custom_segment_image), axis=-1)
        else:
            raise TypeError('invalid image type {}'.format(self.image_type))

    def copy_color_images(self, traj_index, frame_index_list: List[int]):
        frame_offset = self.frame_range_dict[traj_index][0]
        src_image_files = [(i, self.agent.image_path(f, 'extra')) for i, f in enumerate(frame_index_list)]
        src_image_files = list(filter(lambda x: x[1].exists(), src_image_files))
        target_index_list, src_image_files = zip(*src_image_files)
        dst_image_files = [self.directory.replay_image_dir / '{:08d}e.png'.format(i + frame_offset) for i in target_index_list]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]
        logger.info('copied {} images out of {}'.format(len(dst_image_files), len(frame_index_list)))

    def copy_segment_images(self, traj_index, frame_index_list: List[int]):
        frame_offset = self.frame_range_dict[traj_index][0]
        src_image_files = [(i, self.agent.segment_image_path(f)) for i, f in enumerate(frame_index_list)]
        src_image_files = list(filter(lambda x: x[1].exists(), src_image_files))
        target_index_list, src_image_files = zip(*src_image_files)
        dst_image_files = [self.directory.replay_segment_dir / '{:08d}c.png'.format(i + frame_offset) for i in target_index_list]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]

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

    def export_transform_dict(self, t: int):
        target_sensor = None
        if 'extra' in self.agent.camera_sensor_dict:
            target_sensor = self.agent.camera_sensor_dict['extra']
        elif 'center' in self.agent.camera_sensor_dict:
            target_sensor = self.agent.camera_sensor_dict['center']
        if target_sensor is None:
            return dict()
        target_dict = dict()
        for k, v in target_sensor.transform_dict.items():
            loc = list_from_vector(v.location)
            rot = [v.rotation.pitch, v.rotation.yaw, v.rotation.roll]
            target_dict[k] = ','.join([str(v) for v in loc + rot])

        with open(str(self.transform_path(t)), 'w') as file:
            json.dump(target_dict, file, indent=4)
        return target_sensor.transform_dict

    def run_single_trajectory(self, traj_index: int, transform_list: List[carla.Transform]) -> Dict[str, bool]:
        status = {
            'exited': False,  # has to finish the entire loop
            'finished': False,  # this procedure has been finished successfully
            'saved': False,  # successfully saved the evaluation data
            'collided': False,  # the agent has collided
            'restart': False,  # this has to be restarted
            'stopped': True  # low-level controller returns stop
        }
        self.agent.reset()
        self.agent.move_vehicle(transform_list[0])
        logger.info('moved the vehicle to the position {}'.format(traj_index))

        frame = None
        clock = pygame.time.Clock()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)
        frame_index_list = []

        def transform_dist(t1, t2):
            l1, l2 = list_from_vector(t1.location), list_from_vector(t2.location)
            return sqrt(sum((v2 - v1) ** 2 for v1, v2 in zip(l1, l2)))

        index = 0
        while index < len(transform_list):
            keyboard_input = listen_keyboard()
            if keyboard_input == 'q':
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

            if self.agent.vehicle is not None:
                curr_transform = self.agent.vehicle.get_transform()
                target_transform = transform_list[index]
                dist = transform_dist(curr_transform, target_transform)
                if dist < 5e-1:
                    frame_index_list.append(frame)
                    if index < len(transform_list) - 1:
                        self.agent.move_vehicle(transform_list[index + 1])
                    index += 1
            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock)

            # if self.agent.agent is None:
            #     continue
            # transform = self.agent.agent.get_transform()
            # velocity = self.agent.agent.get_velocity()
            # linear_velocity = self.agent.agent.get_angular_velocity()
            # acceleration = self.agent.agent.get_acceleration()
            # state = CarState(transform, velocity, linear_velocity, acceleration)
            # frame_dict[frame] = state.to_str()

        logger.info('saving information')

        print(frame_index_list)
        print(len(frame_index_list), len(transform_list))

        self.copy_color_images(traj_index, frame_index_list)
        self.copy_segment_images(traj_index, frame_index_list)
        transform_dict = self.export_transform_dict(traj_index)

        # frame_map = []
        # for frame, t1 in transform_dict.items():
        #     dists = sorted([(i, transform_dist(t1, t2)) for i, t2 in enumerate(transform_list)], key=lambda x: x[1])
        #     print(frame, dists[0][0])

        return status

    def run(self) -> bool:
        assert self.evaluation
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')

        traj_indices = self.directory.traj_indices_from_state_dir()
        exited = False
        for traj_index in traj_indices:
            video_path = self.directory.video_path(traj_index, 'extra')
            logger.info(video_path)
            logger.info(video_path.exists())
            if video_path.exists():
                continue
            if traj_index not in self.transform_dict:
                raise IndexError('trajectory index {} was not in the dictionary'.format(traj_index))
            transforms = self.transform_dict[traj_index]
            run_status = self.run_single_trajectory(traj_index, transforms)
            generate_video_with_audio(self.directory, traj_index, True)

            if run_status['exited']:
                exited = True
                break
            if run_status['finished']:
                break
            if run_status['restart']:
                continue
        set_world_asynchronous(self.world)
        if self.agent is not None:
            self.agent.destroy()
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
    args = ExperimentArgument(exp_name, data)

    directory = EvaluationDirectory(40, 'ls-town2', 72500, 'online')
    transform_dict = dict()
    frame_range_dict = dict()
    traj_index_set = directory.traj_indices_from_state_dir()
    for traj_index in traj_index_set:
        with open(str(directory.state_path(traj_index)), 'r') as file:
            state_dict = json.load(file)
        frame_range = state_dict['frame_range']
        data_frame_list = state_dict['data_frames']
        transforms = [DriveDataFrame.load_from_str(d).state.transform for d in data_frame_list]
        transform_dict[traj_index] = transforms
        frame_range_dict[traj_index] = frame_range
        logger.info('detected {}'.format(traj_index))

    env = ReplayEnvironment(transform_dict, frame_range_dict, directory, args)
    env.run()


if __name__ == '__main__':
    main()
