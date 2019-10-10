# #!/usr/bin/env python
#
# # Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# # Barcelona (UAB).
# #
# # This work is licensed under the terms of the MIT license.
# # For a copy, see <https://opensource.org/licenses/MIT>.
# import argparse
# from functools import partial
# from operator import itemgetter
# from time import sleep
# from typing import List, Dict, Tuple
#
# import numpy as np
#
# from config import EVAL_FRAMERATE_SCALE, DATASET_FRAMERATE
# from data.dataset import fetch_dataset, fetch_dataset_pair
# from data.storage import DataStorage
# from data.types import DriveDataFrame, LengthComputer
# from environment import set_world_asynchronous, set_world_synchronous, should_quit, FrameCounter, GameEnvironment
import random
from time import sleep
from typing import List, Dict, Any

import cv2

from custom_carla.agents.navigation.agent import ControlWithInfo
from data.types import DriveDataFrame, CarState
from game.common import get_font, draw_image, np_from_carla_image
from game.environment import GameEnvironment, set_world_asynchronous, set_world_synchronous, should_quit, FrameCounter, \
    CarlaSyncWrapper
# from evaluator import LowLevelEvaluator
# from parameter import Parameter
from game.sensors import set_all_sensors, SensorManager
from util.common import add_carla_module, get_logger, fetch_ip_address, get_timestamp
# from util.directory import fetch_dataset_dir
# from util.road_option import fetch_index_from_road_option, fetch_name_from_road_option
#
from util.directory import ExperimentDirectory
from util.serialize import str_from_waypoint

logger = get_logger(__name__)
add_carla_module()
import carla
# import pygame
#
#
# def align_indices_from_dicts(
#         image_frame_dict: Dict[int, np.ndarray],
#         drive_frame_dict: Dict[int, DriveDataFrame],
#         search_range: int) -> Tuple[bool, List[int], List[int]]:
#     image_frame_keys = sorted(image_frame_dict.keys())
#     drive_frame_keys = sorted(drive_frame_dict.keys())
#     min_frame_key = min(image_frame_keys[0], drive_frame_keys[0])
#     max_frame_key = max(image_frame_keys[-1], drive_frame_keys[-1])
#     reference_frame_range = list(range(min_frame_key, max_frame_key + 1, EVAL_FRAMERATE_SCALE))
#     rel_indices = list(filter(lambda x: x != 0, range(-search_range, search_range + 1)))
#     image_indices, drive_indices = [], []
#     for reference_frame_key in reference_frame_range:
#         if reference_frame_key in image_frame_dict:
#             image_indices.append(reference_frame_key)
#         else:
#             found = False
#             for rel in rel_indices:
#                 if reference_frame_key + rel in image_frame_dict:
#                     image_indices.append(reference_frame_key + rel)
#                     found = True
#                     break
#             if not found:
#                 logger.error('could not find a proper neighboring image at {}'.format(reference_frame_key))
#                 return False, [], []
#         if reference_frame_key in drive_frame_dict:
#             drive_indices.append(reference_frame_key)
#         else:
#             found = False
#             for rel in rel_indices:
#                 if reference_frame_key + rel in drive_frame_dict:
#                     drive_indices.append(reference_frame_key + rel)
#                     found = True
#                     break
#             if not found:
#                 logger.error('could not find a proper neighboring drive frame at {}'.format(reference_frame_key))
#                 return False, [], []
#     assert image_indices
#     assert len(image_indices) == len(drive_indices)
#     return True, image_indices, drive_indices
#
#
# class DaggerGeneratorEnvironment(GameEnvironment):
#     def __init__(self, args, eval_param: Parameter, index_data_list: List[Tuple[int, dict]]):
#         assert eval_param.eval_data_name
#         GameEnvironment.__init__(self, args=args, agent_type='basic')
#
#         self.eval_param = eval_param
#         self.evaluator = LowLevelEvaluator(self.eval_param, 0)
#         self.evaluator.load(step=args.exp_step)
#         self.index_data_list = index_data_list
#         logger.info('dagger data indices: {}'.format(list(map(itemgetter(0), index_data_list))))
#         self.dataset_name = eval_param.eval_data_name
#         self.dataset = DataStorage(False, fetch_dataset_dir() / self.dataset_name)
#         self.dagger_segments = []
#
#     @property
#     def eval_info(self):
#         return self.eval_param.exp_index, self.eval_param.exp_name, self.evaluator.step, self.eval_param.eval_keyword
#
#     def run_single_trajectory(self, t: int, data: dict) -> Dict[str, bool]:
#         status = {
#             'exited': False,  # has to finish the entire loop
#             'finished': False,  # this procedure has been finished successfully
#             'collided': False,  # the agent has collided
#             'restart': False  # this has to be restarted
#         }
#         self.evaluator.cmd = data['action_index']
#         self.agent.reset()
#         logger.info('moved the vehicle to the position {}, set action to {}'.format(t, data['action_index']))
#
#         local_image_dict = dict()
#         local_drive_dict = dict()
#
#         count = 0
#         frame = None
#         clock = pygame.time.Clock() if self.show_image else FrameCounter()
#         # todo: implement this function as in the same one in evaluator.py
#
#         set_world_asynchronous(self.world)
#         self.agent.agent.set_destination(data['src_transform'].location, data['dst_location'])
#         self.agent.move_vehicle(data['src_transform'])
#         sleep(0.5)
#         set_world_synchronous(self.world)
#
#         len_waypoints = LengthComputer()
#         for waypoint_with_info in self.agent.agent._route:
#             len_waypoints(waypoint_with_info.waypoint.transform.location)
#         max_len = 0.9 * len_waypoints.length
#         max_iter = 5.0 * EVAL_FRAMERATE_SCALE * len(self.agent.agent._route)
#         len_agent = LengthComputer()
#         while count < max_iter and len_agent.length < max_len:
#             if self.show_image and should_quit():
#                 status['exited'] = True
#                 return status
#
#             if frame is not None and self.agent.collision_sensor.has_collided(frame):
#                 logger.info('collision was detected at frame #{}'.format(frame))
#                 status['collided'] = True
#                 break
#
#             clock.tick()
#             self.world.tick()
#             try:
#                 ts = self.world.wait_for_tick()
#             except RuntimeError as e:
#                 logger.error('runtime error: {}'.format(e))
#                 status['restart'] = True
#                 return status
#
#             if frame is not None:
#                 if ts.frame_count != frame + 1:
#                     logger.info('frame skip!')
#             frame = ts.frame_count
#
#             # image_frame_number, image = self.agent.image_queue.get()
#             if self.agent.image_frame is None:
#                 continue
#
#             # register images
#             image = self.agent.image_frame
#             image_frame_number = self.agent.image_frame_number
#             local_image_dict[image_frame_number] = image
#
#             # store action values from the expert
#             waypoint, road_option, drive_data_frame = self.agent.step_from_pilot(frame, apply=False, update=True)
#             local_drive_dict[frame] = self.agent.data_frame_dict[frame]
#             if waypoint is None:
#                 status['finished'] = True
#                 break
#
#             # apply action values from the agent
#             if count % EVAL_FRAMERATE_SCALE == 0:
#                 model_control = self.evaluator.run_step(self.agent.image_frame)
#                 control_str = 'throttle {:+6.4f}, steer {:+6.4f}, delayed {}'.format(
#                     model_control.throttle, model_control.steer, frame - image_frame_number)
#                 if image_frame_number in local_drive_dict:
#                     expert_control = local_drive_dict[image_frame_number].control
#                     control_str += ' steer {:+6.4f}, steer-diff {:+6.4f}'.format(
#                         expert_control.steer, model_control.steer - expert_control.steer)
#                 logger.info(control_str)
#                 self.agent.step_from_control(frame, model_control, apply=True, update=False)
#                 len_agent(drive_data_frame.state.transform.location)
#
#             if self.show_image:
#                 self.show(image, clock)
#
#             count += 1
#
#         aligned, image_indices, drive_indices = align_indices_from_dicts(
#             local_image_dict, local_drive_dict, EVAL_FRAMERATE_SCALE // 2)
#         if aligned:
#             road_option_name = fetch_name_from_road_option(data['road_option'])
#             self.dataset.put_data_from_dagger(
#                 t, road_option_name, local_image_dict, local_drive_dict, image_indices, drive_indices)
#             logger.info('successfully added {} dagger trajectory'.format(t))
#         else:
#             status['restart'] = True
#         return status
#
#     def run(self):
#         if self.world is None:
#             raise ValueError('world was not initialized')
#         if self.agent is None:
#             raise ValueError('agent was not initialized')
#         if self.evaluator is None:
#             raise ValueError('evluation call function was not set')
#
#         try:
#             i = 0
#             while i < len(self.index_data_list):
#                 index, data = self.index_data_list[i]
#                 if not self.dataset.has_trajectory(index):
#                     run_status = self.run_single_trajectory(index, data)
#                     if run_status['exited']:
#                         break
#                     if run_status['restart']:
#                         continue
#                 i += 1
#         finally:
#             set_world_asynchronous(self.world)
#             if self.agent is not None:
#                 self.agent.destroy()
#
#
# def load_dagger_generator(args):
#     port = args.port
#     ports = args.ports
#     assert ports
#     assert port in ports
#     assert args.dataset_name
#     assert args.eval_dataset_name
#     port_index = ports.index(port)
#
#     eval_param = Parameter()
#     eval_param.exp_index = args.exp_index
#     eval_param.exp_name = args.exp_name
#     eval_param.load()
#     eval_param.batch_size = 1
#     eval_param.dataset_data_names = [args.dataset_name]
#     eval_param.eval_data_name = args.eval_dataset_name
#     eval_param.max_data_length = -1
#
#     if eval_param.split_train:
#         train_dataset, valid_dataset = fetch_dataset_pair(eval_param)
#     else:
#         train_dataset = fetch_dataset(eval_param)
#
#     num_data = len(train_dataset)
#     index_func = partial(fetch_index_from_road_option, low_level=eval_param.use_low_level_segment)
#     data_list = []
#     for i in range(num_data):
#         road_option, images, drives = train_dataset.get_trajectory_data(i)
#         data_list.append({
#             'road_option': road_option,
#             'action_index': index_func(road_option),
#             'src_transform': drives[0].state.transform,
#             'dst_location': drives[-1].state.transform.location,
#             'length': len(images)
#         })
#
#     def chunker_list(seq, size):
#         return (seq[i::size] for i in range(size))
#
#     index_data_lists = list(chunker_list(list(enumerate(data_list)), len(ports)))
#     index_data_list = index_data_lists[port_index]
#     return DaggerGeneratorEnvironment(args, eval_param, index_data_list)
#
#
# class NoiseInjector:
#     def __init__(self, param: Parameter):
#         self.count = 0
#         self.framerate = DATASET_FRAMERATE
#         self.inject_noise = param.inject_noise
#         self.noise_type = param.noise_type
#         self.noise_std = param.noise_std
#         self.noise_interval = param.noise_interval  # in seconds
#         self.noise_duration = param.noise_duration  # in seconds
#         self.step_interval = self.noise_interval * self.framerate
#         self.step_duration = self.noise_duration * self.framerate
#         self.step_start = self.step_interval - self.step_duration
#         self.noise_max_steer = param.noise_max_steer
#
#     @property
#     def inject(self) -> float:
#         if self.inject_noise:
#             if self.noise_type == 'linear':
#                 step = self.count % self.step_interval
#                 sign = 2 * ((self.count // self.step_interval) % 2) - 1
#                 if step < self.step_start:
#                     return 0.0
#                 duration = (step - self.step_start) / self.step_duration
#                 return sign * min(duration, 1.0 - duration)
#             elif self.noise_type == 'normal':
#                 return float(np.random.normal(scale=self.noise_std, size=1)[0])
#         else:
#             return 0.0
#
#     def step(self):
#         self.count += 1


class FrameSnapshot:
    def __init__(self, timestamp, actor_snapshot: carla.ActorSnapshot, rgb_dict: Dict[str, Any], seg_dict: Dict[str, Any]):
        self.timestamp = timestamp
        if actor_snapshot is None:
            self.car_state = None
        else:
            self.car_state = CarState(
                actor_snapshot.get_transform(),
                actor_snapshot.get_velocity(),
                actor_snapshot.get_angular_velocity(),
                actor_snapshot.get_acceleration())
        self.control = None
        self.rgb_dict = rgb_dict
        self.seg_dict = seg_dict
        # self.vis_seg_dict = dict()
        # for keyword in self.seg_dict.keys():
        #     seg_dict[keyword].convert(carla.ColorConverter.CityScapesPalette)

    @property
    def valid(self):
        return self.timestamp is not None

    @property
    def frame(self):
        return self.timestamp.frame

    @property
    def sim_elapsed(self):
        return self.timestamp.elapsed_seconds

    @property
    def real_elapsed(self):
        return self.timestamp.platform_timestamp

    @property
    def data_frame(self):
        return DriveDataFrame(self.car_state, self.control)

    @property
    def waypoint(self):
        return self.control.waypoint if self.control is not None else None


class SnapshotSaver(ExperimentDirectory):
    def __init__(self, timestamp):
        ExperimentDirectory.__init__(self, timestamp)

    def save_image(self, frame, keyword, image):
        array = np_from_carla_image(image)
        cv2.imwrite(str(self.image_path(frame, keyword)), array)

    def save(self, frame_snapshot: FrameSnapshot):
        import numpy as np
        for keyword in frame_snapshot.rgb_dict.keys():
            image = np_from_carla_image(frame_snapshot.rgb_dict[keyword], False)
            cv2.imwrite(str(self.image_path(frame_snapshot.frame, keyword)), image)
        for keyword in frame_snapshot.seg_dict.keys():
            image = np_from_carla_image(frame_snapshot.seg_dict[keyword])
            cv2.imwrite(str(self.segment_image_path(frame_snapshot.frame, keyword)), image)
        with open(str(self.experiment_data_path), 'a') as file:
            file.write('{}:{}\n'.format(self.frame_str(frame_snapshot.frame), frame_snapshot.data_frame))
        if frame_snapshot.waypoint is not None:
            with open(str(self.experiment_waypoint_path), 'a') as file:
                file.write('{}\n'.format(str_from_waypoint(frame_snapshot.waypoint)))


class OfflineGeneratorEnvironment(GameEnvironment, SnapshotSaver):
    def __init__(self, args):
        GameEnvironment.__init__(self, args=args, agent_type='roaming', transform_index=0)
        SnapshotSaver.__init__(self, get_timestamp())

    def collect(self, sync_mode: CarlaSyncWrapper) -> FrameSnapshot:
        actor_snapshot = None
        try:
            snapshot, rgb_dict, seg_dict = sync_mode.tick(timeout=2.0)
            if snapshot.has_actor(self.vehicle.id):
                actor_snapshot = snapshot.find(self.vehicle.id)
        except:
            return FrameSnapshot(None, None, dict(), dict())
        return FrameSnapshot(snapshot.timestamp, actor_snapshot, rgb_dict, seg_dict)

    def run(self):
        import pygame
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        pygame.init()

        display = pygame.display.set_mode(
            (200, 88),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = get_font()
        clock = pygame.time.Clock() if self.show_image else FrameCounter()

        try:
            map = self.world.get_map()
            start_pose = random.choice(map.get_spawn_points())
            waypoint = map.get_waypoint(start_pose.location)

            frame = None
            count = 0

            set_world_asynchronous(self.world)
            sleep(0.5)
            set_world_synchronous(self.world)

            waypoint_dict = dict()
            road_option_dict = dict()

            self.sensor_manager = SensorManager(self.world, self.vehicle, ['center', 'left', 'right'], 200, 88, False)
            with CarlaSyncWrapper(self.world, self.sensor_manager) as sync_mode:
                while True:
                    if should_quit():
                        return
                    clock.tick()

                    frame_snapshot = self.collect(sync_mode)
                    if not frame_snapshot.valid:
                        continue
                    # waypoint = random.choice(waypoint.next(1.5))
                    # self.agent.move_vehicle(waypoint.transform)
                    print(frame_snapshot.timestamp)
                    fps = round(1.0 / frame_snapshot.timestamp.delta_seconds)
                    frame_snapshot.control = self.agent.step_from_pilot()
                    print(frame_snapshot.control.control)
                    print(frame_snapshot.control.waypoint)
                    print(frame_snapshot.control.road_option)
                    print(frame_snapshot.control.possible_road_options)
                    self.save(frame_snapshot)
                    # print(data['timestamp'])
                    # Draw the display.
                    # draw_image(display, rgb_dict['center'])
                    draw_image(display, frame_snapshot.rgb_dict['center'])
                    # for keyword, image in rgb_dict.items():
                    #     draw_image(display, image)
                    # for keyword, image in seg_dict.items():
                    #     draw_image(display, image, blend=True)
                    display.blit(
                        font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                        (8, 10))
                    display.blit(
                        font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 28))
                    pygame.display.flip()
        finally:
            print('destroying actors.')
            set_world_asynchronous(self.world)
            self.destroy()
            pygame.quit()
            print('done.')

            #
            # while count < 200000:
            #     if self.show_image and should_quit():
            #         break
            #
            #     if frame is not None and self.agent.collision_sensor.has_collided(frame):
            #         logger.info('collision was detected at frame #{}'.format(frame))
            #         set_world_asynchronous(self.world)
            #         self.transform_index += 1
            #         self.agent.move_vehicle(self.transforms[self.transform_index])
            #         sleep(0.5)
            #         self.agent.agent.reset_planner()
            #         set_world_synchronous(self.world)
            #
            #     clock.tick()
            #     frame = self.world.tick()
            #     print(frame)
            #     # if frame is not None:
            #     #     if ts.frame_count != frame + 1:
            #     #         logger.info('frame skip!')
            #     # frame = ts.frame_count
            #     # self.injector.step()
            #
            #     if len(self.agent.data_frame_buffer) > 100:
            #         self.agent.export_data()
            #         if not self.show_image:
            #             print(str(clock))
            #
            #     if self.agent.image_frame is None:
            #         continue
            #
            #     waypoint, road_option, _ = self.agent.step_from_pilot(
            #         frame, update=True, apply=True, inject=0.0)
            #     waypoint_dict[frame] = waypoint
            #     road_option_dict[frame] = road_option
            #     if self.show_image:
            #         image = self.agent.image_frame
            #         image_frame_number = self.agent.image_frame_number
            #         image_road_option = road_option_dict[image_frame_number] if image_frame_number in road_option_dict else None
            #         image_waypoint = waypoint_dict[image_frame_number] if image_frame_number in waypoint_dict else None
            #         self.show(image, clock, image_road_option, image_waypoint.is_intersection if image_waypoint is not None else None)
            #
            #     count += 1
            # self.agent.export_data()

        # finally:
            # if self.agent is not None:
            #     self.agent.destroy()


# def main():
#     # Parse arguments
#     argparser = argparse.ArgumentParser(description='Data generator')
#     argparser.add_argument('data_type', type=str)
#     argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
#     argparser.add_argument('--host', metavar='H', default='10.158.54.63', help='host server IP (default: 10.158.54.63')
#     argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port (default: 2000)')
#     argparser.add_argument('--ports', type=int, nargs='*', default=[])
#     argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='200x88', help='window resolution (default: 200x88)')
#     argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
#     argparser.add_argument('--map', metavar='TOWN', default=None, help='start a new episode at the given TOWN')
#     argparser.add_argument('--speed', metavar='S', default=20, type=int, help='Maximum speed in PID controller')
#     argparser.add_argument('--no-rendering', action='store_true', help='switch off server rendering')
#     argparser.add_argument('--show-triggers', action='store_true', help='show trigger boxes of traffic signs')
#     argparser.add_argument('--show-connections', action='store_true', help='show waypoint connections')
#     argparser.add_argument('--show-spawn-points', action='store_true', help='show recommended spawn points')
#     argparser.add_argument('--safe', action='store_true', help='avoid spawning vehicles prone to accidents')
#     argparser.add_argument('--show-game', action='store_true', help='show game display')
#     argparser.add_argument('--eval-timestamp', type=int)
#     argparser.add_argument('--exp-name', type=str)
#     argparser.add_argument('--exp-step', type=int, default=None)
#     argparser.add_argument('--exp-cmd', type=int, default=0)
#     argparser.add_argument('--exp-index', type=int, default=32)
#     argparser.add_argument('--random-seed', type=int, default=0)
#     argparser.add_argument('--position-index', type=int, default=0)
#     argparser.add_argument('--dataset-name', type=str)
#     argparser.add_argument('--eval-dataset-name', type=str)
#     argparser.add_argument('--camera-keywords', type=str, nargs='*', default=['left', 'center', 'right'])
#
#     args = argparser.parse_args()
#     args.description = argparser.description
#     args.width, args.height = [int(x) for x in args.res.split('x')]
#
#     if args.data_type == 'dagger':
#         gen_env = load_dagger_generator(args=args)
#     elif args.data_type == 'offline':
#         gen_env = OfflineGeneratorEnvironment(args=args)
#     elif args.data_type == 'test':
#         gen_env = TestGeneratorEnvironment(args=args)
#     else:
#         raise TypeError('invalid data type {}'.format(args.data_type))
#     gen_env.run()


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
        logger.info('host: {}'.format(self.host))
        self.port: int = info_dict['port']
        self.res: str = '200x88'
        self.width, self.height = [int(x) for x in self.res.split('x')]
        self.filter: str = 'vehicle.*'
        self.map: str = None
        self.speed: int = 20
        self.no_rendering: bool = False
        self.safe: bool = False
        self.show_game: bool = True
        self.eval_name: str = eval_name
        self.eval_keywords: List[str] = info_dict['keywords']
        self.exp_cmd: int = 0
        self.random_seed: int = 0
        self.position_index: int = 0
        self.eval_data_name: str = info_dict['data']['name']
        self.eval_info_name: str = '{}-v{}'.format(self.eval_data_name, info_dict['data']['version'])
        self.camera_keywords: List[str] = ['center']
        self.use_extra: bool = False

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


def main():
    info_dict = {
        'port': 7777,
        'keywords': ['left'],
        'data': {'name': 'control', 'version': 37}
    }
    args = ExperimentArgument('eval_test', info_dict)
    env = OfflineGeneratorEnvironment(args)
    env.run()


if __name__ == '__main__':
    main()
