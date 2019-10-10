import json
import math
import queue
import random
from functools import partial
from operator import attrgetter
from time import perf_counter
from typing import Tuple, List

import cv2
from custom_carla.agents.navigation.agent import ControlWithInfo
from custom_carla.agents.navigation.basic_agent import BasicAgent
from custom_carla.agents.navigation.local_planner import RoadOption
from custom_carla.agents.navigation.roaming_agent import RoamingAgent

from config import DATASET_FRAMERATE
from data.types import CarState, CarControl, DriveDataFrame
from game.common import destroy_actor
from game.sensors import set_all_sensors
from util.common import add_carla_module, get_logger, get_timestamp, set_random_seed
from util.directory import ExperimentDirectory
from util.serialize import str_from_waypoint

add_carla_module()
logger = get_logger(__name__)
import carla
import numpy as np
import pygame


SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


class FrameCounter:
    def __init__(self):
        self.t1 = None
        self.t2 = None
        self.counter = 0

    def tick(self):
        if self.t1 is None:
            self.t1 = perf_counter()
        self.t2 = perf_counter()
        self.counter += 1

    @property
    def framerate(self):
        if self.counter == 0 or self.t1 is None or self.t2 is None or self.t1 == self.t2:
            return 0.0
        else:
            return self.counter / (self.t2 - self.t1)

    def reset(self):
        self.t1 = None
        self.t2 = None
        self.counter = 0

    def __str__(self):
        return 'count: {:5d}, fps: {:4.2f}'.format(self.counter, self.framerate)


# def draw_image(surface, image: np.ndarray):
#     array = image[:, :, ::-1]
#     image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#     surface.blit(image_surface, (0, 0))


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def keyboard_control() -> int:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 2  # break the outer loop
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return 2
            elif event.key in [pygame.K_n, pygame.K_q]:
                return 1  # continue to next iteration
            elif event.key == pygame.K_1:
                return 11
            elif event.key == pygame.K_2:
                return 12
            elif event.key == pygame.K_3:
                return 13
            elif event.key == pygame.K_4:
                return 14
    return 0  # nothing happened


def waypoint_info(display, font, waypoint):
    """abcd"""
    def render(font, key, value):
        return font.render('{}:{}'.format(key, value), True, (255, 255, 255))

    px, py, hy = 8, 20, 10
    keys = ['lane_type', 'lane_change', 'is_intersection', 'road_id', 'section_id', 'lane_id', 's']
    values = [(key, attrgetter(key)(waypoint)) for key in keys]
    texts = []
    nexts = waypoint.next(2)
    texts.append(render(font, 'next', len(nexts)))
    for key_value_pair in values:
        key, value = key_value_pair
        if isinstance(value, carla.LaneMarking):
            value = value.type
        texts.append(render(font, key, value))

    heights = list(range(py, py + hy * (len(texts) - 1), hy))
    for text, height in zip(texts, heights):
        display.blit(text, (px, height))


def get_synchronous_mode(world: carla.World):
    return world.get_settings().synchronous_mode


def set_world_synchronous(world: carla.World):
    if world is None:
        return
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / DATASET_FRAMERATE
        world.apply_settings(settings)


def set_world_asynchronous(world: carla.World):
    if world is None:
        return
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        world.apply_settings(settings)


def set_world_rendering_option(world: carla.World, render: bool):
    if world is None:
        return
    settings = world.get_settings()
    settings.no_rendering_mode = not render
    world.apply_settings(settings)


def set_traffic_lights_green(world: carla.World):
    set_world_asynchronous(world)
    for tl in world.get_actors().filter('traffic.*'):
        if isinstance(tl, carla.TrafficLight):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)


def clean_vehicles(world):
    set_world_asynchronous(world)
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.is_alive:
            actor.destroy()


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def numpy_from_carla_image(carla_image: carla.Image) -> np.ndarray:
    np_image = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape(carla_image.height, carla_image.width, 4)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
    return rgb_image


class CarlaSyncWrapper:
    def __init__(
            self,
            world: carla.World,
            camera_sensor_dict: dict,
            segmentation_sensor_dict: dict):
        self.world = world
        self.delta_seconds = 1.0 / DATASET_FRAMERATE
        self.settings = self.world.get_settings()
        self.camera_sensor_dict = camera_sensor_dict
        self.segmentation_sensor_dict = segmentation_sensor_dict
        self.world_queue = None
        self.camera_queue_dict = dict()
        self.segmentation_queue_dict = dict()
        self.frame = None

    def __enter__(self):
        self.settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            return q

        def make_queue_dict(target_dict, keyword, register_event):
            q = queue.Queue()
            register_event(q.put)
            target_dict[keyword] = q

        self.world_queue = make_queue(self.world.on_tick)
        for keyword, sensor in self.camera_sensor_dict.items():
            make_queue_dict(self.camera_queue_dict, keyword, sensor.listen)
        for keyword, sensor in self.segmentation_sensor_dict.items():
            make_queue_dict(self.segmentation_queue_dict, keyword, sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        world_data = self.retrieve_data(self.world_queue, timeout)
        camera_data_dict = {k: self.retrieve_data(q, timeout) for k, q in self.camera_queue_dict.items()}
        segmentation_data_dict = {k: self.retrieve_data(q, timeout) for k, q in self.segmentation_queue_dict.items()}
        assert all(x.frame == self.frame for x in camera_data_dict.values())
        assert all(x.frame == self.frame for x in segmentation_data_dict.values())
        return world_data, camera_data_dict, segmentation_data_dict

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self.settings)

    def retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class VehicleWrapper:
    def __init__(
            self,
            world,
            transform: carla.Transform,
            agent_type: str = 'roaming',
            speed: float = 20.0,
            safe: bool = True,
            autopilot: bool = True):
        self.world = world
        self.agent_type = agent_type
        self.safe = safe
        self.speed = speed
        self.autopilot = autopilot
        self.vehicle = None
        self.agent = None
        self._set_vehicle(transform)
        if self.autopilot:
            self._set_agent()

    def _set_agent(self):
        if self.vehicle is None:
            raise ValueError('vehicle is not assigned')
        if self.agent_type == 'roaming':
            self.agent = RoamingAgent(self.vehicle, self.speed)
        elif self.agent_type == 'basic':
            self.agent = BasicAgent(self.vehicle, self.speed)
        else:
            raise TypeError('invalid agent type: {}'.format(self.agent_type))

    def _set_vehicle(self, transform):
        blueprints = self.world.get_blueprint_library().filter('vehicle.audi.a2')
        if self.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        if self.vehicle is not None:
            return

        blueprint_vehicle = random.choice(blueprints)
        blueprint_vehicle.set_attribute('role_name', 'hero')
        if blueprint_vehicle.has_attribute('color'):
            color = random.choice(blueprint_vehicle.get_attribute('color').recommended_values)
            blueprint_vehicle.set_attribute('color', color)
        blueprint_vehicle.set_attribute('role_name', 'autopilot')

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(blueprint_vehicle, transform)
        self.vehicle.set_autopilot(False)

    def move_vehicle(self, transform: carla.Transform):
        transform.location.z += 0.1
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(True)

    def set_destination(self, src, dst):
        if self.agent is not None and dst is not None:
            self.agent.set_destination(src, dst)


class SynchronousAgent(ExperimentDirectory):
    def __init__(
            self,
            world: carla.World,
            args,
            transform: carla.Transform,
            agent_type: str,
            render_image: bool,
            evaluation: bool):
        self.world = world
        self.map = world.get_map()
        self.vehicle = None
        self.agent = None

        self.camera_sensor_dict = None
        self.segment_sensor_dict = None
        self.collision_sensor = None
        self.timing_dict = dict()
        self.transform_dict = dict()

        self.args = args
        self.agent_type = agent_type
        self.render_image = render_image
        self.evaluation = evaluation

        self.image_width = args.width
        self.image_height = args.height
        self.camera_keywords: List[str] = args.camera_keywords

        set_world_rendering_option(self.world, self.render_image)

        self.data_frame_dict = dict()
        self.data_frame_number_ = None
        self.progress_index = 0
        self.data_frame_buffer = set()
        self.stop_dict = dict()
        self.cmd_dict = dict()

        ExperimentDirectory.__init__(self, get_timestamp())
        self.target_waypoint_ = None
        self.waypoint_dict = dict()
        self.waypoint_buffer = set()

        set_world_asynchronous(self.world)

        self.set_vehicle(transform)
        if self.autopilot:
            self.set_agent()
        self.export_meta()

        set_world_synchronous(self.world)
        if self.render_image:
            self.set_camera_sensor()
            self.set_segment_sensor()
        self.set_collision_sensor()

    def reset(self):
        self.data_frame_dict = dict()
        self.data_frame_number_ = None
        self.progress_index = 0
        self.data_frame_buffer = set()
        if self.camera_sensor_dict is not None:
            for keyword in self.camera_sensor_dict.keys():
                self.camera_sensor_dict[keyword].image_frame_number = None
                self.camera_sensor_dict[keyword].image_frame = None
        if self.segment_sensor_dict is not None:
            for keyword in self.segment_sensor_dict.keys():
                self.segment_sensor_dict[keyword].image_frame_number = None
                self.segment_sensor_dict[keyword].image_frame = None
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        self.vehicle.apply_control(control)

    @property
    def autopilot(self) -> bool:
        return self.agent_type in ['roaming', 'basic']

    @property
    def image_frame_number(self):
        return self.camera_sensor_dict['center'].image_frame_number

    @property
    def image_frame(self):
        return self.camera_sensor_dict['center'].image_frame

    @property
    def segment_frame_number(self):
        return self.segment_sensor_dict['center'].image_frame_number

    @property
    def segment_frame(self):
        return self.segment_sensor_dict['center'].image_frame

    @property
    def route(self):
        if not self.autopilot:
            raise ValueError('autopilot was not set')
        return self.agent._route

    def set_vehicle(self, transform):
        blueprints = self.world.get_blueprint_library().filter('vehicle.audi.a2')
        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        if self.vehicle is not None and self.agent is not None:
            return

        blueprint_vehicle = random.choice(blueprints)
        blueprint_vehicle.set_attribute('role_name', 'hero')
        if blueprint_vehicle.has_attribute('color'):
            color = random.choice(blueprint_vehicle.get_attribute('color').recommended_values)
            blueprint_vehicle.set_attribute('color', color)
        blueprint_vehicle.set_attribute('role_name', 'autopilot')

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(blueprint_vehicle, transform)
        self.vehicle.set_autopilot(False)

    def move_vehicle(self, transform: carla.Transform):
        transform.location.z += 0.1
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(True)

    def set_destination(self, src, dst):
        if self.agent is not None and dst is not None:
            self.agent.set_destination(src, dst)

    def set_agent(self):
        if self.vehicle is None:
            raise ValueError('vehicle is not assigned')
        if self.agent_type == 'roaming':
            self.agent = RoamingAgent(self.vehicle, self.args.speed)
        elif self.agent_type == 'basic':
            self.agent = BasicAgent(self.vehicle, self.args.speed)
        else:
            raise TypeError('invalid agent type: {}'.format(self.agent_type))

    def set_camera_sensor(self):
        self.camera_sensor_dict = {
            camera_keyword: CameraSensor(
                self.vehicle,
                partial(self.image_path, camera_keyword=camera_keyword),
                self.timing_dict,
                self.transform_dict,
                self.image_width,
                self.image_height,
                camera_keyword)
            for camera_keyword in self.camera_keywords}
        self.camera_sensor_dict['extra'] = CameraSensor(
            self.vehicle, partial(self.image_path, camera_keyword='extra'),
            self.timing_dict, self.transform_dict, 1280, 720, 'extra')
            # self.timing_dict, 640, 480, 'extra')

    def set_segment_sensor(self):
        self.segment_sensor_dict = {
            camera_keyword: SegmentationSensor(
                self.vehicle,
                partial(self.segment_image_path, camera_keyword=camera_keyword),
                self.image_width,
                self.image_height,
                camera_keyword)
            for camera_keyword in self.camera_keywords}

    def set_collision_sensor(self):
        self.collision_sensor = CollisionSensor(self.vehicle)

    @property
    def data_frame_number(self):
        return self.data_frame_number_

    @data_frame_number.setter
    def data_frame_number(self, frame: int):
        if self.data_frame_number is None or self.data_frame_number < frame:
            self.data_frame_buffer.add(frame)
            self.data_frame_number_ = frame

    @property
    def target_waypoint(self):
        return self.target_waypoint_

    @target_waypoint.setter
    def target_waypoint(self, waypoint: carla.Waypoint):
        if waypoint is None:
            return
        if waypoint.id not in self.waypoint_dict:
            self.waypoint_buffer.add(waypoint.id)
            self.waypoint_dict[waypoint.id] = waypoint
        self.target_waypoint_ = waypoint

    def fetch_image_frame(self, camera_keyword: str) -> Tuple[int, np.ndarray]:
        if camera_keyword not in self.camera_sensor_dict:
            raise KeyError('invalid camera index {}'.format(camera_keyword))
        return self.camera_sensor_dict[camera_keyword].image_frame_number, \
               self.camera_sensor_dict[camera_keyword].image_frame

    def fetch_car_state(self) -> CarState:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        acceleration = self.vehicle.get_acceleration()
        return CarState(transform, velocity, angular_velocity, acceleration)

    def save_stop(self, frame: int, stop: float, sub_goal: str):
        if stop is not None:
            self.stop_dict[frame] = stop, sub_goal

    def save_cmd(self, frame: int, action_values: List[str]):
        self.cmd_dict[frame] = action_values

    def step_from_pilot(
            self,
            frame: int,
            apply: bool = True,
            update: bool = True,
            inject: float = 0.0) -> Tuple[carla.Waypoint, RoadOption, DriveDataFrame]:
        control_with_info: ControlWithInfo = self.agent.run_step(debug=False)
        vehicle_control: carla.VehicleControl = control_with_info.control
        vehicle_control.manual_gear_shift = False
        car_state = self.fetch_car_state()
        drive_data_frame = DriveDataFrame(car_state, control_with_info)
        velocity = car_state.velocity
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if apply:
            vehicle_control.steer += inject
            # if abs(inject) > 1e-3 and abs(vehicle_control.steer - control_with_info.control.steer) < 1e-3:
            #     logger.error('failed to inject noise')
            # print('{:+4.2f}, {:+4.2f}'.format(vehicle_control.throttle, vehicle_control.steer))
            self.vehicle.apply_control(vehicle_control)

        if update:
            self.data_frame_number = frame
            self.data_frame_dict[self.data_frame_number] = drive_data_frame
            self.target_waypoint = control_with_info.waypoint
            # assert control_with_info.has_waypoint
        return control_with_info.waypoint, control_with_info.road_option, drive_data_frame

    def step_from_control(
            self,
            frame: int,
            vehicle_control: carla.VehicleControl,
            apply: bool = True,
            update: bool = True) -> None:
        throttle_value = vehicle_control.throttle
        if apply:
            vehicle_control.manual_gear_shift = False
            if throttle_value < 0.4:
                vehicle_control.throttle = 0.4  # avoid stopping
            # todo: implement PID controller
            if self.data_frame_number is not None and self.data_frame_number in self.data_frame_dict:
                velocity = self.data_frame_dict[self.data_frame_number].state.velocity
                speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                # logger.info('speed {:+5.3f}'.format(speed))
                if speed > 20:
                    vehicle_control.throttle = 0.0
            self.vehicle.apply_control(vehicle_control)
        if update:
            car_control = CarControl.load_from_vehicle_control(vehicle_control)
            car_control.throttle = throttle_value
            control_with_info = ControlWithInfo(control=car_control, road_option=RoadOption.VOID)
            car_state = self.fetch_car_state()
            drive_data_frame = DriveDataFrame(car_state, control_with_info)
            self.data_frame_number = frame
            self.data_frame_dict[self.data_frame_number] = drive_data_frame

    def destroy(self):
        destroy_actor(self.vehicle)

    def __del__(self):
        self.destroy()

    def export_meta(self):
        meta = {
            'world_id': self.world.id,
            'map_name': self.world.get_map().name,
            'vehicle_id': self.vehicle.id,
            'vehicle_type_id': self.vehicle.type_id,
            'vehicle_color': self.vehicle.attributes['color']
        }
        with open(str(self.experiment_meta_path), 'w') as file:
            json.dump(meta, file, indent=4)

    def export_data(self):
        logger.info('export data: {} data frames, {} waypoints'.format(
            len(self.data_frame_buffer), len(self.waypoint_buffer)))
        frame_strs = []
        data_frame_numbers = sorted(self.data_frame_buffer)
        for data_frame_number in data_frame_numbers:
            frame_strs.append('{}:{}'.format(self.frame_str(data_frame_number), self.data_frame_dict[data_frame_number]))
        with open(str(self.experiment_data_path), 'a') as file:
            file.write('\n'.join(frame_strs) + '\n')
        self.data_frame_buffer.clear()

        waypoint_strs = []
        for waypoint_id in self.waypoint_buffer:
            waypoint_strs.append(str_from_waypoint(self.waypoint_dict[waypoint_id]))
        with open(str(self.experiment_waypoint_path), 'a') as file:
            file.write('\n'.join(waypoint_strs) + '\n')
        self.waypoint_buffer.clear()

    def export_eval_data(self, collided: bool, sentence) -> dict:
        logger.info('export data: {} data frames'.format(len(self.data_frame_buffer)))
        data_frame_numbers = sorted(self.data_frame_buffer)
        if not data_frame_numbers:
            return None
        data_frame_range = data_frame_numbers[0], data_frame_numbers[-1] + 1
        data_frame_strs = [str(self.data_frame_dict[f]) for f in data_frame_numbers]
        stop_sub_goal_lists = [self.stop_dict[f] if f in self.stop_dict else (0.0, 'None') for f in data_frame_numbers]
        if list(self.cmd_dict.keys()):
            sentences = [self.cmd_dict[f] for f in data_frame_numbers]
        else:
            sentences = []
        self.data_frame_buffer.clear()
        return {
            'sentence': sentence,
            'frame_range': data_frame_range,
            'collided': collided,
            'data_frames': data_frame_strs,
            'stop_frames': stop_sub_goal_lists,
            'sentences': sentences
        }


def show_game(
        display,
        font,
        image,
        clock,
        road_option = None,
        is_intersection = None,
        extra_str: str = ''):
    draw_image(display, image)
    strs = ['{:5.3f}'.format(clock.get_fps())]
    if road_option is not None:
        strs += [road_option.name.lower()]
    if is_intersection is not None:
        strs += [str(is_intersection)]
    if extra_str:
        strs += [extra_str]
    text_surface = font.render(', '.join(strs), True, (255, 255, 255))
    display.blit(text_surface, (8, 10))
    pygame.display.flip()


class GameEnvironment:
    def __init__(self, args, agent_type: str, transform_index: int = 0):
        self.args = args
        self.transform_index = transform_index

        set_random_seed(0)

        self.agent_type = agent_type
        self.evaluation = True if agent_type == 'evaluation' else False
        self.client = None
        self.world = None
        self.agent = None

        self.display = None
        self.font = None
        self.render_image = not args.no_rendering
        self.show_image = args.show_game if self.render_image else False

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.transforms = self.world.get_map().get_spawn_points()
        random.shuffle(self.transforms)

        if self.show_image:
            import pygame
            pygame.init()
            self.display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()

        set_world_asynchronous(self.world)
        clean_vehicles(self.world)
        set_traffic_lights_green(self.world)
        # self.agent = SynchronousAgent(
        #     world=self.world,
        #     args=self.args,
        #     transform=self.transforms[self.transform_index],
        #     agent_type=self.agent_type,
        #     render_image=self.render_image,
        #     evaluation=self.evaluation)
        assert self.world.get_settings().synchronous_mode

    @property
    def transform_index(self):
        return self.transform_index_ % len(self.transforms)

    @transform_index.setter
    def transform_index(self, value: int):
        self.transform_index_ = value

    def show(self, image, clock, road_option = None, is_intersection = None, extra_str: str = ''):
        assert self.show_image
        show_game(self.display, self.font, image, clock, road_option, is_intersection, extra_str)

    def run(self) -> None:
        raise NotImplementedError


def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (200, 88),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    collision_sensor = None
    camera_sensor_dict = dict()
    segmentation_sensor_dict = dict()
    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_sensor_dict, segmentation_sensor_dict, collision_sensor = set_all_sensors(
            world, vehicle, ['right', 'center', 'left'], 200, 88, False)
        print(vehicle.id)
        with CarlaSyncWrapper(world, camera_sensor_dict, segmentation_sensor_dict) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                snapshot, rgb_dict, seg_dict = sync_mode.tick(timeout=2.0)
                # if snapshot.has_actor(vehicle.id):
                #     actor_snapshot = snapshot.find(vehicle.id)
                #     print(actor_snapshot.get_transform())

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                for keyword in seg_dict.keys():
                    seg_dict[keyword].convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                print(snapshot.timestamp)

                # Draw the display.
                # draw_image(display, rgb_dict['center'])
                draw_image(display, seg_dict['center'])
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
        for actor in actor_list:
            actor.destroy()
        for sensor in camera_sensor_dict.values():
            sensor.destroy()
        for sensor in segmentation_sensor_dict.values():
            sensor.destroy()
        if collision_sensor is not None:
            collision_sensor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    main()
