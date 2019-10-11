import json
import math
import random
import weakref
from collections import defaultdict
from functools import partial
from operator import attrgetter
from time import perf_counter
from typing import Tuple, List

import cv2
from custom_carla.agents.navigation.agent import ControlWithInfo
from custom_carla.agents.navigation.basic_agent import BasicAgent
from custom_carla.agents.navigation.local_planner import RoadOption
from custom_carla.agents.navigation.roaming_agent import RoamingAgent

from config import EVAL_FRAMERATE_SCALE, DATASET_FRAMERATE, CAMERA_KEYWORDS
from data.types import CarState, CarControl, DriveDataFrame
from util.common import add_carla_module, get_logger, get_timestamp, set_random_seed, get_current_time
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


def destroy_actor(actor):
    if actor is not None and actor.is_alive:
        actor.destroy()


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


def draw_image(surface, image: np.ndarray):
    array = image[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
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


class SensorBase:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.get_world()
        self.sensor = self.generate_sensor()

    def generate_sensor(self):
        raise NotImplementedError

    def destroy(self):
        destroy_actor(self.sensor)

    def __del__(self):
        self.destroy()


class CollisionSensor(SensorBase):
    def __init__(self, parent_actor):
        SensorBase.__init__(self, parent_actor)
        self.history = defaultdict(int)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        return self.world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

    def has_collided(self, frame_number: int) -> bool:
        return self.history[frame_number] > 0

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        # actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history[event.frame_number] += intensity


def numpy_from_carla_image(carla_image: carla.Image) -> np.ndarray:
    np_image = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape(carla_image.height, carla_image.width, 4)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
    return rgb_image


CAMERA_SHIFT = 0.4
__camera_transforms__ = {
    'center': carla.Transform(carla.Location(x=1.6, z=1.7)),
    'left': carla.Transform(
        carla.Location(x=1.6, y=-CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(-CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'right': carla.Transform(
        carla.Location(x=1.6, y=+CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'extra': carla.Transform(carla.Location(x=1.6, z=1.7))
}
for key in __camera_transforms__.keys():
    assert key in CAMERA_KEYWORDS
for key in CAMERA_KEYWORDS:
    assert key in __camera_transforms__


class CameraSensor(SensorBase):
    def __init__(self, parent_actor, image_path_func, timing_dict, transform_dict, width, height, camera_keyword: str):
        self.width = width
        self.height = height
        self.camera_keyword = camera_keyword
        SensorBase.__init__(self, parent_actor)
        self.image_path_func = image_path_func
        self.image_frame_number = None
        self.image_frame = None
        self.timing_dict = timing_dict
        self.transform_dict = transform_dict
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    @staticmethod
    def on_listen(weak_self, carla_image: carla.Image):
        self = weak_self()
        if not self:
            return
        frame_number = carla_image.frame_number
        self.timing_dict[frame_number] = get_current_time()
        self.transform_dict[frame_number] = self.sensor.get_transform()
        self.image_frame_number = frame_number
        numpy_image = numpy_from_carla_image(carla_image)
        self.image_frame = numpy_image
        # print(frame_number, self.image_frame.shape, self.image_path_func(frame_number))
        cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)


class SegmentationSensor(SensorBase):
    def __init__(self, parent_actor, image_path_func, width, height, camera_keyword: str):
        self.width = width
        self.height = height
        self.camera_keyword = camera_keyword
        SensorBase.__init__(self, parent_actor)
        self.image_path_func = image_path_func
        self.image_frame_number = None
        self.image_frame = None
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: SegmentationSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    @staticmethod
    def on_listen(weak_self, carla_image: carla.Image):
        self = weak_self()
        if not self:
            return
        frame_number = carla_image.frame_number
        self.image_frame_number = frame_number
        numpy_image = numpy_from_carla_image(carla_image)
        self.image_frame = numpy_image
        cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)


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
                self.image_width * self.args.display_scale,
                self.image_height * self.args.display_scale,
                camera_keyword)
            for camera_keyword in ['center']}
        # self.camera_sensor_dict['extra'] = CameraSensor(
        #     self.vehicle, partial(self.image_path, camera_keyword='extra'),
        #     self.timing_dict, self.transform_dict, 1280, 720, 'extra')

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
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.transforms = self.world.get_map().get_spawn_points()
        random.shuffle(self.transforms)

        if self.show_image:
            import pygame
            pygame.init()
            self.display = pygame.display.set_mode(
                (args.width * args.display_scale, args.height * args.display_scale),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()

        set_world_asynchronous(self.world)
        clean_vehicles(self.world)
        set_traffic_lights_green(self.world)
        self.agent = SynchronousAgent(
            world=self.world,
            args=self.args,
            transform=self.transforms[self.transform_index],
            agent_type=self.agent_type,
            render_image=self.render_image,
            evaluation=self.evaluation)
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
