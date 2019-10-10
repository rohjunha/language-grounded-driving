from typing import Dict, Any

import carla
import cv2
import numpy as np
import pygame

from custom_carla.agents.navigation.agent import ControlWithInfo
from data.types import CarState, DriveDataFrame
from util.directory import ExperimentDirectory
from util.serialize import str_from_waypoint


def destroy_actor(actor):
    if actor is not None and actor.is_alive:
        actor.destroy()


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def np_from_carla_image(image, reverse: bool = True):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    if reverse:
        array = array[:, :, ::-1]
    return array


def draw_image(surface, image, blend=False):
    if not isinstance(image, np.ndarray):
        array = np_from_carla_image(image)
    else:
        array = image
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


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
        for key, value in self.rgb_dict.items():
            self.rgb_dict[key] = np_from_carla_image(value, False)
        for key, value in self.seg_dict.items():
            self.seg_dict[key] = np_from_carla_image(value)
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
        if self.control is not None and isinstance(self.control, ControlWithInfo):
            return self.control.waypoint
        else:
            return None


class SnapshotSaver(ExperimentDirectory):
    def __init__(self, timestamp):
        ExperimentDirectory.__init__(self, timestamp)

    def save(self, frame_snapshot: FrameSnapshot):
        for keyword in frame_snapshot.rgb_dict.keys():
            cv2.imwrite(str(self.image_path(frame_snapshot.frame, keyword)), frame_snapshot.rgb_dict[keyword])
        for keyword in frame_snapshot.seg_dict.keys():
            cv2.imwrite(str(self.segment_image_path(frame_snapshot.frame, keyword)), frame_snapshot.seg_dict[keyword])
        # with open(str(self.experiment_data_path), 'a') as file:
        #     file.write('{}:{}\n'.format(self.frame_str(frame_snapshot.frame), frame_snapshot.data_frame))
        # if frame_snapshot.waypoint is not None:
        #     with open(str(self.experiment_waypoint_path), 'a') as file:
        #         file.write('{}\n'.format(str_from_waypoint(frame_snapshot.waypoint)))