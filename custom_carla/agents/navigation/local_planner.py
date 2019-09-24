#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """
import random
from collections import deque
from enum import Enum
from functools import partial
from math import sqrt
from typing import List, Callable

import carla
import numpy as np
from custom_carla.agents.navigation.agent import ControlWithInfo, WaypointWithInfo, WaypointModifiableLocation
from custom_carla.agents.navigation.controller import VehiclePIDController
from custom_carla.agents.tools.misc import draw_waypoints

from util.common import get_logger, unique_with_islices

logger = get_logger(__name__)


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


ROAD_OPTION_ACTIONS = [RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.RIGHT]
ROAD_OPTION_NON_ACTIONS = [RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._target_possible_options = []
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoint_list = []

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.25 / 3.6  # 1 seconds horizon
        # self._left_wsize = 11
        # self._right_wsize = 7
        # self._default_wsize = 2
        # self._num_smooth = 1
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}
        self._noise_std: float = 0.0

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                    opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'noise_std' in opt_dict:
                self._noise_std = opt_dict['noise_std']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                       args_lateral=args_lateral_dict,
                                                       args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        next_waypoint = WaypointModifiableLocation(self._current_waypoint.next(self._sampling_radius)[0])
        waypoint_with_info = WaypointWithInfo(next_waypoint, RoadOption.LANEFOLLOW, [RoadOption.LANEFOLLOW])
        self._waypoints_queue.append(waypoint_with_info)

        self._target_road_option = RoadOption.LANEFOLLOW
        self._target_possible_options = [RoadOption.LANEFOLLOW]
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        def generate_new_waypoint(last_waypoint: carla.Waypoint) -> WaypointWithInfo:
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
                possible_options = [RoadOption.LANEFOLLOW]
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]
                possible_options = road_options_list

            if abs(self._noise_std) > 1e-3:
                next_waypoint.transform.location.x += np.random.normal(scale=self._noise_std, size=1)[0]
                next_waypoint.transform.location.y += np.random.normal(scale=self._noise_std, size=1)[0]
            return WaypointWithInfo(WaypointModifiableLocation(next_waypoint), road_option, possible_options)

        waypoint_list: List[WaypointWithInfo] = [self._waypoints_queue[-1]]
        for _ in range(k):
            last_waypoint: carla.Waypoint = waypoint_list[-1].waypoint.waypoint
            waypoint_list.append(generate_new_waypoint(last_waypoint))
        del waypoint_list[0]

        def get_smoothed_waypoint_transform(waypoints: List[WaypointWithInfo]):
            xs = list(map(lambda x: x.waypoint.x, waypoints))
            ys = list(map(lambda x: x.waypoint.y, waypoints))
            return sum(xs) / len(xs), sum(ys) / len(ys)

        def has_left_turn(waypoints: List[WaypointWithInfo]):
            return any(list(map(lambda x: x.road_option == RoadOption.LEFT, waypoints)))

        def has_right_turn(waypoints: List[WaypointWithInfo]):
            return any(list(map(lambda x: x.road_option == RoadOption.RIGHT, waypoints)))

        def get_values(values, index, wsize, centered: bool):
            if centered:
                hsize = (wsize - 1 // 2)
                i1 = max(0, index - hsize)
                i2 = min(len(values), index + hsize + 1)
            else:
                i1 = index
                i2 = min(len(values), index + wsize + 1)
            return values[i1:i2]

        def predict_turn(
                wl: List[WaypointWithInfo],
                detector: Callable[[List[WaypointWithInfo]], List[bool]],
                wsize: int,
                centered: bool) -> List[bool]:
            al = [detector(get_values(wl, i, wsize, centered)) for i in range(len(wl))]
            nal = list(filter(lambda i: al[i - 1] and not al[i], range(1, len(al))))
            for i in nal:
                i2 = min(i + wsize, len(wl))
                for j in range(i, i2):
                    al[j] = True
            return al

        def smooth_waypoint_list_(
                waypoint_list: List[WaypointWithInfo],
                left_list: List[bool],
                right_list: List[bool],
                left_sampler: Callable[[], int],
                right_sampler: Callable[[], int],
                default_wsize: int,
                centered: bool) -> List[WaypointWithInfo]:
            indicators = ['l' if l else ('r' if r else 's') for l, r in zip(left_list, right_list)]
            wsizes = [default_wsize for _ in indicators]
            for indicator, index_range in unique_with_islices(indicators):
                if indicator == 's':
                    continue
                wsize = left_sampler() if indicator == 'l' else right_sampler()
                for i in range(*index_range):
                    wsizes[i] = wsize
            logger.info(wsizes)

            new_waypoint_list = []
            location_list = []
            for i, wsize in enumerate(wsizes):
                values = get_values(waypoint_list, i, wsize, centered)
                location_list.append(get_smoothed_waypoint_transform(values))
            for i, (x, y) in enumerate(location_list):
                new_waypoint = waypoint_list[i]
                new_waypoint.waypoint.x = x
                new_waypoint.waypoint.y = y
                new_waypoint_list.append(new_waypoint)
            return new_waypoint_list

        def smooth_waypoint_list(
                waypoint_list: List[WaypointWithInfo],
                left_sampler: Callable[[], int],
                right_sampler: Callable[[], int],
                default_wsize: int,
                centered: bool) -> List[WaypointWithInfo]:
            left_list = predict_turn(waypoint_list, has_left_turn, 15, centered)
            right_list = predict_turn(waypoint_list, has_right_turn, 15, centered)
            return smooth_waypoint_list_(waypoint_list, left_list, right_list,
                                         left_sampler, right_sampler, default_wsize, centered)

        def integer_sampler(i1, i2):
            return np.random.randint(i1, i2, 1)[0]

        def second_sampler():
            return 3

        left_sampler = partial(integer_sampler, 7, 15)
        right_sampler = partial(integer_sampler, 5, 17)

        # left_size = 5  # np.random.randint(7, 15, 1)[0]
        # right_size = 5  # np.random.randint(5, 17, 1)[0]
        default_size = 8
        waypoint_list = smooth_waypoint_list(waypoint_list, left_sampler, right_sampler, default_size, centered=False)
        waypoint_list = smooth_waypoint_list(waypoint_list, second_sampler, second_sampler, 2, centered=False)
        for w in waypoint_list:
            self._waypoints_queue.append(w)

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        self._waypoint_buffer.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._target_possible_options = [RoadOption.LANEFOLLOW]
        self._global_plan = True

    @property
    def finish(self):
        return len(self._waypoints_queue) == 0

    @property
    def target_waypoint_with_info(self) -> WaypointWithInfo:
        return WaypointWithInfo(self.target_waypoint, self._target_road_option, self._target_possible_options)

    def run_step(self, debug=True) -> ControlWithInfo:
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=10000)

        if len(self._waypoints_queue) == 0:
            return ControlWithInfo()

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        self._current_waypoint: carla.Waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint: WaypointModifiableLocation = self._waypoint_buffer[0].waypoint
        self._target_road_option = self._waypoint_buffer[0].road_option
        self._target_possible_options = self._waypoint_buffer[0].possible_road_options
        self._waypoint_list.append(self.target_waypoint_with_info)
        if len(self._waypoint_list) > 100:
            self._waypoint_list = self._waypoint_list[-100:]
        current_option = RoadOption.LANEFOLLOW
        current_possible_options = [RoadOption.LANEFOLLOW]

        def compute_dist(l1, l2):
            return sqrt((l1.x - l2.x) ** 2 + (l1.y - l2.y) ** 2)

        if self._waypoint_list:
            current_location = self._current_waypoint.transform.location
            compute_dist_ = partial(compute_dist, current_location)
            dists = list(map(compute_dist_, map(lambda x: x.waypoint.location, self._waypoint_list)))
            dists = sorted(enumerate(dists), key=lambda x: x[1])  # rel_index, dist
            best_info = self._waypoint_list[dists[0][0]]
            if best_info.waypoint.id != self.target_waypoint.id:
                current_option = best_info.road_option
                current_possible_options = best_info.possible_road_options

        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint.location)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        def distance_vehicle_location(location, vehicle_transform):
            loc = vehicle_transform.location
            dx = location.x - loc.x
            dy = location.y - loc.y
            return sqrt(dx * dx + dy * dy)

        for i, info in enumerate(self._waypoint_buffer):
            if distance_vehicle_location(info.waypoint.location, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(
                self._vehicle.get_world(),
                [self.target_waypoint.waypoint],
                self._vehicle.get_location().z + 1.0)

        # if current_option in [RoadOption.LEFT, RoadOption.RIGHT] or abs(control.steer) > 0.3:
        #     control.throttle = min(0.75, control.throttle)

        return ControlWithInfo(
            control=control,
            waypoint=self._current_waypoint,
            road_option=current_option,
            possible_road_options=current_possible_options)


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    t = (n - c) % 180.0
    str_old = None
    if t < 1.0:
        str_old = 'straight'
    elif t > 90.0:
        str_old = 'left'
    else:
        str_old = 'right'

    n -= 180.0
    c -= 180.0
    assert -180 <= n < 180
    assert -180 <= c < 180
    diff_angle = n - c
    diff_angle += -360 if diff_angle > 180 else (360 if diff_angle < -180 else 0)

    angle_threshold = 5

    str_new = None
    if abs(diff_angle) < angle_threshold:
        str_new = 'straight'
        # return RoadOption.STRAIGHT
    elif diff_angle < 0.0:
        str_new = 'left'
        # return RoadOption.LEFT
    else:
        str_new = 'right'
        # return RoadOption.RIGHT

    # print(str_old, str_new, t, diff_angle)
    if str_new == 'straight':
        return RoadOption.STRAIGHT
    elif str_new == 'left':
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
