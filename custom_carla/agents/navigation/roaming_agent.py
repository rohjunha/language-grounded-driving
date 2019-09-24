#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from custom_carla.agents.navigation.agent import Agent, AgentState, ControlWithInfo
from custom_carla.agents.navigation.local_planner import LocalPlanner

from config import DATASET_FRAMERATE


class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed: int = 20, noise_std: float = 0.0):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        args_lateral_dict = {
            'K_P': 1,
            'K_I': 0.7,
            'K_D': 0.01,
            'dt': 1.0 / DATASET_FRAMERATE}
        args_longitudinal_dict = {
            'K_P': 0.15,
            'K_I': 0,
            'K_D': 0,
            'dt': 1.0 / DATASET_FRAMERATE}
        self._local_planner_opt_dict = {
            'dt': 1.0 / DATASET_FRAMERATE,
            'target_speed': target_speed,
            'lateral_control_dict': args_lateral_dict,
            'longitudinal_control_dict': args_longitudinal_dict,
            'noise_std': noise_std}
        self._local_planner = LocalPlanner(self._vehicle, self._local_planner_opt_dict)

    def reset_planner(self):
        self._local_planner._waypoints_queue.clear()
        self._local_planner._waypoint_buffer.clear()
        self._local_planner._init_controller(self._local_planner_opt_dict)

    def restart(self, vehicle):
        Agent.restart(self, vehicle)
        self._local_planner = LocalPlanner(self._vehicle)

    def run_step(self, debug=False) -> ControlWithInfo:
        """
        Execute one step of navigation.
        :return: custom_carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            return ControlWithInfo.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            return self._local_planner.run_step(debug=debug)
