from math import sqrt
from typing import List

from custom_carla.agents.navigation.agent import ControlWithInfo
from util.common import add_carla_module
from util.road_option import fetch_name_from_road_option, fetch_road_option_from_str
from util.serialize import list_from_vector, parse_bool

add_carla_module()
import carla


class CarState:
    def __init__(self, transform, velocity, angular_velocity, acceleration):
        self.transform = transform
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration

    def to_dict(self):
        return {
            'x': self.transform.location.x,
            'y': self.transform.location.y,
            'z': self.transform.location.z,
            'pitch': self.transform.rotation.pitch,
            'yaw': self.transform.rotation.yaw,
            'roll': self.transform.rotation.roll,
            'velocity': list_from_vector(self.velocity),
            'angular_velocity': list_from_vector(self.angular_velocity),
            'acceleration': list_from_vector(self.acceleration)}

    def to_str(self):
        loc = list_from_vector(self.transform.location)
        rot = [self.transform.rotation.pitch, self.transform.rotation.yaw, self.transform.rotation.roll]
        vel = list_from_vector(self.velocity)
        ang = list_from_vector(self.angular_velocity)
        acc = list_from_vector(self.acceleration)
        values = loc + rot + vel + ang + acc
        return ','.join([str(v) for v in values])

    def __str__(self):
        return self.to_str()

    @staticmethod
    def load_from_str(state_str: str):
        words = state_str.split(',')
        assert len(words) == 15
        transform = carla.Transform()
        transform.location.x = float(words[0])
        transform.location.y = float(words[1])
        transform.location.z = float(words[2])
        transform.rotation.pitch = float(words[3])
        transform.rotation.yaw = float(words[4])
        transform.rotation.roll = float(words[5])
        velocity = carla.Vector3D()
        velocity.x = float(words[6])
        velocity.y = float(words[7])
        velocity.z = float(words[8])
        angular_velocity = carla.Vector3D()
        angular_velocity.x = float(words[9])
        angular_velocity.y = float(words[10])
        angular_velocity.z = float(words[11])
        acceleration = carla.Vector3D()
        acceleration.x = float(words[12])
        acceleration.y = float(words[13])
        acceleration.z = float(words[14])
        return CarState(transform, velocity, angular_velocity, acceleration)


class CarControl(carla.VehicleControl):
    def __init__(self):
        carla.VehicleControl.__init__(self)

    def to_str(self):
        values = [
            self.throttle,
            self.steer,
            self.brake,
            self.hand_brake,
            self.reverse,
            self.gear,
            self.manual_gear_shift]
        return ','.join([str(v) for v in values])

    @staticmethod
    def load_from_str(control_str: str):
        words = control_str.split(',')
        assert len(words) == 7
        control = CarControl()
        control.throttle = float(words[0])
        control.steer = float(words[1])
        control.brake = float(words[2])
        control.hand_brake = parse_bool(words[3])
        control.reverse = parse_bool(words[4])
        control.gear = int(words[5])
        control.manual_gear_shift = parse_bool(words[6])
        return control

    @staticmethod
    def load_from_vehicle_control(vehicle_control: carla.VehicleControl):
        control = CarControl()
        control.throttle = vehicle_control.throttle
        control.steer = vehicle_control.steer
        control.brake = vehicle_control.brake
        control.hand_brake = vehicle_control.hand_brake
        control.reverse = vehicle_control.reverse
        control.gear = vehicle_control.gear
        control.manual_gear_shift = vehicle_control.manual_gear_shift
        return control

    @property
    def default_(self):
        return abs(self.throttle) < 1e-3 and \
               abs(self.steer) < 1e-3 and \
               not self.hand_brake and \
               not self.reverse and \
               self.gear == 0 and \
               not self.manual_gear_shift

    @property
    def empty(self):
        return self.default_ and abs(self.brake) < 1e-3

    @property
    def emergency_stop(self):
        return self.default_ and self.brake > 0.0

    def __str__(self):
        return self.to_str()


class DriveDataFrame:
    def __init__(self, state: CarState, control_with_info: ControlWithInfo):
        self.state = state
        self.control = CarControl.load_from_vehicle_control(control_with_info.control)
        self.waypoint_id = control_with_info.waypoint_id
        self.road_option = control_with_info.road_option
        self.possible_road_options = control_with_info.possible_road_options

    def to_str(self):
        state_str = str(self.state)
        control_str = str(self.control)
        waypoint_id_str = str(self.waypoint_id)
        road_option_str = fetch_name_from_road_option(self.road_option)
        possible_road_option_str = '|'.join([fetch_name_from_road_option(ro) for ro in self.possible_road_options])
        strs = [state_str, control_str, waypoint_id_str, road_option_str, possible_road_option_str]
        return ';'.join(strs)

    @staticmethod
    def load_from_str(data_str: str):
        words = list(filter(lambda x: x, data_str.split(';')))
        assert len(words) in [4, 5]
        car_state = CarState.load_from_str(words[0])
        car_control = CarControl.load_from_str(words[1])
        waypoint_id = int(words[2])
        road_option = None if words[3] == 'None' else fetch_road_option_from_str(words[3])
        if len(words) == 4:
            possible_road_options = [road_option]
        else:
            possible_road_options = [fetch_road_option_from_str(w) for w in words[4].split('|')]
        control_with_info = ControlWithInfo(
            control=car_control,
            waypoint_id=waypoint_id,
            road_option=road_option,
            possible_road_options=possible_road_options)
        return DriveDataFrame(car_state, control_with_info)

    def __str__(self):
        return self.to_str()


def compute_dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class LengthComputer:
    def __init__(self):
        self.positions = []
        self.length = 0.0

    def reset(self):
        self.positions = []
        self.length = 0.0

    def __call__(self, location):
        position = location.x, location.y
        self.positions.append(position)
        if len(self.positions) > 1:
            self.length += compute_dist(self.positions[-1], self.positions[-2])


class FrameInfo:
    def __init__(
            self,
            index: str,
            x: float,
            y: float,
            theta: float,
            is_intersection: bool,
            option: str,
            possible_options: List[str]):
        self.index = int(index)
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.is_intersection = is_intersection
        self.option = option.lower()
        self.possible_options = [option.lower() for option in possible_options]


class SegmentInfo:
    def __init__(
            self,
            index_range,
            is_intersection,
            option,
            possible_options,
            has_stop: bool,
            split_index: int = -1,
            ref_index: int = -1):
        self.index_range = index_range
        self.split_index = split_index
        self.is_intersection = is_intersection
        self.option = option
        self.possible_options = possible_options
        self.has_stop = has_stop
        self.ref_index = ref_index

    @property
    def mid_index(self):
        return (self.index_range[0] + self.index_range[1]) // 2

    @property
    def last_index(self):
        return self.index_range[1] - 1
