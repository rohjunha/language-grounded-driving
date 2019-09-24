from collections import namedtuple
from typing import List, Tuple, Dict
from util.common import add_carla_module, get_logger

logger = get_logger(__name__)
add_carla_module()
import carla
from custom_carla.agents.navigation.local_planner import RoadOption


def list_from_vector(value: carla.Vector3D):
    return [value.x, value.y, value.z]


# def control_from_str(control_str: str):
#     words = control_str.split(',')
#     assert len(words) == 7
#     control = custom_carla.VehicleControl()
#     control.throttle = float(words[0])
#     control.steer = float(words[1])
#     control.brake = float(words[2])
#     control.hand_brake = parse_bool(words[3])
#     control.reverse = parse_bool(words[4])
#     control.gear = int(words[5])
#     control.manual_gear_shift = parse_bool(words[6])
#     return control


# def str_from_control(control):
#     values = [
#         control.throttle,
#         control.steer,
#         control.brake,
#         control.hand_brake,
#         control.reverse,
#         control.gear,
#         control.manual_gear_shift]
#     return ','.join([str(v) for v in values])


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ['False', 'false']:
        return False
    elif value in ['True', 'true']:
        return True
    else:
        raise ValueError('invalid value for bool: {}'.format(value))


# def dict_from_control(control):
#     return {
#         'throttle': control.throttle,
#         'steer': control.steer,
#         'brake': control.brake,
#         'hand_brake': control.hand_brake,
#         'reverse': control.reverse,
#         'gear': control.gear,
#         'manual_gear_shift': control.manual_gear_shift
#     }


# def dict_from_state(transform, velocity, angular_velocity, acceleration):
#     state = {
#         'x': transform.location.x,
#         'y': transform.location.y,
#         'z': transform.location.z,
#         'pitch': transform.rotation.pitch,
#         'yaw': transform.rotation.yaw,
#         'roll': transform.rotation.roll,
#         'velocity': list_from_vector(velocity),
#         'angular_velocity': list_from_vector(angular_velocity),
#         'acceleration': list_from_vector(acceleration)
#     }
#     return state


# def str_from_state(transform, velocity, angular_velocity, acceleration):
#     loc = list_from_vector(transform.location)
#     rot = [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]
#     vel = list_from_vector(velocity)
#     ang = list_from_vector(angular_velocity)
#     acc = list_from_vector(acceleration)
#     values = loc + rot + vel + ang + acc
#     return ','.join([str(v) for v in values])


# def state_from_str(state_str: str):
#     words = state_str.split(',')
#     assert len(words) == 15
#     transform = custom_carla.Transform()
#     transform.location.x = float(words[0])
#     transform.location.y = float(words[1])
#     transform.location.z = float(words[2])
#     transform.rotation.pitch = float(words[3])
#     transform.rotation.yaw = float(words[4])
#     transform.rotation.roll = float(words[5])
#     velocity = custom_carla.Vector3D()
#     velocity.x = float(words[6])
#     velocity.y = float(words[7])
#     velocity.z = float(words[8])
#     angular_velocity = custom_carla.Vector3D()
#     angular_velocity.x = float(words[9])
#     angular_velocity.y = float(words[10])
#     angular_velocity.z = float(words[11])
#     acceleration = custom_carla.Vector3D()
#     acceleration.x = float(words[12])
#     acceleration.y = float(words[13])
#     acceleration.z = float(words[14])
#     return transform, velocity, angular_velocity, acceleration


def hash_from_transform(t: carla.Transform) -> str:
    l, r = t.location, t.rotation
    return ','.join(['{:+8.6f}'.format(v) for v in [l.x, l.y, l.z] + [r.pitch, r.yaw, r.roll]])


def transform_from_hash(hash: str) -> carla.Transform:
    values = [float(v) for v in hash.split(',')]
    t = carla.Transform()
    t.location = carla.Location(*values[:3])
    t.rotation = carla.Rotation(*values[3:])
    return t


def dict_from_lane_marking(lm: carla.LaneMarking):
    data = {
        'type': lm.type,
        'color': lm.color,
        'lane_change': lm.lane_change,
        'width': lm.width
    }
    return data


def str_from_lane_marking(lm: carla.LaneMarking):
    return ','.join([str(lm.type), str(lm.color), str(lm.lane_change), str(lm.width)])


CustomLaneMarking = namedtuple('CustomLaneMarking', ['type', 'color', 'lane_change', 'width'])


def lane_marking_from_dict(data_dict: dict) -> CustomLaneMarking:
    values = [
        carla.LaneMarkingType(data_dict['type']),
        carla.LaneMarkingColor(data_dict['color']),
        carla.LaneChange(data_dict['lane_change']),
        int(data_dict['width'])]
    return CustomLaneMarking(*values)


def lane_marking_from_str(lm_str: str) -> CustomLaneMarking:
    words = lm_str.split(',')
    assert len(words) == 4
    values = [
        carla.LaneMarkingType.__dict__['names'][words[0]],
        carla.LaneMarkingColor.__dict__['names'][words[1]],
        carla.LaneChange.__dict__['names'][words[2]],
        float(words[3])]
    return CustomLaneMarking(*values)


def dict_from_waypoint(waypoint: carla.Waypoint):
    data = {
        'id': waypoint.id,
        'transform': hash_from_transform(waypoint.transform),
        'is_intersection': waypoint.is_intersection,
        'lane_width': waypoint.lane_width,
        'road_id': waypoint.road_id,
        'section_id': waypoint.section_id,
        'lane_id': waypoint.lane_id,
        's': waypoint.s,
        'lane_change': waypoint.lane_change,
        'lane_type': waypoint.lane_type,
        'left_lane_marking': dict_from_lane_marking(waypoint.left_lane_marking),
        'right_lane_marking': dict_from_lane_marking(waypoint.right_lane_marking)
    }
    return data


def str_from_waypoint(waypoint: carla.Waypoint):
    strs = [str(waypoint.id),
            hash_from_transform(waypoint.transform),
            str(waypoint.is_intersection),
            str(waypoint.lane_width),
            str(waypoint.road_id),
            str(waypoint.section_id),
            str(waypoint.lane_id),
            str(waypoint.s),
            str(waypoint.lane_change),
            str(waypoint.lane_type),
            str_from_lane_marking(waypoint.left_lane_marking),
            str_from_lane_marking(waypoint.right_lane_marking)]
    return ','.join(strs)


CustomWaypoint = namedtuple('CustomWaypoint',
                            ['id', 'transform', 'is_intersection', 'lane_width', 'road_id', 'section_id', 'lane_id', 's',
                             'lane_change', 'lane_type', 'left_lane_marking', 'right_lane_marking'])


def waypoint_from_dict(data_dict: dict) -> CustomWaypoint:
    values = [int(data_dict['id']),
              transform_from_hash(data_dict['transform']),
              parse_bool(data_dict['is_intersection']),
              float(data_dict['lane_width']),
              int(data_dict['road_id']),
              int(data_dict['section_id']),
              int(data_dict['lane_id']),
              float(data_dict['s']),
              carla.LaneChange(data_dict['lane_change']),
              carla.LaneType(data_dict['lane_type']),
              lane_marking_from_dict(data_dict['left_lane_marking']),
              lane_marking_from_dict(data_dict['right_lane_marking'])]
    return CustomWaypoint(*values)


def waypoint_from_str(waypoint_str: str) -> CustomWaypoint:
    words = waypoint_str.split(',')
    assert len(words) == 23
    values = [int(words[0]),  # id
              transform_from_hash(','.join(words[1:7])),  # transform (6-dim)
              parse_bool(words[7]),  # is_intersection
              float(words[8]),  # lane_width
              int(words[9]),  # road_id
              int(words[10]),  # section_id
              int(words[11]),  # lane_id
              float(words[12]),  # s
              carla.LaneChange.__dict__['names'][words[13]],  # lane_change
              carla.LaneType.__dict__['names'][words[14]],
              lane_marking_from_str(','.join(words[15:19])),
              lane_marking_from_str(','.join(words[19:23]))]
    return CustomWaypoint(*values)


class RawSegment:
    def __init__(self, road_option: RoadOption, index_range: Tuple[int, int], step: int = 1, has_stop: bool = True):
        self.road_option = road_option
        self.index_range = index_range
        self.step = step
        self.has_stop = has_stop

    @property
    def range(self):
        return range(self.index_range[0], self.index_range[1], self.step)

    def __str__(self):
        return 'RawSegment({}, {}, {})'.format(
            self.road_option.name, self.index_range[1] - self.index_range[0], self.step)

    def __repr__(self):
        return str(self)

    def to_dict(self):
        data_dict = {
            'option': self.road_option.name,
            'indices': self.index_range,
            'has_stop': self.has_stop
        }
        if self.step > 1:
            data_dict['step'] = self.step
        return data_dict

    @staticmethod
    def from_dict(data_dict: dict):
        option = RoadOption.__dict__[data_dict['option']]
        indices = tuple(data_dict['indices'])
        step = data_dict['step'] if 'step' in data_dict else 1
        has_stop = parse_bool(data_dict['has_stop']) if 'has_stop' in data_dict else True
        return RawSegment(option, indices, step, has_stop)


def dict_list_from_raw_segments(raw_segments: List[RawSegment]) -> List[dict]:
    return list(map(lambda s: s.to_dict(), raw_segments))


def raw_segments_from_dict_list(data_list: List[Dict[str, list]]) -> List[RawSegment]:
    return list(map(RawSegment.from_dict, data_list))
