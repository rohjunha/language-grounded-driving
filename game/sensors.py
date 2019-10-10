import math
import weakref
from collections import defaultdict
from functools import partial

from config import CAMERA_KEYWORDS, DATASET_FRAMERATE, EVAL_FRAMERATE_SCALE
from game.common import destroy_actor
from util.common import add_carla_module, get_logger

add_carla_module()
logger = get_logger(__name__)
import carla


CAMERA_SHIFT = 0.4
__camera_transforms__ = {
    'center': carla.Transform(
        carla.Location(x=1.6, z=1.7),
        carla.Rotation(yaw=0)),
    'left': carla.Transform(
        carla.Location(x=1.6, y=-CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(-CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'right': carla.Transform(
        carla.Location(x=1.6, y=+CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'extra': carla.Transform(
        carla.Location(x=1.6, z=1.7),
        carla.Rotation(yaw=0)),
}
for key in __camera_transforms__.keys():
    assert key in CAMERA_KEYWORDS
for key in CAMERA_KEYWORDS:
    assert key in __camera_transforms__


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
        # weak_self = weakref.ref(self)
        # self.sensor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

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
        # weak_self = weakref.ref(self)
        # self.sensor.listen(lambda image: CameraSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    # @staticmethod
    # def on_listen(weak_self, carla_image: carla.Image):
    #     self = weak_self()
    #     if not self:
    #         return
    #     frame_number = carla_image.frame_number
    #     self.timing_dict[frame_number] = get_current_time()
    #     self.transform_dict[frame_number] = self.sensor.get_transform()
    #     self.image_frame_number = frame_number
    #     numpy_image = numpy_from_carla_image(carla_image)
    #     self.image_frame = numpy_image
    #     # print(frame_number, self.image_frame.shape, self.image_path_func(frame_number))
    #     cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)


class SegmentationSensor(SensorBase):
    def __init__(self, parent_actor, image_path_func, width, height, camera_keyword: str):
        self.width = width
        self.height = height
        self.camera_keyword = camera_keyword
        SensorBase.__init__(self, parent_actor)
        self.image_path_func = image_path_func
        self.image_frame_number = None
        self.image_frame = None
        # weak_self = weakref.ref(self)
        # self.sensor.listen(lambda image: SegmentationSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    # @staticmethod
    # def on_listen(weak_self, carla_image: carla.Image):
    #     self = weak_self()
    #     if not self:
    #         return
    #     frame_number = carla_image.frame_number
    #     self.image_frame_number = frame_number
    #     numpy_image = numpy_from_carla_image(carla_image)
    #     self.image_frame = numpy_image
    #     cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)


def generate_camera_sensor(world, agent, camera_keyword, width, height):
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
    bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
    return world.spawn_actor(bp, __camera_transforms__[camera_keyword], attach_to=agent)


def generate_segmentation_sensor(world, agent, camera_keyword, width, height):
    bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
    bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
    return world.spawn_actor(bp, __camera_transforms__[camera_keyword], attach_to=agent)


def generate_collision_sensor(world, agent):
    bp = world.get_blueprint_library().find('sensor.other.collision')
    return world.spawn_actor(bp, carla.Transform(), attach_to=agent)


def set_all_sensors(world, agent, camera_keywords, width, height, use_extra):
    camera_sensor_dict = dict()
    segmentation_sensor_dict = dict()
    if use_extra:
        camera_sensor_dict['extra'] = generate_camera_sensor(world, agent, 'extra', 1280, 720)
    for keyword in camera_keywords:
        camera_sensor_dict[keyword] = generate_camera_sensor(world, agent, keyword, width, height)
        segmentation_sensor_dict[keyword] = generate_segmentation_sensor(world, agent, keyword, width, height)
    collision_sensor = generate_collision_sensor(world, agent)
    return camera_sensor_dict, segmentation_sensor_dict, collision_sensor
