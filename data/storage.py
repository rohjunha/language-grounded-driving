import random
from array import array
from functools import partial
from itertools import product
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

from config import IMAGE_HEIGHT, IMAGE_WIDTH, CAMERA_STEER_OFFSET_DICT
from custom_carla.agents.navigation.local_planner import RoadOption
from data.types import DriveDataFrame
from model import prepare_deeplab_model
from util.common import timethis, get_logger
from util.directory import frame_str, mkdir_if_not_exists
from util.image import put_text, video_from_memory
from util.road_option import fetch_road_option_from_str, fetch_name_from_road_option
from util.serialize import RawSegment

logger = get_logger(__name__)


def encode_integer_list(value: List[int]) -> bytes:
    return array('i', value).tobytes()


def decode_integer_list(byte_str: bytes) -> List[int]:
    if byte_str is None:
        return []
    a = array('i')
    a.frombytes(byte_str)
    return a.tolist()


def decode_image(byte_str: bytes) -> np.ndarray:
    return np.frombuffer(byte_str, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def decode_segment(byte_str: bytes) -> np.ndarray:
    return np.frombuffer(byte_str, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)


def decode_drive(byte_str: bytes) -> DriveDataFrame:
    return DriveDataFrame.load_from_str(byte_str.decode('utf-8'))


def decode_cluster(byte_str: bytes) -> Tuple[float, float]:
    words = byte_str.decode('utf-8').split(',')
    assert len(words) == 2
    return float(words[0]), float(words[1])


class DataFrame:
    def __init__(
            self,
            images: List[np.ndarray] = [],
            drives: List[DriveDataFrame] = [],
            segments: List[np.ndarray] = [],
            custom_segments: List[np.ndarray] = [],
            stop_values: List[float] = []):
        self.option = RoadOption.VOID
        self.sentence = ''
        self.images = images
        self.drives = drives
        self.segments = segments
        self.custom_segments = custom_segments
        self.stop_values = stop_values
        self.action_types = []

    @property
    def valid(self):
        lengths = [len(self.images), len(self.drives),
                   len(self.segments), len(self.custom_segments), len(self.stop_values)]
        return all(list(map(lambda x: x == lengths[0], lengths)))


class DataStorage:
    def __init__(
            self,
            read_only: bool = True,
            db_path: Path = None,
            use_multi_cam: bool = True):
        self.db_path = db_path
        self.num_dbs = 4
        self.env = lmdb.open(
            path=str(self.db_path),
            max_dbs=self.num_dbs,
            map_size=5e11,
            max_readers=1,
            readonly=read_only,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.use_multi_cam: bool = use_multi_cam
        self.image_frames = self.env.open_db('image'.encode())
        self.segment_frames = self.env.open_db('segment'.encode())
        self.custom_segment_frames = self.env.open_db('custom_segment'.encode())
        self.drive_frames = self.env.open_db('drive'.encode())

    def drop_custom_segment_frames(self):
        with self.env.begin(write=True) as txn:
            txn.drop(self.custom_segment_frames)
            print(txn.stat())

    def get_first_image(self):
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=self.image_frames)
            for key, value in image_cursor:
                image = decode_image(value)
                return image

    def get_first_segment(self):
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=self.segment_frames)
            for key, value in image_cursor:
                image = decode_segment(value)
                return image

    def get_frames(self, num_frames: int = -1):
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=self.image_frames)
            keys = []
            for key, _ in image_cursor:
                keys.append(key)
                if 0 < num_frames <= len(keys):
                    break
            return keys

    def put_images_to_queue(self, keys, queue):
        with self.env.begin(db=self.image_frames, write=False) as txn:
            for key in keys:
                queue.put((key, decode_image(txn.get(key.encode()))))
            queue.put(None)


    #
    # def put_images_to_queue(self, indices, camera_keywords, queue_list):
    #     with self.env.begin(db=self.image_frames, write=False) as txn:
    #         for index, keyword in zip(indices, camera_keywords):
    #             decode_image(txn.get(self.key_image(x, keyword)))
    #
    #         return list(map(lambda x: decode_image(txn.get(self.key_image(x, camera_keyword))), frame_numbers))
    #     with self.env.begin(write=False) as txn:
    #         # image_cursor = txn.cursor(db=self.image_frames)
    #         image_cursor.
    #         for key, value in image_cursor:
    #             key = key.decode('utf-8')
    #             image = decode_image(value)
    #             index = int(key[:-1]) % len(queue_list)
    #             queue_list[index].put((key, image))
    #     for i in range(len(queue_list)):
    #         queue_list[i].put(None)

    @property
    def num_drive_frames(self) -> int:
        with self.env.begin(db=self.drive_frames, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_image_frames(self) -> int:
        with self.env.begin(db=self.image_frames, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_segment_frames(self) -> int:
        with self.env.begin(db=self.segment_frames, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_custom_segment_frames(self) -> int:
        with self.env.begin(db=self.custom_segment_frames, write=False) as txn:
            return txn.stat()['entries']

    def key(self, frame_number: int):
        return frame_str(frame_number).encode()

    def key_image(self, frame_number: int, camera_keyword: str):
        if self.use_multi_cam:
            return '{}{}'.format(frame_str(frame_number), camera_keyword[0]).encode()
        else:
            return self.key(frame_number)

    def put_image(self, frame_number: int, camera_keyword: str, image: np.ndarray):
        with self.env.begin(db=self.image_frames, write=True) as txn:
            txn.put(self.key_image(frame_number, camera_keyword), image.tobytes())

    def put_images(self, key_image_list: List[Tuple[str, np.ndarray]]):
        with self.env.begin(db=self.image_frames, write=True) as txn:
            for key, image in key_image_list:
                txn.put(key.encode(), image.tobytes())

    def put_segment_image(self, frame_number: int, camera_keyword: str, image: np.ndarray):
        with self.env.begin(db=self.segment_frames, write=True) as txn:
            txn.put(self.key_image(frame_number, camera_keyword), image.tobytes())

    def put_segment_images(self, key_image_list: List[Tuple[str, np.ndarray]]):
        with self.env.begin(db=self.segment_frames, write=True) as txn:
            for key, image in key_image_list:
                txn.put(key.encode(), image.tobytes())

    def put_custom_segment_image(self, frame_number: int, camera_keyword: str, image: np.ndarray):
        with self.env.begin(db=self.custom_segment_frames, write=True) as txn:
            txn.put(self.key_image(frame_number, camera_keyword), image.tobytes())

    def put_custom_segment_images(self, key_image_list: List[Tuple[str, np.ndarray]]):
        with self.env.begin(db=self.custom_segment_frames, write=True) as txn:
            for key, image in key_image_list:
                txn.put(key.encode(), image.tobytes())

    def put_images_from_paths(self, image_paths: List[Path]):
        self.put_images(list(map(lambda x: (x.stem, cv2.imread(str(x))), image_paths)))

    def put_segment_images_from_paths(self, image_paths: List[Path]):
        def binary_segment_image(image_path: Path) -> Tuple[str, np.ndarray]:
            image = cv2.imread(str(image_path))[:, :, 2]
            return image_path.stem, (image == 7).astype(dtype=np.uint8) * 255
        self.put_segment_images(list(map(binary_segment_image, image_paths)))

    def put_custom_segment_images_from_paths(self, image_paths: List[Path]):
        def binary_segment_image(image_path: Path) -> Tuple[str, np.ndarray]:
            image = cv2.imread(str(image_path))[:, :, 0]
            return image_path.stem, image
        self.put_custom_segment_images(list(map(binary_segment_image, image_paths)))

    def get_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        with self.env.begin(db=self.image_frames, write=False) as txn:
            return decode_image(txn.get(self.key_image(frame_number, camera_keyword)))

    def get_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        with self.env.begin(db=self.image_frames, write=False) as txn:
            return list(map(lambda x: decode_image(txn.get(self.key_image(x, camera_keyword))), frame_numbers))

    def get_segment_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        with self.env.begin(db=self.segment_frames, write=False) as txn:
            return decode_segment(txn.get(self.key_image(frame_number, camera_keyword)))

    def get_segment_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        with self.env.begin(db=self.segment_frames, write=False) as txn:
            return list(map(lambda x: decode_segment(txn.get(self.key_image(x, camera_keyword))), frame_numbers))

    def get_custom_segment_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        with self.env.begin(db=self.custom_segment_frames, write=False) as txn:
            return decode_segment(txn.get(self.key_image(frame_number, camera_keyword)))

    def get_custom_segment_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        with self.env.begin(db=self.custom_segment_frames, write=False) as txn:
            return list(map(lambda x: decode_segment(txn.get(self.key_image(x, camera_keyword))), frame_numbers))

    def put_drive(self, frame_number: int, drive_frame: DriveDataFrame):
        with self.env.begin(db=self.drive_frames, write=True) as txn:
            txn.put(self.key(frame_number), str(drive_frame).encode())

    def put_drives_from_dict(self, data_dict: Dict[str, DriveDataFrame]):
        with self.env.begin(db=self.drive_frames, write=True) as txn:
            for frame_key, drive_frame in data_dict.items():
                txn.put(self.key(int(frame_key)), str(drive_frame).encode())

    def get_drive(self, frame_number: int) -> DriveDataFrame:
        with self.env.begin(db=self.drive_frames, write=False) as txn:
            return decode_drive(txn.get(self.key(frame_number)))

    def get_drives(self, frame_numbers: List[int]) -> List[DriveDataFrame]:
        with self.env.begin(db=self.drive_frames, write=False) as txn:
            return list(map(lambda x: decode_drive(txn.get(self.key(x))), frame_numbers))

    def get_data_frame(
            self,
            drive_frame_numbers: List[int],
            image_frame_numbers: List[int],
            segment_frame_numbers: List[int],
            camera_keyword: str) -> DataFrame:

        def fetch_data(cursor, decode_func, key_func, indices):
            return list(map(lambda x: decode_func(cursor.get(key_func(x))), indices))

        key_func = partial(self.key_image, camera_keyword=camera_keyword)
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=self.image_frames)
            drive_cursor = txn.cursor(db=self.drive_frames)
            segment_cursor = txn.cursor(db=self.segment_frames)
            custom_segment_cursor = txn.cursor(db=self.custom_segment_frames)

            images = fetch_data(image_cursor, decode_image, key_func, image_frame_numbers)
            drives = fetch_data(drive_cursor, decode_drive, self.key, drive_frame_numbers)
            segments = fetch_data(segment_cursor, decode_segment, key_func, segment_frame_numbers)
            custom_segments = fetch_data(custom_segment_cursor, decode_segment, key_func, segment_frame_numbers)
            return DataFrame(images, drives, segments, custom_segments)

    def put_custom_segments_from_images(self, model):
        with self.env.begin(write=True) as txn:
            image_cursor = txn.cursor(db=self.image_frames)
            segment_cursor = txn.cursor(db=self.custom_segment_frames)
            for key, value in tqdm(image_cursor):
                segment_cursor.put(key, model(decode_image(value)).tobytes())

    def put_data_from_dagger(
            self,
            index: int,
            road_option: RoadOption,
            image_dict: Dict[int, np.ndarray],
            drive_dict: Dict[int, DriveDataFrame],
            image_indices: List[int],
            drive_indices: List[int],
            camera_keywords: List[str]):
        new_image_indices = [int('{:04d}{:04d}'.format(index, i % 10000)) for i in image_indices]
        new_images = [image_dict[i] for i in image_indices]
        new_drive_dict = {i: drive_dict[i] for i in drive_indices}

        with self.env.begin(write=True) as txn:
            image_cursor = txn.cursor(db=self.image_frames)
            for image_index, image in zip(new_image_indices, new_images):
                for camera_keyword in camera_keywords:
                    image_cursor.put(self.key_image(image_index, camera_keyword), image.tobytes())
            drive_cursor = txn.cursor(db=self.drive_frames)
            last_option = road_option
            for frame_key, drive_frame in new_drive_dict.items():
                if drive_frame.road_option is not None:
                    last_option = drive_frame.road_option
                else:
                    drive_frame.road_option = last_option
                drive_cursor.put(self.key(int(frame_key)), str(drive_frame).encode())
            traj_cursor = txn.cursor(db=self.trajectories)
            data_dict = {
                'type': 'trajectory',
                'option': road_option.name,
                'image_frames': new_image_indices,
                'drive_frames': drive_indices
            }
            traj_cursor.put(self.key(index), msgpack.packb(data_dict))

    def get_frame_numbers(self):
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor(db=self.segment_frames)
            count = 0
            for key, value in cursor:
                image = decode_segment(value)
                print(key, image.shape)
                if count > 10:
                    break
                count += 1

    def __del__(self):
        self.env.close()

    def copy_drive_frames(self, other):
        with self.env.begin(write=False) as src:
            for key, value in tqdm(src.cursor(db=self.drive_frames)):
                with other.env.begin(db=other.drive_frames, write=True) as dst:
                    dst.put(key, value)

    def copy_image_frames(self, other):
        with self.env.begin(write=False) as src:
            for key, value in tqdm(src.cursor(db=self.image_frames)):
                key_str = key.decode('utf-8')
                index, keyword = int(key_str[:-1]), key_str[-1]
                if index < 50520:
                    continue
                with other.env.begin(db=other.image_frames, write=True) as dst:
                    dst.put(key, value)



    # @property
    # def num_trajectory(self) -> int:
    #     with self.env.begin(db=self.trajectories, write=False) as txn:
    #         return txn.stat()['entries']
    #
    # def get_trajectory(self, index: int):
    #     with self.env.begin(db=self.trajectories, write=False) as txn:
    #         data = txn.get(self.key(index))
    #         data_dict = msgpack.unpackb(data, raw=False)
    #         if data_dict['type'] == 'segment':
    #             return RawSegment.from_dict(data_dict)
    #         else:
    #             data_dict['option'] = RoadOption.__dict__[data_dict['option']]
    #             return data_dict
    #
    # def get_trajectories(self):
    #     return list(map(self.get_trajectory, range(self.num_trajectory)))


class InfoStorage:
    def __init__(
            self,
            ref_data_storage: DataStorage,
            use_split_index: bool,
            read_only: bool = True,
            db_path: Path = None):
        self.ref_data = ref_data_storage
        self.use_multi_cam = self.ref_data.use_multi_cam
        self.use_split_index = use_split_index  # related to have valid stop output and biased sampling

        self.db_path = db_path
        self.num_dbs = 3
        self.env = lmdb.open(
            path=str(self.db_path),
            max_dbs=self.num_dbs,
            map_size=5e11,
            max_readers=1,
            readonly=read_only,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.trajectories = self.env.open_db('trajectory'.encode())
        self.sequences = self.env.open_db('sequence'.encode())
        self.clusters = self.env.open_db('cluster'.encode())

    def drop_trajectories(self):
        with self.env.begin(write=True) as txn:
            txn.drop(self.trajectories)
            print(txn.stat())

    def drop_sequences(self):
        with self.env.begin(write=True) as txn:
            txn.drop(self.sequences)
            print(txn.stat())

    @property
    def num_sequence(self) -> int:
        with self.env.begin(db=self.sequences, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_trajectory(self) -> int:
        with self.env.begin(db=self.trajectories, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_cluster(self) -> int:
        with self.env.begin(db=self.clusters, write=False) as txn:
            return txn.stat()['entries']

    @property
    def num_drive_frames(self) -> int:
        return self.ref_data.num_drive_frames

    @property
    def num_image_frames(self) -> int:
        return self.ref_data.num_image_frames

    def key(self, frame_number: int):
        return frame_str(frame_number).encode()

    def get_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        return self.ref_data.get_image(frame_number, camera_keyword)

    def get_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        return self.ref_data.get_images(frame_numbers, camera_keyword)

    def get_segment_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        return self.ref_data.get_segment_image(frame_number, camera_keyword)

    def get_segment_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        return self.ref_data.get_segment_images(frame_numbers, camera_keyword)

    def get_custom_segment_image(self, frame_number: int, camera_keyword: str) -> np.ndarray:
        return self.ref_data.get_custom_segment_image(frame_number, camera_keyword)

    def get_custom_segment_images(self, frame_numbers: List[int], camera_keyword: str) -> List[np.ndarray]:
        return self.ref_data.get_custom_segment_images(frame_numbers, camera_keyword)

    def get_drive(self, frame_number: int) -> DriveDataFrame:
        return self.ref_data.get_drive(frame_number)

    def get_drives(self, frame_numbers: List[int]) -> List[DriveDataFrame]:
        return self.ref_data.get_drives(frame_numbers)

    def get_data_frame(
            self,
            drive_frame_numbers,
            image_frame_numbers,
            segment_frame_numbers,
            camera_keyword) -> DataFrame:
        return self.ref_data.get_data_frame(
            drive_frame_numbers, image_frame_numbers, segment_frame_numbers, camera_keyword)

    def put_cluster(self, index: int, cluster: Tuple[float, float]) -> None:
        with self.env.begin(db=self.clusters, write=True) as txn:
            txn.put(self.key(index), ','.join(['{:+8.5f}'.format(v) for v in cluster]).encode())

    def put_clusters(self, clusters: List[Tuple[float, float]]) -> None:
        cluster_strs = [','.join(['{:+8.5f}'.format(v) for v in cluster]) for cluster in clusters]
        with self.env.begin(db=self.clusters, write=True) as txn:
            for index, cluster_str in enumerate(cluster_strs):
                txn.put(self.key(index), cluster_str.encode())

    def get_cluster(self, index: int) -> Tuple[float, float]:
        with self.env.begin(db=self.clusters, write=False) as txn:
            return decode_cluster(txn.get(self.key(index)))

    def get_clusters(self) -> List[Tuple[float, float]]:
        with self.env.begin(db=self.clusters, write=False) as txn:
            return list(map(lambda x: decode_cluster(txn.get(self.key(x))), range(self.num_cluster)))

    def has_trajectory(self, index: int) -> bool:
        with self.env.begin(db=self.trajectories, write=False) as txn:
            return txn.get(self.key(index)) is not None

    def put_sequences_from_segment_dict_list(self, segment_dict_list: List[Dict[str, Any]]):
        with self.env.begin(db=self.sequences, write=True) as txn:
            for index, segment_dict in enumerate(segment_dict_list):
                txn.put(self.key(index), msgpack.packb(segment_dict))

    def put_trajectories_from_segment_list(self, segments: List[RawSegment]):
        with self.env.begin(db=self.trajectories, write=True) as txn:
            for index, segment in enumerate(segments):
                data_dict = segment.to_dict()
                data_dict['type'] = 'segment'
                txn.put(self.key(index), msgpack.packb(data_dict))

    @timethis
    def put_trajectories_from_segment_dict_list(self, segment_dict_list: List[Dict[str, Any]]):
        with self.env.begin(db=self.trajectories, write=True) as txn:
            for index, segment_dict in enumerate(segment_dict_list):
                segment_dict['type'] = 'segment'
                txn.put(self.key(index), msgpack.packb(segment_dict))

    def put_trajectory_from_segment(self, index: int, segment: RawSegment):
        with self.env.begin(db=self.trajectories, write=True) as txn:
            data_dict = segment.to_dict()
            data_dict['type'] = 'segment'
            txn.put(self.key(index), msgpack.packb(data_dict))

    def put_trajectory_from_indices(self, index: int, road_option, image_frames, drive_frames):
        with self.env.begin(db=self.trajectories, write=True) as txn:
            data_dict = {
                'type': 'trajectory',
                'option': road_option,
                'image_frames': image_frames,
                'drive_frames': drive_frames
            }
            txn.put(self.key(index), msgpack.packb(data_dict))

    def get_trajectory(self, index: int):
        with self.env.begin(db=self.trajectories, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            if data_dict['type'] == 'segment':
                return RawSegment.from_dict(data_dict)
            else:
                data_dict['option'] = RoadOption.__dict__[data_dict['option']]
                return data_dict

    def get_sequence(self, index: int):
        with self.env.begin(db=self.sequences, write=False) as txn:
            data = txn.get(self.key(index))
            return msgpack.unpackb(data, raw=False)

    def get_sentence_from_sequence(self, index: int) -> str:
        with self.env.begin(db=self.sequences, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            if data_dict is None or 'sentence' not in data_dict:
                return ''
            else:
                return data_dict['sentence']

    def get_road_option_from_trajectory(self, index: int) -> RoadOption:
        with self.env.begin(db=self.trajectories, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            if data_dict is None or 'type' not in data_dict:
                return RoadOption.VOID
            elif data_dict['type'] in ['segment', 'trajectory']:
                return fetch_road_option_from_str(data_dict['option'])
            else:
                raise KeyError('type should be either segment or trajectory: {}'.format(data_dict['type']))

    def randomize_sub_trajectory(
            self,
            indices: List[int],
            split_index: int,
            max_len: int) -> Tuple[List[int], List[float]]:
        stop_values = [0 for _ in indices] if split_index < 0 else list(map(lambda i: float(i >= split_index), indices))
        if max_len <= 0 or max_len > len(indices):
            return indices, stop_values
        # balance between two regions
        if indices[0] < split_index < indices[-1] - 1:
            if random.uniform(0, 1) < 0.9:
                mid_index = max(indices[0], split_index - max_len)
                min_index = max(indices[0], 2 * mid_index - indices[-1] + max_len)
                index1, index2 = min_index - indices[0], max(indices[0], indices[-1] - max_len) - indices[0]
            else:
                index1, index2 = 0, len(indices) - max_len
        else:
            index1, index2 = 0, len(indices) - max_len
        assert index1 <= index2
        index = random.randint(index1, index2)
        return indices[index:index + max_len], stop_values[index:index + max_len]

    def get_stop_probability(self, index: int, max_len: int = -1, camera_keyword: str = 'center') -> float:
        with self.env.begin(db=self.trajectories, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            if data_dict is None or 'type' not in data_dict:
                return -1
            elif data_dict['type'] in ['segment', 'trajectory']:
                option = fetch_road_option_from_str(data_dict['option'])
                split_index = data_dict['split_index'] if ('split_index' in data_dict and self.use_split_index) else -1
                has_stop = False
                if data_dict['type'] == 'segment':
                    segment = RawSegment.from_dict(data_dict)
                    indices = list(segment.range)
                else:
                    indices = list(range(len(data_dict['image_frames'])))
                if not indices:
                    return -1
                stop_values = [0 for _ in indices] if split_index < 0 else list(
                    map(lambda i: int(i >= split_index), indices))
                if max_len < 0:
                    return sum(stop_values) / len(stop_values)
                else:
                    return sum(stop_values[:-max_len]) / len(stop_values[:-max_len])
            else:
                raise KeyError('type should be either segment or trajectory: {}'.format(data_dict['type']))

    def get_data_frame_from_trajectory(
            self,
            index: int,
            max_len: int = -1,
            camera_keyword: str = 'center') -> Tuple[RoadOption, DataFrame]:
        data_frame = DataFrame()
        with self.env.begin(db=self.trajectories, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            if data_dict is None or 'type' not in data_dict:
                return RoadOption.VOID, data_frame
            elif data_dict['type'] in ['segment', 'trajectory']:
                option = fetch_road_option_from_str(data_dict['option'])
                split_index = data_dict['split_index'] if ('split_index' in data_dict and self.use_split_index) else -1
                has_stop = False
                if data_dict['type'] == 'segment':
                    segment = RawSegment.from_dict(data_dict)
                    image_indices, stop_values = self.randomize_sub_trajectory(list(segment.range), split_index, max_len)
                    drive_indices = image_indices
                    segment_indices = image_indices
                    has_stop = segment.has_stop
                else:
                    indices, stop_values = self.randomize_sub_trajectory(list(range(len(data_dict['image_frames']))), split_index, max_len)
                    image_indices = [data_dict['image_frames'][i] for i in indices]
                    drive_indices = [data_dict['drive_frames'][i] for i in indices]
                    segment_indices = [data_dict['segment_frames'][i] for i in indices]
                data_frame = self.get_data_frame(drive_indices, image_indices, segment_indices, camera_keyword)
                if not has_stop:
                    stop_values = [0.0 for _ in data_frame.images]
            else:
                raise KeyError('type should be either segment or trajectory: {}'.format(data_dict['type']))
        steer_offset = CAMERA_STEER_OFFSET_DICT[camera_keyword]
        for i in range(len(data_frame.drives)):
            data_frame.drives[i].control.steer += steer_offset
        data_frame.stop_values = stop_values
        return option, data_frame

    def randomize_sub_sequence(self, indices: List[int]) -> int:
        return indices[random.randint(0, len(indices) - 1)]

    def randomize_data_list(self, data_list, max_len: int):
        if max_len <= 0 or max_len >= len(data_list):
            return data_list
        else:
            i = random.randint(0, len(data_list) - max_len)
            return data_list[i:i + max_len]

    def get_data_from_sequence(self, index: int, max_len: int, camera_keyword: str = 'center') -> Tuple[str, List[dict]]:
        with self.env.begin(db=self.sequences, write=False) as txn:
            data_list = []
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            keyword, sequences = data_dict['sentence'], data_dict['sequence']
            sequences = list(filter(lambda x: x[0] < x[1], sequences))
            for i1, i2, option in sequences:
                lower_option = option.lower()
                if lower_option in ['extraleft', 'extraright']:
                    option = 'STRAIGHT'
                elif lower_option == 'extrastraight':
                    option = 'STOP'
                elif lower_option in ['extrastraightleft', 'extrastraightright']:
                    option = lower_option.replace('extrastraight', '').upper()
                index = self.randomize_sub_sequence(list(range(i1, i2)))
                data_list.append({
                    'image': self.get_image(index, camera_keyword),
                    'drive': self.get_drive(index),
                    'segment': self.get_segment_image(index, camera_keyword),
                    'custom_segment': self.get_custom_segment_image(index, camera_keyword),
                    'action_type': option
                })
                if option == 'STOP':
                    break
            return keyword, self.randomize_data_list(data_list, max_len)

    def get_noisy_data_from_sequence(
            self,
            index: int,
            max_len: int,
            max_sample_per_segment: int,
            camera_keyword: str = 'center'):
        with self.env.begin(db=self.sequences, write=False) as txn:
            data = txn.get(self.key(index))
            data_dict = msgpack.unpackb(data, raw=False)
            keyword, sequences = data_dict['sentence'], data_dict['sequence']
            if any(list(map(lambda x: x[0] >= x[1], sequences))):
                raise ValueError('invalid sequence was given {}'.format(sequences))
            # sequences = list(filter(lambda x: x[0] < x[1], sequences))
            inter_sequences = []
            for i in range(len(sequences) - 1):
                i1 = sequences[i][1] + 1
                i2 = sequences[i + 1][0]
                opt = sequences[i][2]
                inter_sequences.append((i1, i2, opt))

            def convert_element(tuple_element: Tuple[int, int, str], randomized: bool) -> dict:
                return {
                    'index_range': (tuple_element[0], tuple_element[1]),
                    'option': tuple_element[2],
                    'randomized': randomized
                }

            combined_sequences = []
            for i in range(len(inter_sequences)):
                combined_sequences.append(convert_element(sequences[i], False))
                combined_sequences.append(convert_element(inter_sequences[i], True))
            combined_sequences.append(convert_element(sequences[-1], False))

            def convert_option(option: str) -> str:
                lower_option = option.lower()
                if lower_option in ['extraleft', 'extraright']:
                    return 'STRAIGHT'
                elif lower_option == 'extrastraight':
                    return 'STOP'
                elif lower_option in ['extrastraightleft', 'extrastraightright']:
                    return lower_option.replace('extrastraight', '').upper()
                else:
                    return option

            def extract_item_from_index(index: int, camera_keyword: str, option: str) -> dict:
                return {
                    'image': self.get_image(index, camera_keyword),
                    'drive': self.get_drive(index),
                    'segment': self.get_segment_image(index, camera_keyword),
                    'custom_segment': self.get_custom_segment_image(index, camera_keyword),
                    'action_type': option
                }

            data_list = []
            for item in combined_sequences:
                index_range = list(range(*item['index_range']))
                option = convert_option(item['option'])
                randomized = item['randomized']
                extractor = partial(extract_item_from_index, camera_keyword=camera_keyword, option=option)
                if randomized:
                    num_samples = random.randint(0, max_sample_per_segment) if len(index_range) > 5 else 0
                    indices = sorted(set([self.randomize_sub_sequence(index_range) for _ in range(num_samples)]))
                    for index in indices:
                        data_list.append(extractor(index))
                else:
                    index = self.randomize_sub_sequence(index_range)
                    data_list.append(extractor(index))
                if option == 'STOP':
                    break
            return keyword, self.randomize_data_list(data_list, max_len)

    def get_mid_drive_from_sequence(self, index: int) -> List[DriveDataFrame]:
        with self.env.begin(db=self.sequences, write=False) as txn:
            data = txn.get(self.key(index))
            sequences = msgpack.unpackb(data, raw=False)['sequence']
            sequences = list(filter(lambda x: x[0] < x[1], sequences))
            return [self.get_drive((i1 + i2) // 2) for i1, i2, _ in sequences]

    def get_trajectory_data_from_sequence_index(self, sequence_index: int, camera_keyword: str = 'center') -> \
            Tuple[str, DataFrame]:
        with self.env.begin(db=self.sequences, write=False) as txn:
            data = txn.get(self.key(sequence_index))
            data_dict = msgpack.unpackb(data, raw=False)
            sentence, sequences = data_dict['sentence'], data_dict['sequence']
            sequences = list(filter(lambda x: x[0] < x[1], sequences))
            index_range = sequences[0][0], sequences[-2][1]  # generate an index range for the sequence except for stop
            indices = list(range(*index_range))
            data_frame = self.get_data_frame(indices, indices, indices, camera_keyword)
            return sentence, data_frame

    def get_trajectories(self):
        return list(map(self.get_trajectory, range(self.num_trajectory)))

    def get_sequences(self):
        return list(map(self.get_sequence, range(self.num_sequence)))

    def get_video_from_trajectory(self, index: int, camera_keyword: str):
        video_path = mkdir_if_not_exists(self.db_path / 'videos') / 'traj{:04d}.mp4'.format(index)
        option, data_frame = self.get_data_frame_from_trajectory(index, -1, camera_keyword)
        for image, drive, stop in zip(data_frame.images, data_frame.drives, data_frame.stop_values):
            cmd_str = '{:>6s}:{:>6s}:{:+5.2f}:{:+5.3f}'.format(
                fetch_name_from_road_option(option)[:5],
                fetch_name_from_road_option(drive.road_option)[:5],
                drive.control.steer, stop)
            put_text(image, cmd_str, (1, 20), (255, 255, 255), 0.35)
        video_from_memory(data_frame.images, video_path)

    def __del__(self):
        self.env.close()


from multiprocessing import Event, Queue, Process


# def producer(queue_list):
#     data_db_path = Path.home() / '.carla/dataset/data/semantic1'
#     data_storage = DataStorage(read_only=True, db_path=data_db_path, use_multi_cam=True)
#     camera_keywords = ['l', 'c', 'r']
#     frame_numbers = list(range(50520, 202193))
#     keys = ['{:08d}{c}'.format(i, k).encode() for i, k in product(frame_numbers, camera_keywords)]
#     data_storage.put_images_to_queue(queue_list)


def producer(indices, queue):
    data_db_path = Path.home() / '.carla/dataset/data/semantic1'
    data_storage = DataStorage(read_only=True, db_path=data_db_path, use_multi_cam=True)
    camera_keywords = ['l', 'c', 'r']
    keys = ['{:08d}{}'.format(index, keyword) for index, keyword in product(indices, camera_keywords)]
    data_storage.put_images_to_queue(keys, queue)


def consumer(queue, event, index):
    model = prepare_deeplab_model(index)
    out_dir = Path.home() / '.carla/dataset/data/semantic1_images'
    while not event.is_set():
        if not queue.empty():
            item = queue.get_nowait()
            if item is None:
                event.set()
                break
            key, image = item
            print('got {}'.format(key))
            segment = model.run(image)
            print('segmented {}'.format(key))
            cv2.imwrite(str(out_dir / '{}.png'.format(key)), segment)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def server():
    num_proc = 5
    indices = list(range(50520, 202193))
    indices = list(split(indices, num_proc))
    queue_list = []
    for i in range(num_proc):
        queue_list.append(Queue())
    event = Event()
    processes = []
    for i, q in enumerate(queue_list):
        processes.append(Process(target=producer, args=(indices[i], q,)))
        processes.append(Process(target=consumer, args=(q, event, i,)))
    for p in processes:
        p.start()

