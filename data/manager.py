import json
import numpy as np
from argparse import ArgumentParser
from functools import partial
from math import sqrt
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Any, Tuple

from data.segment_extractor import fetch_segment_extractor_by_version
from data.storage import DataStorage, InfoStorage
from data.types import DriveDataFrame, FrameInfo
from util.common import get_logger
from util.directory import ExperimentDirectory, DatasetDirectory, mkdir_if_not_exists
from util.image import video_from_files, video_from_memory
from util.serialize import CustomWaypoint, waypoint_from_str

logger = get_logger(__name__)


def read_meta(meta_path: Path):
    with open(str(meta_path), 'r') as file:
        meta_dict = json.load(file)
    return meta_dict


def read_data(data_path: Path) -> Dict[str, DriveDataFrame]:
    with open(str(data_path), 'r') as file:
        lines = file.read().splitlines()
    data_dict = dict()
    for i, line in enumerate(lines):
        words = line.split(':')  # word, drive_data_frame_str
        if len(words) != 2:
            logger.error(words)
        assert len(words) == 2
        data_dict[words[0]] = DriveDataFrame.load_from_str(words[1])
    return data_dict


def read_waypoint_dict(waypoint_path: Path) -> Dict[int, CustomWaypoint]:
    with open(str(waypoint_path), 'r') as file:
        lines = file.read().splitlines()
    lines = list(filter(lambda x: x, lines))
    waypoints = [waypoint_from_str(l) for l in lines]
    return {w.id: w for w in waypoints}


def read_segment(segment_path: Path):
    with open(str(segment_path), 'r') as file:
        data_dict = json.load(file)
    return data_dict


class DriveDataManager(ExperimentDirectory, DatasetDirectory):
    def __init__(
            self,
            timestamp: int,
            data_name: str,
            version: int,
            noise_injected: bool,
            unique: bool = False,
            augment: bool = True,
            verbose: bool = False,
            overwrite: bool = False):
        self.version = version
        info_name = '{}-v{}'.format(data_name, version)
        ExperimentDirectory.__init__(self, timestamp=timestamp)
        DatasetDirectory.__init__(self, data_name=data_name, info_name=info_name)
        self.verbose = verbose
        self.overwrite = overwrite
        self.noise_injected = noise_injected
        self.unique = unique
        self.augment = augment

        # manually set this variable when storage should be updated
        self.update_data: bool = False
        self.update_info: bool = False

        self.method = None
        self.data: Dict[str, DriveDataFrame] = dict()
        self.waypoint_dict = dict()
        self.image_frames = []
        self.indices: List[int] = []
        self.segment_dict: Dict[str, List[Any]] = dict()
        self.data_storage = DataStorage(read_only=not self.update_data or self.update_segment,
                                        db_path=self.dataset_data_storage_dir)
        self.info_storage = InfoStorage(ref_data_storage=self.data_storage,
                                        use_split_index=True,
                                        read_only=not self.update_info,
                                        db_path=self.dataset_info_storage_dir)

        self.read_data()
        if self.update_data:
            self.update_data_storage()
        if self.update_info or not self.dataset_segment_path.exists():
            infos = self.compute_segments()
            self.update_info_storage()
            self.save_figures(infos)
        else:
            self.read_segments()
        self.save_videos()

    def compute_segments(self):
        self.read_waypoint_dict()
        self.read_image_frames()
        self.check_consistency()
        self.update_waypoint_indices()
        infos = self.extract_info_from_data()
        extractor_cls = fetch_segment_extractor_by_version(self.version)
        extractor = extractor_cls(infos, self.unique, self.augment)
        self.segment_dict = extractor.extract_segments()
        self.save_segments()
        self.save_infos(infos)
        return infos

    def save_infos(self, infos: List[FrameInfo]):
        locations = list(map(lambda x: (x.x, x.y), infos))
        info_path = Path.home() / '.tmp/high-level/info.json'
        with open(str(info_path), 'w') as file:
            json.dump(locations, file, indent=2)

    @property
    def low_level_segments(self):
        if 'low_level_segments' not in self.segment_dict:
            self.segment_dict['low_level_segments'] = []
        return self.segment_dict['low_level_segments']

    @property
    def high_level_segments(self):
        if 'high_level_segments' not in self.segment_dict:
            self.segment_dict['high_level_segments'] = []
        return self.segment_dict['high_level_segments']

    @property
    def clusters(self):
        if 'clusters' not in self.segment_dict:
            self.segment_dict['clusters'] = []
        return self.segment_dict['clusters']

    @low_level_segments.setter
    def low_level_segments(self, values):
        self.segment_dict['low_level_segments'] = values

    @high_level_segments.setter
    def high_level_segments(self, values):
        self.segment_dict['high_level_segments'] = values

    @clusters.setter
    def clusters(self, values):
        self.segment_dict['clusters'] = values

    def read_data(self):
        self.data = read_data(self.experiment_data_path)

    def read_image_frames(self):
        self.image_frames = sorted(list(set(list(map(lambda x: int(x.stem[:-1]),
                                                     self.experiment_image_dir.glob('*.png'))))))

    def read_waypoint_dict(self):
        self.waypoint_dict = read_waypoint_dict(self.experiment_waypoint_path)

    def read_segments(self):
        segment_dict = read_segment(self.dataset_segment_path)
        self.method = segment_dict['meta']['method']
        if 'clusters' in segment_dict['meta']:
            self.clusters = segment_dict['meta']['clusters']
        self.indices = list(range(*segment_dict['meta']['index_range']))
        self.low_level_segments = segment_dict['low_level_segments']
        self.high_level_segments = segment_dict['high_level_segments']

    def fetch_data_frame(self, frame_number: int) -> DriveDataFrame:
        return self.data[self.frame_str(frame_number)]

    def update_waypoint_indices(self):
        # all the waypoint indices are randomly assigned; now make these indices start from 0 in the order of segments
        new_index_dict = dict()
        new_id = 0
        for frame_number in self.indices:
            key = self.frame_str(frame_number)
            old_id = self.fetch_data_frame(frame_number).waypoint_id
            if old_id not in new_index_dict:
                new_index_dict[old_id] = new_id
                new_id += 1
            self.data[key].waypoint_id = new_index_dict[old_id]

        new_waypoint_dict = dict()
        for old_id, new_id in new_index_dict.items():
            new_waypoint_dict[new_id] = self.waypoint_dict[old_id]
        self.waypoint_dict = new_waypoint_dict

        # delete data frames out of indices
        for i in filter(lambda x: x < self.indices[0] or x > self.indices[-1], [int(k) for k in self.data.keys()]):
            del self.data[self.frame_str(i)]
        for valid_key in list(filter(lambda key: self.data[key].waypoint_id < 0, self.data.keys())):
            del self.data[valid_key]

    def check_consistency(self):
        index_set1 = set([int(k) for k in self.data.keys()])
        index_set2 = set(self.image_frames)
        index_set = index_set1.intersection(index_set2)
        self.indices = sorted(list(index_set))

        consecutive_indices = []
        i = 0
        while i < len(self.indices) - 1:
            i1, i2 = i, i
            for j in range(i, len(self.indices) - 1):
                if self.indices[j] + 1 == self.indices[j+1]:
                    i2 = j + 1
                else:
                    break
            consecutive_indices.append((i1, i2 + 1))
            i = j + 1
        assert consecutive_indices
        logger.info('consecutive indices were computed {}'.format(consecutive_indices))
        best_index = sorted(enumerate(consecutive_indices), key=lambda x: x[1][1] - x[1][0], reverse=True)[0][0]
        best_islice = consecutive_indices[best_index]
        best_indices = self.indices[best_islice[0]:best_islice[1]]
        best_index_set = set(best_indices)
        invalid_indices = list(filter(lambda i: i not in best_index_set, self.indices))
        self.indices = best_indices
        for i in invalid_indices:
            del self.data[self.frame_str(i)]

        for i, j in zip(self.indices[:-1], self.indices[1:]):
            if i + 1 != j:
                di, dj = self.fetch_data_frame(i), self.fetch_data_frame(j)
                li, lj = di.state.transform.location, dj.state.transform.location
                dist = sqrt((li.x - lj.x) ** 2 + (li.y - lj.y) ** 2)
                if dist < 5:
                    raise ValueError('non-consecutive frames were detected {}, {}'.format(i, j))

    def extract_info_from_data(self) -> List[FrameInfo]:
        keys = sorted([key for key in self.data.keys()])
        keys = list(filter(lambda key: self.data[key].waypoint_id in self.waypoint_dict, keys))
        xs = list(map(lambda key: self.data[key].state.transform.location.x, keys))
        ys = list(map(lambda key: self.data[key].state.transform.location.y, keys))
        ts = list(map(lambda key: self.data[key].state.transform.rotation.yaw, keys))
        iis = list(map(lambda key: self.waypoint_dict[self.data[key].waypoint_id].is_intersection, keys))
        ros = list(map(lambda key: self.data[key].road_option.name, keys))
        pos = list(map(lambda key: [opt.name for opt in self.data[key].possible_road_options], keys))
        return list(map(lambda x: FrameInfo(*x), zip(keys, xs, ys, ts, iis, ros, pos)))

    def save_segments(self):
        if not self.low_level_segments:
            raise ValueError('segment was not created properly')
        segment_info = {
            'meta': {
                'index_range': (self.low_level_segments[0]['indices'][0], self.low_level_segments[-1]['indices'][1]),
                'clusters': self.clusters,
                'method': 'offline'
            },
            'low_level_segments': self.low_level_segments,
            'high_level_segments': self.high_level_segments}

        with open(str(self.dataset_segment_path), 'w') as file:
            json.dump(segment_info, file, indent=2)

    def save_figures(self, infos: List[FrameInfo]):
        if not self.verbose:
            return

        import matplotlib.pyplot as plt
        info_dict = self.segment_dict['info_dict']

        def fetch_color_code(intersection: bool, option: str) -> str:
            option_code = {
                'left': 'g',
                'right': 'r',
                'straight': 'b',
                'extraleft': 'b',
                'extraright': 'b',
                'extrastraight': 'b',
                'extrastraightleft': 'g',
                'extrastraightright': 'r',
                'lanefollow': 'k',
                'stop': 'y'}
            inter_code = {True: 'o', False: '-'}
            return '{}{}'.format(option_code[option], inter_code[intersection])

        def draw_segment_(intersection: bool, option: str, index_range: Tuple[int, int]):
            indices = [info_dict[i] for i in range(index_range[0], index_range[1], 5)]
            xs, ys = zip(*[(infos[i].x, -infos[i].y) for i in indices])
            cc = fetch_color_code(intersection, option.lower())
            plt.plot(xs, ys, cc, label='{}:{}'.format(option, str(intersection)))

        def draw_text(text: str, index_range: Tuple[int, int]):
            i1, i2 = info_dict[index_range[0]], info_dict[index_range[1]]
            index = i1 + (i2 - i1) // 2
            plt.text(infos[index].x + 2, -infos[index].y, text)

        def draw_segment(segment: Dict[str, Any]):
            option, index_range = segment['option'], segment['indices']
            intersection = False if option == 'LANEFOLLOW' else True
            draw_segment_(intersection, option, index_range)

        def draw_high_level_segment(segment: Dict[str, Any]):
            sentence, sequence = segment['sentence'], segment['sequence']
            sequence = list(filter(lambda x: x[0] < x[1], sequence))
            for i1, i2, opt in sequence:
                intersection = False if opt == 'LANEFOLLOW' else True
                draw_segment_(intersection, opt, (i1, i2))

        ixs, iys = list(map(lambda x: x.x, infos))[:10000], list(map(lambda x: -x.y, infos))[:10000]

        logger.info('saving {} high-level segments'.format(len(self.high_level_segments)))

        if not self.unique:
            for i, segment in enumerate(self.high_level_segments):
                # logger.info('drawing {}, {}'.format(i, segment))
                try:
                    plt.plot(ixs, iys, 'C1-')
                    draw_high_level_segment(segment)
                    draw_text(str(i), segment['sequence'][0][:2])
                    plt.legend()
                    plt.axes().set_aspect('equal', 'datalim')
                    plt.savefig(str(self.dataset_high_level_figure_path(i)))
                finally:
                    plt.cla()
        for i, segment in enumerate(self.low_level_segments):
            plt.plot(ixs, iys, 'C1-')
            draw_segment(segment)
            draw_text(str(i), segment['indices'])
            plt.legend()
            plt.axes().set_aspect('equal', 'datalim')
            plt.savefig(str(self.dataset_low_level_figure_path(i)))
            plt.cla()

    def save_videos(self):
        option_name_set = set(map(lambda x: x['option'], self.low_level_segments))
        for option_name in option_name_set:
            mkdir_if_not_exists(self.dataset_video_dir / option_name.lower())
        clusters = self.info_storage.get_clusters()

        def generate_single_video(isegment):
            index, segment_dict = isegment
            option_name = segment_dict['option'].lower()
            si = segment_dict['split_index']
            i1, i2 = segment_dict['indices']
            if i1 >= i2:
                logger.error('invalid frame {}, {}, {}'.format(index, option_name, (i1, i2)))
            labels = [0 for _ in range(i1, i2)] if si < 0 else list(map(lambda i: int(i >= si), range(i1, i2)))

            # find closest cluster
            drive_frames: List[DriveDataFrame] = self.data_storage.get_drives(list(range(i1, i2)))
            custom_segment_frames: List[np.ndarray] = self.data_storage.get_custom_segment_images(
                list(range(i1, i2)), 'center')
            query_frame = drive_frames[-1] if option_name == 'lanefollow' else drive_frames[len(drive_frames) // 2]

            def compute_dist(ci, query_frame) -> float:
                x1, y1 = clusters[ci]
                l = query_frame.state.transform.location
                x2, y2 = l.x, l.y
                return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            min_dist, min_cluster_index = 1e10, -1
            # logger.info(len(clusters))
            for ci, _ in enumerate(clusters):
                di = compute_dist(ci, query_frame)
                # logger.info((ci, di))
                if di < min_dist:
                    min_dist = di
                    min_cluster_index = ci
            assert min_cluster_index >= 0
            compute_cluster_dist = partial(compute_dist, min_cluster_index)
            dists = list(map(compute_cluster_dist, drive_frames))
            texts = ['label {}, dist {:+5.3f}'.format(l, d) for d, l in zip(dists, labels)]

            video_path = self.dataset_video_dir / option_name / 'video{:05d}.mp4'.format(index)
            segment_video_path = self.dataset_video_dir / option_name / 'custom{:05d}.mp4'.format(index)
            if not self.overwrite and video_path.exists():
                return
            image_paths = [self.image_path(f) for f in range(i1, i2)]
            video_from_files(image_paths, video_path, texts, framerate=30)
            video_from_memory(custom_segment_frames, segment_video_path, texts, framerate=30)

        pool = ThreadPool(8)
        pool.map(generate_single_video, enumerate(self.low_level_segments))

    def update_data_storage(self):
        self.data_storage.put_images_from_paths(sorted(self.experiment_image_dir.glob('*.png')))
        self.data_storage.put_drives_from_dict(self.data)
        self.data_storage.put_segment_images_from_paths(sorted(self.experiment_segment_dir.glob('*.png')))

    def update_info_storage(self):
        self.info_storage.put_trajectories_from_segment_dict_list(self.low_level_segments)
        self.info_storage.put_sequences_from_segment_dict_list(self.high_level_segments)
        self.info_storage.put_clusters(self.clusters)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('timestamp', type=int)
    arg_parser.add_argument('data_name', type=str)
    arg_parser.add_argument('version', type=int)
    arg_parser.add_argument('--unique', action='store_true')
    arg_parser.add_argument('--augment', action='store_true')
    arg_parser.add_argument('--noise-injected', action='store_true')
    arg_parser.add_argument('--verbose', action='store_true')
    arg_parser.add_argument('--overwrite', action='store_true')
    args = arg_parser.parse_args()
    DriveDataManager(
        timestamp=args.timestamp,
        data_name=args.data_name,
        version=args.version,
        noise_injected=args.noise_injected,
        unique=args.unique,
        augment=args.augment,
        verbose=args.verbose,
        overwrite=args.overwrite)


if __name__ == '__main__':
    main()
