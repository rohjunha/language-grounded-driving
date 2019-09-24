import re
from functools import wraps
from pathlib import Path
from typing import Set

from util.common import get_logger

logger = get_logger(__name__)


def mkdir_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
    if not path.exists():
        raise FileNotFoundError('the directory was not found: {}'.format(path))
    return path


def safe(func):
    @wraps(func)
    def wrapper_safe_fetch(*args, **kwargs):
        value = None
        error = 'failed to fetch the directory with {}, {}'.format(func.__name__, args)
        for _ in range(5):
            try:
                value = func(*args, **kwargs)
                if value is not None:
                    break
            except:
                logger.error(error)
        if value is None:
            raise FileNotFoundError(error)
        return value
    return wrapper_safe_fetch


def fetch_exp_dir_str(exp_index: int):
    return 'exp{:02d}'.format(exp_index)


@safe
def fetch_home_dir():
    return Path.cwd().absolute()


@safe
def fetch_data_dir():
    return mkdir_if_not_exists(fetch_home_dir() / '.carla')


@safe
def fetch_raw_data_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'rawdata')


@safe
def fetch_dataset_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'dataset')


@safe
def fetch_evaluation_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'evaluations')


@safe
def fetch_evaluation_summary_dir():
    return mkdir_if_not_exists(fetch_evaluation_dir() / 'summary')


@safe
def fetch_settings_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'settings')


@safe
def fetch_checkpoint_root_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'checkpoints')


def fetch_checkpoint_subdir(exp_index: int, exp_name: str) -> str:
    return 'exp{:02d}/{}'.format(exp_index, exp_name)


@safe
def fetch_checkpoint_dir(exp_index: int, exp_name: str):
    return mkdir_if_not_exists(fetch_checkpoint_root_dir() / fetch_checkpoint_subdir(exp_index, exp_name))


def fetch_checkpoint_path(exp_index: int, exp_name: str, exp_step: int):
    return fetch_checkpoint_dir(exp_index, exp_name) / 'step{:06d}.pth'.format(exp_step)


def fetch_checkpoint_meta_path(exp_index: int, exp_name: str, exp_step: int):
    return fetch_checkpoint_dir(exp_index, exp_name) / 'step{:06d}.json'.format(exp_step)


@safe
def fetch_evaluation_setting_dir():
    return mkdir_if_not_exists(fetch_settings_dir() / 'evaluation')


@safe
def fetch_word_embeddings_dir():
    return mkdir_if_not_exists(fetch_settings_dir() / 'word-embeddings')


@safe
def fetch_param_dir():
    return mkdir_if_not_exists(fetch_data_dir() / 'params')


@safe
def fetch_param_path(exp_index: int, exp_name: str):
    return mkdir_if_not_exists(fetch_param_dir() / fetch_exp_dir_str(exp_index)) / '{}.json'.format(exp_name)


def fetch_carla_binary_path():
    return Path.home() / 'projects/carla095/CarlaUE4.sh'


def frame_format() -> str:
    return '{:08d}'


def frame_str(frame_number: int) -> str:
    return frame_format().format(frame_number)


class ExperimentDirectory:
    def __init__(self, timestamp: int):
        self.experiment_parent = fetch_raw_data_dir()
        self.timestamp = '{:016d}'.format(timestamp)
        self.experiment_root_dir = mkdir_if_not_exists(self.experiment_parent / self.timestamp)
        self.experiment_meta_path = self.experiment_root_dir / 'meta.json'
        self.experiment_data_path = self.experiment_root_dir / 'data.txt'
        self.experiment_waypoint_path = self.experiment_root_dir / 'waypoint.txt'
        self.segment_path = self.experiment_root_dir / 'segment.json'
        self.experiment_image_dir = mkdir_if_not_exists(self.experiment_root_dir / 'images')
        self.experiment_segment_dir = mkdir_if_not_exists(self.experiment_root_dir / 'segments')

    def frame_str(self, frame: int):
        return frame_str(frame)

    def image_path(self, frame_number: int, camera_keyword: str = 'center'):
        return self.experiment_image_dir / '{}{}.png'.format(self.frame_str(frame_number), camera_keyword[0])

    def segment_image_path(self, frame_number: int, camera_keyword: str = 'center'):
        return self.experiment_segment_dir / '{}{}.png'.format(self.frame_str(frame_number), camera_keyword[0])


class DatasetDirectory:
    def __init__(self, data_name: str, info_name: str):
        self.dataset_parent = fetch_dataset_dir()
        self.data_name = data_name
        self.info_name = info_name
        self.dataset_data_storage_dir = mkdir_if_not_exists(self.dataset_parent / 'data' / self.data_name)
        self.dataset_info_storage_dir = mkdir_if_not_exists(self.dataset_parent / 'info' / self.info_name)
        self.dataset_segment_path = self.dataset_info_storage_dir / 'segment.json'
        self.dataset_low_level_figure_dir = mkdir_if_not_exists(self.dataset_info_storage_dir / 'low_level_figures')
        self.dataset_high_level_figure_dir = mkdir_if_not_exists(self.dataset_info_storage_dir / 'high_level_figures')
        self.dataset_video_dir = mkdir_if_not_exists(self.dataset_info_storage_dir / 'videos')

    def dataset_low_level_figure_path(self, index: int) -> Path:
        return self.dataset_low_level_figure_dir / 'segment{:05d}.png'.format(index)

    def dataset_high_level_figure_path(self, index: int) -> Path:
        return self.dataset_high_level_figure_dir / 'segment{:05d}.png'.format(index)

    def dataset_video_path(self, index: int) -> Path:
        return self.dataset_video_dir / 'segment{:05d}.mp4'.format(index)


class EvaluationDirectory:
    def __init__(self, exp_index: int, exp_name: str, exp_step: int, data_keyword: str):
        logger.info((exp_index, exp_name, exp_step, data_keyword))
        self.exp_index = exp_index
        self.exp_name = exp_name
        self.exp_step = exp_step
        self.data_keyword = data_keyword
        self.online = self.data_keyword == 'online'

        self.parent = fetch_evaluation_dir()
        self.exp_name_subdir = 'exp{:02d}/{}'.format(exp_index, exp_name)
        self.exp_name_dir = mkdir_if_not_exists(self.parent / self.exp_name_subdir)
        self.step_str = 'step{:06d}'.format(exp_step)
        assert exp_step > 0
        self.step_dir = mkdir_if_not_exists(self.exp_name_dir / self.step_str)
        assert data_keyword
        self.root_dir = mkdir_if_not_exists(self.step_dir / data_keyword)
        self.root_subdir = '{}/{}/{}'.format(self.exp_name_subdir, self.step_str, data_keyword)

        self.image_dir = mkdir_if_not_exists(self.root_dir / 'images')
        self.segment_dir = mkdir_if_not_exists(self.root_dir / 'segments')
        self.video_dir = mkdir_if_not_exists(self.root_dir / 'videos')
        self.state_dir = mkdir_if_not_exists(self.root_dir / 'states')
        self.summary_dir = mkdir_if_not_exists(self.root_dir / 'summary')
        self.summary_path = self.summary_dir / 'summary.json'

    def video_path(self, traj_index: int, camera_keyword: str = 'center') -> Path:
        return self.video_dir / 'traj{:02d}{}.mp4'.format(traj_index, camera_keyword[0])

    def state_path(self, traj_index: int) -> Path:
        return self.state_dir / 'traj{:02d}.json'.format(traj_index)

    def traj_indices_from_state_dir(self) -> Set[int]:
        eval_state_files = sorted(self.state_dir.glob('*.json'))
        video_files = sorted(self.video_dir.glob('traj*.mp4'))
        state_indices = [int(re.findall('traj([\w]+)', f.stem)[0]) for f in eval_state_files]
        video_indices = [int(re.findall('traj([\w]+)', f.stem[:-1])[0]) for f in video_files]
        for i, s in enumerate(state_indices):
            if s not in video_indices:
                eval_state_files[i].unlink()
                logger.info('incomplete {} was removed'.format(eval_state_files[i]))
        return set(video_indices)
