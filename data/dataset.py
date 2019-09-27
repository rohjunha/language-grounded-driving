import collections
import json
import random
from functools import partial
from itertools import chain, product
from operator import attrgetter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch.utils
from msgpack import packb, unpackb
from torch._six import int_classes, string_classes
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
# from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as OriginalDataLoaderIter
from torch.utils.data.dataloader import _DataLoaderIter as OriginalDataLoaderIter
from torchvision.transforms import RandomApply, ColorJitter

from config import CAMERA_KEYWORDS
from custom_carla.agents.navigation.local_planner import RoadOption
from data.storage import DataStorage, InfoStorage, DataFrame
from data.types import CarState, CarControl, DriveDataFrame
from parameter import Parameter
from util.common import get_logger
from util.directory import fetch_dataset_dir, fetch_word_embeddings_dir
from util.image import tensor_from_numpy_image
from util.road_option import fetch_name_from_road_option, fetch_road_option_from_str, fetch_onehot_vector_dim, \
    fetch_onehot_vector_from_road_option, fetch_index_from_road_option, \
    fetch_onehot_index_from_high_level_str, fetch_nmc_onehot_vector_from_road_option, fetch_nmc_index_from_road_option

logger = get_logger(__name__)


def action_from_control(controls: list) -> List[List[float]]:
    return [[c.throttle, c.steer] for c in controls]


class DatasetBase(torch.utils.data.Dataset):
    def __init__(
            self,
            param: Parameter,
            dataset_names: List[str],
            dataset_dict: Dict[str, InfoStorage],
            indices: List[Tuple[str, int]],
            train: bool):
        torch.utils.data.Dataset.__init__(self)
        self.device_type = param.device_type

        self.train = train
        self.dataset_names = dataset_names
        self.dataset_dict = dataset_dict
        self.dataset_probs = param.dataset_data_probs
        self.index_from_dataset_name = {n: i for i, n in enumerate(self.dataset_names)}
        self.indices = indices

        self.use_color_jitter = param.use_color_jitter
        self.max_data_length = param.max_data_length
        self.use_multi_cam = param.use_multi_cam
        self.image_type = param.image_type

    def finalize_images(self, data_frame: DataFrame) -> List[np.ndarray]:
        if self.image_type == 'bgr':
            return data_frame.images
        elif self.image_type == 'd':
            return data_frame.custom_segments
        elif self.image_type == 's':
            return data_frame.segments
        elif self.image_type == 'bgrd':
            return [np.concatenate((i, s), axis=-1) for i, s in zip(data_frame.images, data_frame.custom_segments)]
        elif self.image_type == 'bgrs':
            return [np.concatenate((i, s), axis=-1) for i, s in zip(data_frame.images, data_frame.segments)]
        elif self.image_type == 'ds':
            return [np.concatenate((i, s), axis=-1) for i, s in zip(data_frame.custom_segments, data_frame.segments)]
        else:
            raise TypeError('invalid image type {}'.format(self.image_type))

    def __len__(self):
        return len(self.indices)

    def _get_index_pair(self, global_index: int):
        assert 0 <= global_index < len(self)
        return self.indices[global_index]

    def to_batch_items(self, data_dict):
        raise NotImplementedError

    def __getitem__(self, global_index) -> dict:
        raise NotImplementedError

    def copy_to_device(self, item: dict) -> dict:
        raise NotImplementedError

    def sample_weight_list(self, custom_weights: List[float], custom_stop_counter: Dict[bool, int]) -> List[float]:
        raise NotImplementedError


class LowLevelDataset(DatasetBase):
    def __init__(
            self,
            param: Parameter,
            dataset_names: List[str],
            dataset_dict: Dict[str, InfoStorage],
            indices: List[Tuple[str, int]],
            train: bool):
        DatasetBase.__init__(self, param, dataset_names, dataset_dict, indices, train)
        self.single_label = param.single_label
        self.use_stop = param.model_level == 'low' and not param.is_control
        if self.use_stop:
            self.stop_probabilities = [self.get_stop_probability(i, self.max_data_length) for i in range(len(indices))]

        self.use_low_level_segment = param.use_low_level_segment
        self.model_level = param.model_level
        self.output_dim = 1 if self.single_label else fetch_onehot_vector_dim(self.use_low_level_segment)
        self.onehot_func = partial(fetch_onehot_vector_from_road_option, low_level=self.use_low_level_segment)
        self.index_func = partial(fetch_index_from_road_option, low_level=self.use_low_level_segment)

        self.balance_label = param.balance_label if train else False
        self.balance_counts = [0 for _ in range(self.output_dim)]
        self.balance_weights = [1.0 for _ in range(self.output_dim)]
        if not param.grouped_batch and (self.balance_label or self.single_label):
            self.balance_index()
        logger.info('loaded {} trajectories'.format(len(self)))

    def get_clusters(self, name: str) -> List[Tuple[float, float]]:
        return self.dataset_dict[name].get_clusters()

    def get_trajectory_data(self, global_index: int, max_len: int = -1, camera_keyword: str = 'center') -> \
            Tuple[RoadOption, DataFrame]:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_dict[name].get_data_frame_from_trajectory(local_index, max_len, camera_keyword)

    def get_stop_probability(self, global_index: int, max_len: int = -1, camera_keyword: str = 'ceneter') -> float:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_dict[name].get_stop_probability(local_index, max_len, camera_keyword)

    def get_road_option(self, global_index: int) -> RoadOption:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_dict[name].get_road_option_from_trajectory(local_index)

    def get_probability_on_dataset(self, global_index: int) -> float:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_probs[self.index_from_dataset_name[name]]

    def balance_index(self):
        if self.single_label:
            single_road_option = fetch_road_option_from_str(self.single_label)
            self.balance_counts[self.index_func(single_road_option)] = 1
        else:
            for i in range(len(self)):
                action_type = self.get_road_option(i)
                self.balance_counts[self.index_func(action_type)] += 1
        all_counts = sum(self.balance_counts)
        for i in range(len(self.balance_counts)):
            self.balance_weights[i] = all_counts / self.balance_counts[i]

    def sample_weight_list(
            self,
            custom_weights: List[float] = list(),
            custom_stop_counter: Dict[bool, int] = dict()) -> List[float]:
        use_stop_counter = list(custom_stop_counter.keys())
        if use_stop_counter:
            stop_count_all = custom_stop_counter[True] + custom_stop_counter[False]
            stop_weight = custom_stop_counter[False] / stop_count_all
            non_stop_weight = custom_stop_counter[True] / stop_count_all
            # logger.info('custom weights for stop was used {:+5.3f}, {:+5.3f}'.format(stop_weight, non_stop_weight))

        weight_list = []
        for i in range(len(self)):
            action_type = self.get_road_option(i)
            weight_dataset = self.get_probability_on_dataset(i)
            index = self.index_func(action_type)
            weight = self.balance_weights[index] * weight_dataset
            if use_stop_counter and self.use_stop:
                raw_stop_prob = self.stop_probabilities[i]
                stop_weight = raw_stop_prob * stop_weight + (1 - raw_stop_prob) * non_stop_weight
                weight *= stop_weight
            if custom_weights:
                weight *= custom_weights[i]
            weight_list.append(weight)
        return weight_list

    def to_batch_items(self, data_dict):
        data_dict['images'] = torch.stack(
            [tensor_from_numpy_image(i, self.use_color_jitter) for i in data_dict['images']])  # S x C x H x W
        if data_dict['images'].dim() == 3:
            data_dict['images'].unsqueeze_(1)
        data_dict['actions'] = torch.tensor(data_dict['actions'], dtype=torch.float32)
        data_dict['type'] = fetch_name_from_road_option(data_dict['type'])
        data_dict['controls'] = packb(([str(c) for c in data_dict['controls']]))
        data_dict['states'] = packb(([str(s) for s in data_dict['states']]))
        data_dict['stops'] = torch.tensor(data_dict['stops'], dtype=torch.float32).view(data_dict['actions'].size(0), 1)
        return data_dict

    def __getitem__(self, global_index) -> dict:
        camera_keyword = random.choice(CAMERA_KEYWORDS)
        option, data_frame = self.get_trajectory_data(global_index, self.max_data_length, camera_keyword)
        states = list(map(attrgetter('state'), data_frame.drives))
        controls = list(map(attrgetter('control'), data_frame.drives))
        actions = action_from_control(controls)
        data_dict = {
            'index': global_index,
            'type': option,
            'action_index': self.index_func(option),
            'onehot': self.onehot_func(option),
            'images': self.finalize_images(data_frame),
            'controls': controls,
            'states': states,
            'actions': actions,
            'stops': data_frame.stop_values
        }
        return self.to_batch_items(data_dict)

    @staticmethod
    def from_batch_items(data_dict: dict):
        data_dict['type'] = [fetch_road_option_from_str(t) for t in data_dict['type']]
        data_dict['controls'] = [[CarControl.load_from_str(s) for s in unpackb(ss, raw=False)] for ss in
                                 data_dict['controls']]
        data_dict['states'] = [[CarState.load_from_str(s) for s in unpackb(ss, raw=False)] for ss in
                               data_dict['states']]
        return data_dict

    def copy_to_device(self, item: dict):
        assert 'states' in item
        assert 'actions' in item
        assert 'images' in item

        item = LowLevelDataset.from_batch_items(item)  # unpack data

        # if the image is once packed, sort other data according to the indices
        tuple_keys = list(filter(lambda x: isinstance(item[x], tuple), item.keys()))
        if tuple_keys:
            indices = None
            for key in item.keys():
                if key in tuple_keys:
                    values, indices = item[key]
                    item[key] = values
            if indices is None:
                raise ValueError('tuple is not generated from the collate function: {}'.format(key))

            list_keys = list(filter(lambda x: isinstance(item[x], list), item.keys()))
            for key in list_keys:
                item[key] = [item[key][i] for i in indices]

            lens = [len(v) for v in item['states']]
            mask = torch.ones_like(item['actions'])
            for b in range(mask.size(0)):
                mask[b, lens[b]:, :] = 0.0
            item['mask'] = mask

        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(self.device_type)
        # for key, value in item.items():
        #     if isinstance(value, torch.Tensor):
        #         print(key, value.shape, value.device)

        return item


def load_index_from_word() -> Dict[str, int]:
    word_list_filepath = fetch_word_embeddings_dir() / 'words.txt'
    with open(str(word_list_filepath), 'r') as file:
        word_list = file.read().split(',')
    assert len(word_list) == 100
    return {w: i for i, w in enumerate(word_list)}


def generate_sentence_list_from_templates(
        expressions: List[str],
        prefixes: List[str],
        suffixes: List[str],
        keyword: str = ''):
    sentences = []
    if keyword:
        expressions = [expression.format(keyword=keyword) for expression in expressions]
    if prefixes:
        for prefix, suffix, expression in product(prefixes, suffixes, expressions):
            prefix_words = set(prefix.split(' '))
            suffix_words = set(suffix.split(' '))
            valid = all(list(map(lambda w: w not in prefix_words and w not in suffix_words, expression.split(' '))))
            if valid:
                sentences.append(' '.join([prefix, expression, suffix]).strip())
    else:
        for suffix, expression in product(suffixes, expressions):
            suffix_words = set(suffix.split(' '))
            valid = all(list(map(lambda w: w not in suffix_words, expression.split(' '))))
            if valid:
                sentences.append(' '.join([expression, suffix]).strip())
    sentences = list(map(lambda x: x.replace("'", ' '), sentences))
    return sorted(sentences)


STRAIGHT_PREFIXES = ['', 'you can', "you're going to", "you'll", 'you want to', 'continue to', 'just']
STRAIGHT_SUFFIXES = ['', 'further', 'furthermore', 'for a while', 'ahead', 'for a bit']
STRAIGHT_SUFFIXES_FIRST = ['for one block', 'for a block', 'at the first intersection']
STRAIGHT_SUFFIXES_END = ['until you meet the end', 'until the end of this road',
                         'to the end of this road', 'till the end of the street']
STRAIGHT_SUFFIXES_TWO = ['for two blocks']
STRAIGHT_EXPRESSIONS = [
    'straight', 'forward',
    'go straight', 'go a little bit straight',
    'go forward', 'go a little bit forward',
    'go further', 'go a little bit further',
    'go straight down',
    'keep going straight', 'keep going forward',
    'continue going straight', 'continue going forward'
]
TURN_SUFFIXES = ['', 'here', 'up here', 'there', 'ahead', 'at the intersection']  # removed 'at the end'
TURN_SUFFIXES_FIRST = ['', 'here', 'up here', 'there', 'ahead', 'at the intersection', 'at this intersection']
TURN_SUFFIXES_SECOND = ['', 'at the second intersection']
TURN_SUFFIXES_END = ['', 'at the end of this road', 'at the end', 'at the end of the street']
TURN_EXPRESSIONS = [
    '{keyword}', 'a {keyword}', 'turn {keyword}', 'go {keyword}', 'go {keyword} turn',
    'go make a {keyword}', 'go make a {keyword} turn', 'it will be a {keyword}',
    'it will be a {keyword} turn',
    'make a {keyword}', 'make a {keyword} turn', "you're going to make a {keyword}",
    "you're going to make a {keyword} turn", "you'll make a {keyword}",
    "you'll make a {keyword} turn",
    'take a {keyword}', 'take a {keyword} turn',
    'take your {keyword}', 'take your {keyword} turn',
    "you're going to take a {keyword}", "you're going to take a {keyword} turn",
    "you're going to take your {keyword}", "you're going to take your {keyword} turn",
    "you'll take a {keyword}", "you'll take a {keyword} turn"]
CONJUNCTIONS = ['and', 'and then', 'then']
SECOND_SUFFIX = ['', 'again']


def generate_templated_sentence(keyword: str) -> List[str]:
    if keyword == 'straight':
        return generate_sentence_list_from_templates(STRAIGHT_EXPRESSIONS, STRAIGHT_PREFIXES, STRAIGHT_SUFFIXES, '')
    elif keyword in ['left', 'right', 'first left', 'first right',
                     'second left', 'second right', 'another left', 'another right']:
        return generate_sentence_list_from_templates(TURN_EXPRESSIONS, [], TURN_SUFFIXES, keyword)
    else:
        raise ValueError('invalid keyword was given {}'.format(keyword))


def generate_templated_sentence_dict() -> Dict[str, List[List[str]]]:
    sentence_dict = collections.defaultdict(list)
    sentence_group_dict = collections.defaultdict(list)
    for keyword in ['left', 'right', 'straight', 'first left', 'first right',
                    'second left', 'second right', 'another left', 'another right']:
        sentence_dict[keyword] += generate_templated_sentence(keyword)

    sentence_dict['first straight'] += generate_sentence_list_from_templates(
        STRAIGHT_EXPRESSIONS, STRAIGHT_PREFIXES, STRAIGHT_SUFFIXES_FIRST)
    sentence_dict['straight end'] += generate_sentence_list_from_templates(
        STRAIGHT_EXPRESSIONS, STRAIGHT_PREFIXES, STRAIGHT_SUFFIXES_END)
    sentence_dict['left end'] += generate_sentence_list_from_templates(TURN_EXPRESSIONS, [], TURN_SUFFIXES_END, 'left')
    sentence_dict['right end'] += generate_sentence_list_from_templates(TURN_EXPRESSIONS, [], TURN_SUFFIXES_END, 'right')

    def combine_templates(first_word, second_word, second_suffixes: List[str] = ['']):
        sentences = []
        for first, conjunction, second, second_suffix in \
                product(sentence_dict[first_word], CONJUNCTIONS, sentence_dict[second_word], second_suffixes):
            if not second_suffix and first == second:
                continue
            sentences.append(' '.join([first, conjunction, second, second_suffix]).strip())
        return sentences

    # compositional
    sentence_group_dict['left,left'].append(combine_templates('left', 'left', SECOND_SUFFIX))
    sentence_group_dict['right,right'].append(combine_templates('right', 'right', SECOND_SUFFIX))
    sentence_group_dict['left,left'].append(combine_templates('left', 'another left', SECOND_SUFFIX))
    sentence_group_dict['right,right'].append(combine_templates('right', 'another right', SECOND_SUFFIX))
    sentence_group_dict['left,right'].append(combine_templates('left', 'right'))
    sentence_group_dict['left,right'].append(combine_templates('left', 'another right'))
    sentence_group_dict['right,left'].append(combine_templates('right', 'left'))
    sentence_group_dict['right,left'].append(combine_templates('right', 'another left'))
    sentence_group_dict['left,straight'].append(combine_templates('left', 'straight'))
    sentence_group_dict['right,straight'].append(combine_templates('right', 'straight'))
    sentence_group_dict['left,straight'].append(combine_templates('first left', 'straight'))
    sentence_group_dict['right,straight'].append(combine_templates('first right', 'straight'))

    # only used for extraleft/right
    sentence_group_dict['straight,left'].append(combine_templates('straight', 'left'))
    sentence_group_dict['straight,right'].append(combine_templates('straight', 'right'))

    # special
    sentence_group_dict['second left'].append(combine_templates('first straight', 'left'))
    sentence_group_dict['second right'].append(combine_templates('first straight', 'right'))
    sentence_group_dict['first left'].append(combine_templates('straight end', 'left end'))
    sentence_group_dict['first right'].append(combine_templates('straight end', 'right end'))
    sentence_group_dict['straight,straight'].append(generate_sentence_list_from_templates(
        STRAIGHT_EXPRESSIONS, STRAIGHT_PREFIXES, STRAIGHT_SUFFIXES_TWO))

    for key in sentence_dict.keys():
        sentence_group_dict[key].append(sentence_dict[key])

    for key in ['first left', 'second left', 'first right', 'second right']:
        new_key = key.replace(' ', '')
        sentence_group_dict[new_key] = sentence_group_dict[key]

    return sentence_group_dict


class HighLevelDataset(DatasetBase):
    def __init__(
            self,
            param: Parameter,
            dataset_names: List[str],
            dataset_dict: Dict[str, InfoStorage],
            indices: List[Tuple[str, int]],
            train: bool):
        DatasetBase.__init__(self, param, dataset_names, dataset_dict, indices, train)

        self.noisy_data = param.noisy_data
        self.num_noisy_samples = param.num_noisy_samples
        self.index_from_word = load_index_from_word()
        self.use_low_level_segment = param.use_low_level_segment
        self.index_func = fetch_onehot_index_from_high_level_str
        self.sentence_library_dict = generate_templated_sentence_dict()
        logger.info('loaded {} trajectories'.format(len(self)))

        self.second_turn_dict = dict()
        self.load_second_turn_dict()

    def load_second_turn_dict(self):
        self.second_turn_dict = dict()
        for key, info_dict in self.dataset_dict.items():
            info_dir = info_dict.db_path
            turn_path = info_dir / 'second_turn.json'
            if not turn_path.exists():
                raise FileNotFoundError('second_turn.json was not found in {}'.format(info_dir))
            with open(str(turn_path), 'r') as file:
                data = json.load(file)
            set_dict = dict()
            for turn_type, turn_indices in data.items():
                set_dict[turn_type] = set(turn_indices)
            self.second_turn_dict[key] = set_dict

    def get_sentence(self, global_index: int) -> str:
        name, local_index = self._get_index_pair(global_index)
        keyword = self.dataset_dict[name].get_sentence_from_sequence(local_index).lower()

        def replace_word(word: str):
            if word.startswith('extrastraight'):
                return 'straight'
            elif word == 'extraleft':
                return 'left'
            elif word == 'extraright':
                return 'right'
            else:
                return word

        words = list(map(replace_word, keyword.split(',')))
        keyword = ','.join(words)
        if len(words) == 1:
            sentence_group = self.sentence_library_dict[keyword]
            if not sentence_group:
                logger.error('empty group from keyword {}, {}'.format(keyword, words))
        elif len(words) == 2:
            if words[0] == 'straight' and words[1] != 'straight':
                turn_dict = self.second_turn_dict[name]
                turn_types = ['firstleft', 'secondleft', 'firstright', 'secondright']
                sentence_group = []
                for turn_type in turn_types:
                    if local_index in turn_dict[turn_type]:
                        sentence_group = self.sentence_library_dict[turn_type]
                if not sentence_group:
                    sentence_group = self.sentence_library_dict[keyword]
                if not sentence_group:
                    raise ValueError('invalid first or second turn {}, {}'.format(keyword, words))
            else:
                sentence_group = self.sentence_library_dict[keyword]
            if not sentence_group:
                logger.error('empty group from keyword {}, {}'.format(keyword, words))
        else:
            raise ValueError('invalid keyword {}'.format(keyword))
        sentences = random.choice(sentence_group)
        sentence = random.choice(sentences)
        return sentence

    def get_probability_on_dataset(self, global_index: int) -> float:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_probs[self.index_from_dataset_name[name]]

    def get_sequence_data(self, global_index: int, camera_keyword: str = 'center') -> Tuple[str, str, List[dict]]:
        name, local_index = self._get_index_pair(global_index)
        if self.noisy_data:
            keyword, data_list = self.dataset_dict[name].get_noisy_data_from_sequence(
                local_index, self.max_data_length, self.num_noisy_samples, camera_keyword)
        else:
            keyword, data_list = self.dataset_dict[name].get_data_from_sequence(
                local_index, self.max_data_length, camera_keyword)
        sentence = self.get_sentence(global_index)
        return keyword.lower(), sentence, data_list

    def get_mid_drive_data(self, global_index: int) -> List[DriveDataFrame]:
        name, local_index = self._get_index_pair(global_index)
        return self.dataset_dict[name].get_mid_drive_from_sequence(local_index)

    def get_keywords(self) -> List[str]:
        keywords = []
        for global_index in range(len(self)):
            name, local_index = self._get_index_pair(global_index)
            keywords.append(self.dataset_dict[name].get_sequence(local_index)['sentence'])
        return keywords

    def get_extended_keywords(self) -> List[str]:
        keywords = []
        for global_index in range(len(self)):
            name, local_index = self._get_index_pair(global_index)
            keyword = self.dataset_dict[name].get_sentence_from_sequence(local_index).lower()
            words = keyword.split(',')
            if len(words) == 2 and words[0] == 'straight' and words[1] != 'straight':
                turn_dict = self.second_turn_dict[name]
                turn_types = ['firstleft', 'secondleft', 'firstright', 'secondright']
                for turn_type in turn_types:
                    if local_index in turn_dict[turn_type]:
                        keyword = turn_type
            keywords.append(keyword)
        return keywords

    def get_trajectory_data_from_sequence_index(self, global_index: int, camera_keyword: str = 'center') -> \
            Tuple[str, str, DataFrame]:
        name, local_index = self._get_index_pair(global_index)
        keyword, data_frame = self.dataset_dict[name].get_trajectory_data_from_sequence_index(
            local_index, camera_keyword)
        sentence = self.get_sentence(global_index)
        return keyword.lower(), sentence, data_frame

    def sample_weight_list(
            self,
            custom_weights: List[float] = list(),
            stop_counter: Dict[bool, int] = dict()):
        weight_list = []
        for i in range(len(self)):
            weight_dataset = self.get_probability_on_dataset(i)
            weight = weight_dataset
            if custom_weights:
                weight *= custom_weights[i]
            weight_list.append(weight)
        return weight_list

    def to_batch_items(self, data_dict):
        data_dict['images'] = torch.stack(
            [tensor_from_numpy_image(i, self.use_color_jitter) for i in data_dict['images']])  # S x C x H x W
        if data_dict['images'].dim() == 3:
            data_dict['images'].unsqueeze_(1)
        data_dict['word_indices'] = torch.tensor(data_dict['word_indices'], dtype=torch.long)
        return data_dict

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def __getitem__(self, global_index) -> dict:
        camera_keyword = random.choice(CAMERA_KEYWORDS)
        keyword, sentence, data_list = self.get_sequence_data(global_index, camera_keyword)
        # sentence, sequence_data, drive_data
        # location_list = list(map(lambda x: (x.state.transform.location.x, x.state.transform.location.y), drive_data))
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        # images, segments, action_types = zip(*sequence_data)
        data_frame = DataFrame()
        data_frame.images = list(map(lambda x: x['image'], data_list))
        data_frame.segments = list(map(lambda x: x['segment'], data_list))
        data_frame.custom_segments = list(map(lambda x: x['custom_segment'], data_list))
        action_types = list(map(lambda x: x['action_type'], data_list))
        indices = torch.tensor([self.index_func(action_type) for action_type in action_types])
        data_dict = {
            'index': global_index,
            'sentence': sentence.lower(),
            'word_indices': word_indices,
            'length': len(word_indices),
            'onehots': indices,  # low-level action index tensors
            'images': self.finalize_images(data_frame),
            'sequence_length': len(data_list)
        }
        return self.to_batch_items(data_dict)

    def copy_to_device(self, item: dict):
        assert 'images' in item

        # if the image is once packed, sort other data according to the indices
        tuple_keys = list(filter(lambda x: isinstance(item[x], tuple), item.keys()))
        if tuple_keys:
            indices = None
            for key in item.keys():
                if key in tuple_keys:
                    values, indices = item[key]
                    item[key] = values
            if indices is None:
                raise ValueError('tuple is not generated from the collate function: {}'.format(key))

            list_keys = list(filter(lambda x: isinstance(item[x], list), item.keys()))
            for key in list_keys:
                item[key] = [item[key][i] for i in indices]

        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(self.device_type)
        # for key, value in item.items():
        #     if isinstance(value, torch.Tensor):
        #         logger.info((key, value.shape, value.device))
        #     else:
        #         logger.info((key, type(value)))

        return item


class AblationSingleDataset(HighLevelDataset):
    def __init__(
            self,
            param: Parameter,
            dataset_names: List[str],
            dataset_dict: Dict[str, InfoStorage],
            indices: List[Tuple[str, int]],
            train: bool):
        HighLevelDataset.__init__(self, param, dataset_names, dataset_dict, indices, train)

    def to_batch_items(self, data_dict):
        data_dict['images'] = torch.stack(
            [tensor_from_numpy_image(i, self.use_color_jitter) for i in data_dict['images']])  # S x C x H x W
        if data_dict['images'].dim() == 3:
            data_dict['images'].unsqueeze_(1)
        data_dict['word_indices'] = torch.tensor(data_dict['word_indices'], dtype=torch.long)
        return data_dict

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def __getitem__(self, global_index) -> dict:
        camera_keyword = random.choice(CAMERA_KEYWORDS)
        keyword, sentence, data_frame = self.get_trajectory_data_from_sequence_index(global_index, camera_keyword)
        states = list(map(attrgetter('state'), data_frame.drives))
        controls = list(map(attrgetter('control'), data_frame.drives))
        actions = action_from_control(controls)
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        data_dict = {
            'index': global_index,
            'sentence': sentence.lower(),
            'word_indices': word_indices,
            'length': len(word_indices),
            'images': self.finalize_images(data_frame),
            'controls': controls,
            'states': states,
            'actions': actions
        }
        return self.to_batch_items(data_dict)

    @staticmethod
    def from_batch_items(data_dict: dict):
        data_dict['controls'] = [[CarControl.load_from_str(s) for s in unpackb(ss, raw=False)] for ss in
                                 data_dict['controls']]
        data_dict['states'] = [[CarState.load_from_str(s) for s in unpackb(ss, raw=False)] for ss in
                               data_dict['states']]
        return data_dict

    def to_batch_items(self, data_dict):
        data_dict['images'] = torch.stack(
            [tensor_from_numpy_image(i, self.use_color_jitter) for i in data_dict['images']])  # S x C x H x W
        if data_dict['images'].dim() == 3:
            data_dict['images'].unsqueeze_(1)
        data_dict['word_indices'] = torch.tensor(data_dict['word_indices'], dtype=torch.long)
        data_dict['actions'] = torch.tensor(data_dict['actions'], dtype=torch.float32)
        data_dict['controls'] = packb(([str(c) for c in data_dict['controls']]))
        data_dict['states'] = packb(([str(s) for s in data_dict['states']]))
        return data_dict

    def copy_to_device(self, item: dict):
        assert 'states' in item
        assert 'actions' in item
        assert 'images' in item

        item = AblationSingleDataset.from_batch_items(item)  # unpack data

        # if the image is once packed, sort other data according to the indices
        tuple_keys = list(filter(lambda x: isinstance(item[x], tuple), item.keys()))
        if tuple_keys:
            indices = None
            for key in item.keys():
                if key in tuple_keys:
                    values, indices = item[key]
                    item[key] = values
            if indices is None:
                raise ValueError('tuple is not generated from the collate function: {}'.format(key))

            list_keys = list(filter(lambda x: isinstance(item[x], list), item.keys()))
            for key in list_keys:
                item[key] = [item[key][i] for i in indices]

            lens = [len(v) for v in item['states']]
            mask = torch.ones_like(item['actions'])
            for b in range(mask.size(0)):
                mask[b, lens[b]:, :] = 0.0
            item['mask'] = mask

        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(self.device_type)

        return item


def _fetch_name_index_list(param: Parameter) -> \
        Tuple[List[str], Dict[str, InfoStorage], Dict[str, InfoStorage], List[List[Tuple[str, int]]]]:
    dataset_data_names = param.dataset_data_names
    dataset_info_names = param.dataset_info_names
    data_storage_dict: Dict[str, DataStorage] = {
        data_name: DataStorage(
            db_path=fetch_dataset_dir() / 'data' / data_name,
            use_multi_cam=param.use_multi_cam)
        for data_name in dataset_data_names}
    stop_dataset_dict: Dict[str, InfoStorage] = {
        data_name: InfoStorage(
            ref_data_storage=data_storage_dict[data_name],
            use_split_index=True,
            read_only=True,
            db_path=fetch_dataset_dir() / 'info' / info_name)
        for data_name, info_name in zip(dataset_data_names, dataset_info_names)}
    control_dataset_dict: Dict[str, InfoStorage] = {
        data_name: InfoStorage(
            ref_data_storage=data_storage_dict[data_name],
            use_split_index=False,
            read_only=True,
            db_path=fetch_dataset_dir() / 'info' / info_name)
        for data_name, info_name in zip(dataset_data_names, dataset_info_names)}

    shuffle = param.shuffle
    len_func = lambda name: stop_dataset_dict[name].num_trajectory if param.model_level == 'low' else \
        stop_dataset_dict[name].num_sequence
    lens: List[int] = list(map(len_func, dataset_data_names))
    range_func = np.random.permutation if shuffle else range
    indices = [[(n, i) for i in range_func(l)] for n, l in zip(dataset_data_names, lens)]
    return dataset_data_names, control_dataset_dict, stop_dataset_dict, indices


def _fetch_dataset_class(model_level: str, ablation_type: str):
    if ablation_type == 'single':
        return AblationSingleDataset
    else:
        if model_level == 'low':
            return LowLevelDataset
        elif model_level == 'high':
            return HighLevelDataset
        else:
            raise TypeError('invalid model_level and ablation_type {}, {}'.format(model_level, ablation_type))


def fetch_dataset(param: Parameter, is_control: bool) -> DatasetBase:
    split_train = param.split_train
    assert not split_train
    dataset_names, control_dataset_dict, stop_dataset_dict, indices = _fetch_name_index_list(param)
    dataset_dict = control_dataset_dict if param.model_level == 'low' and is_control else stop_dataset_dict
    indices = list(chain.from_iterable(indices))
    cls = _fetch_dataset_class(param.model_level, param.ablation_type)
    return cls(param, dataset_names, dataset_dict, indices, True)


def fetch_dataset_pair(param: Parameter, is_control: bool) -> Tuple[DatasetBase, DatasetBase]:
    split_train = param.split_train
    train_ratio = param.train_ratio
    assert split_train
    dataset_names, control_dataset_dict, stop_dataset_dict, indices = _fetch_name_index_list(param)
    dataset_dict = control_dataset_dict if param.model_level == 'low' and is_control else stop_dataset_dict
    train_indices, valid_indices = [], []
    for subindices in indices:
        num_segments = len(subindices)
        num_train = int(round(num_segments * train_ratio))
        train_indices.append(subindices[:num_train])
        valid_indices.append(subindices[num_train:])
    train_indices = list(chain.from_iterable(train_indices))
    valid_indices = list(chain.from_iterable(valid_indices))
    cls = _fetch_dataset_class(param.model_level, param.ablation_type)
    train_dataset = cls(param, dataset_names, dataset_dict, train_indices, True)
    valid_dataset = cls(param, dataset_names, dataset_dict, valid_indices, False)
    return train_dataset, valid_dataset


def fetch_dataset_list(param: Parameter, is_control: bool) -> List[LowLevelDataset]:
    split_train = param.split_train
    assert not split_train
    dataset_names, control_dataset_dict, stop_dataset_dict, indices = _fetch_name_index_list(param)
    dataset_dict = control_dataset_dict if param.model_level == 'low' and is_control else stop_dataset_dict
    indices = list(chain.from_iterable(indices))
    num_labels = fetch_onehot_vector_dim(param.use_low_level_segment) if not param.single_label else 1
    label_indices = [[] for _ in range(num_labels)]
    cls = _fetch_dataset_class(param.model_level, param.ablation_type)
    for name, index in indices:
        road_option = dataset_dict[name].get_road_option_from_trajectory(index)
        option_index = fetch_index_from_road_option(road_option, param.use_low_level_segment)
        label_indices[option_index].append((name, index))
    return [cls(param, dataset_names, dataset_dict, l, True) for l in label_indices]


def fetch_dataset_list_pair(param: Parameter, is_control: bool) -> \
        Tuple[List[LowLevelDataset], List[LowLevelDataset]]:
    split_train = param.split_train
    train_ratio = param.train_ratio
    assert split_train
    dataset_names, control_dataset_dict, stop_dataset_dict, indices = _fetch_name_index_list(param)
    dataset_dict = control_dataset_dict if param.model_level == 'low' and is_control else stop_dataset_dict
    indices = list(chain.from_iterable(indices))

    num_labels = fetch_onehot_vector_dim(param.use_low_level_segment) if not param.single_label else 1
    train_indices, valid_indices = [], []
    for subindices in indices:
        num_segments = len(subindices)
        num_train = int(round(num_segments * train_ratio))
        train_indices.append(subindices[:num_train])
        valid_indices.append(subindices[num_train:])
    train_indices = list(chain.from_iterable(train_indices))
    valid_indices = list(chain.from_iterable(valid_indices))

    train_label_indices = [[] for _ in range(num_labels)]
    valid_label_indices = [[] for _ in range(num_labels)]
    for name, index in train_indices:
        road_option = dataset_dict[name].get_road_option_from_trajectory(index)
        option_index = fetch_index_from_road_option(road_option)
        train_label_indices[option_index].append((name, index))
    for name, index in valid_indices:
        road_option = dataset_dict[name].get_road_option_from_trajectory(index)
        option_index = fetch_index_from_road_option(road_option)
        valid_label_indices[option_index].append((name, index))
    cls = _fetch_dataset_class(param.model_level, param.ablation_type)
    train_dataset_list = [cls(param, dataset_names, dataset_dict, l, True) for l in train_label_indices]
    valid_dataset_list = [cls(param, dataset_names, dataset_dict, l, False) for l in valid_label_indices]
    return train_dataset_list, valid_dataset_list


class DataLoaderIter(OriginalDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self.device_type = loader.device_type
        self.device_func = loader.device_func

    def __next__(self):
        data = super().__next__()
        return self.device_func(data)


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, device_type, **kwargs):
        self.device_type = device_type
        self.device_func = dataset.copy_to_device
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        return DataLoaderIter(self)


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        matched = True
        for dim in range(batch[0].dim()):
            lst = list(map(lambda x: x.size(dim), batch))
            matched = not lst or lst.count(lst[0]) == len(lst)
            if not matched:
                break
        if matched:
            return torch.stack(batch, 0, out=out)
        else:
            return pad_sequence(batch, batch_first=True)
            # indices, items = zip(*sorted(enumerate(batch), key=lambda x: x[1].size(0), reverse=True))
            # lengths = [batch[i].size(0) for i in indices]
            # logger.info(lengths)
            # return pad_sequence([batch[i] for i in indices], batch_first=True), lengths
    elif isinstance(batch[0], np.ndarray):
        matched = True
        for dim in range(batch[0].ndim):
            lst = list(map(lambda x: x.shape[dim], batch))
            matched = not lst or lst.count(lst[0]) == len(lst)
            if not matched:
                break
        if matched:
            return np.stack(batch, 0)
        else:
            raise ValueError('dimensions are not matched {}'.format(batch[0].shape))
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        raise ValueError('cannot handle numpy data')
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    raise TypeError((error_msg.format(type(batch[0]))))


def fetch_data_loader(
        dataset: DatasetBase,
        batch_size,
        num_workers,
        device_type,
        custom_weights=[],
        stop_counter = dict()):
    weight_list = dataset.sample_weight_list(custom_weights, stop_counter)
    sampler = WeightedRandomSampler(weight_list, len(weight_list))
    return DeviceDataLoader(
        dataset=dataset, device_type=device_type, batch_size=batch_size,
        sampler=sampler, collate_fn=custom_collate, num_workers=num_workers, drop_last=True)


def fetch_grouped_data_loaders(
        dataset_list: List[DatasetBase],
        batch_size,
        num_workers,
        device_type,
        custom_weights=[],
        stop_counter = dict()):
    weight_list = [dataset.sample_weight_list(custom_weights, stop_counter) for dataset in dataset_list]
    samplers = [WeightedRandomSampler(w, len(w)) for w in weight_list]
    num_workers_ = max(num_workers // 4, 1)
    return [DeviceDataLoader(
        dataset=dataset, device_type=device_type, batch_size=batch_size,
        sampler=sampler, collate_fn=custom_collate, num_workers=num_workers_, drop_last=True)
        for dataset, sampler in zip(dataset_list, samplers)]


def main():
    param = Parameter()
    dataset = fetch_dataset(param, False)
    logger.info(len(dataset))
    locations = []
    for i in range(len(dataset)):
        locations.append(dataset.__getitem__(i)['locations'])
    with open(str(Path.home() / '.tmp/high-level/training.json'), 'w') as file:
        json.dump(locations, file, indent=2)


if __name__ == '__main__':
    main()
