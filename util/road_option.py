from typing import List

import torch
from custom_carla.agents.navigation.local_planner import ROAD_OPTION_ACTIONS, RoadOption

ROAD_OPTION_NAMES = RoadOption.__dict__['_member_names_']
ROAD_OPTION_LIST = [RoadOption.__dict__['_member_map_'][name] for name in ROAD_OPTION_NAMES]
ROAD_OPTION_FROM_NAME = {name: RoadOption.__dict__['_member_map_'][name] for name in ROAD_OPTION_NAMES}
ROAD_OPTION_NAME_FROM_OPTION = {v: k for k, v in ROAD_OPTION_FROM_NAME.items()}
LOW_LEVEL_OPTION_LIST = ROAD_OPTION_LIST[1:]
HIGH_LEVEL_OPTION_LIST = ROAD_OPTION_LIST[1:5]
LOW_LEVEL_INDEX_FROM_OPTION = {o: i for i, o in enumerate(LOW_LEVEL_OPTION_LIST)}
HIGH_LEVEL_INDEX_FROM_OPTION = {o: i for i, o in enumerate(HIGH_LEVEL_OPTION_LIST)}

HIGH_LEVEL_COMMAND_NAMES = ['left', 'right', 'straight', 'lanefollow', 'stop']
INDEX_FROM_HIGH_LEVEL_COMMAND_NAMES = {n: i for i, n in enumerate(HIGH_LEVEL_COMMAND_NAMES)}

SENTENCE_COMMAND_NAMES = ['left', 'right', 'straight']
INDEX_FROM_SENTENCE_COMMAND_NAMES = {n: i for i, n in enumerate(SENTENCE_COMMAND_NAMES)}

ABLATION_COMMAND_NAMES = ['left', 'right', 'straight', 'lanefollow']
INDEX_FROM_ABLATION_COMMAND_NAMES = {n: i for i, n in enumerate(ABLATION_COMMAND_NAMES)}


def fetch_road_option_from_str(name: str) -> RoadOption:
    words = name.split('.')
    if len(words) == 2:
        name = words[1]
    return ROAD_OPTION_FROM_NAME[name.upper()]


def fetch_name_from_road_option(opt: RoadOption) -> str:
    return opt.name


def fetch_road_option_list(low_level: bool) -> List[RoadOption]:
    return LOW_LEVEL_OPTION_LIST if low_level else HIGH_LEVEL_OPTION_LIST


def fetch_road_option_from_index(index: int, low_level: bool) -> RoadOption:
    return fetch_road_option_list(low_level)[index]


def fetch_onehot_vector_dim(low_level: bool) -> int:
    return len(fetch_road_option_list(low_level))


def fetch_index_from_road_option(opt: RoadOption, low_level: bool) -> int:
    return LOW_LEVEL_INDEX_FROM_OPTION[opt] if low_level else HIGH_LEVEL_INDEX_FROM_OPTION[opt]


def fetch_onehot_vector_from_index(index: int, low_level: bool) -> torch.Tensor:
    tensor = torch.zeros((fetch_onehot_vector_dim(low_level),), dtype=torch.float32)
    tensor[index] = 1.0
    return tensor


def fetch_onehot_vector_from_road_option(opt: RoadOption, low_level: bool) -> torch.Tensor:
    index = fetch_index_from_road_option(opt, low_level)
    return fetch_onehot_vector_from_index(index, low_level)


def fetch_nmc_index_from_road_option(opt: RoadOption) -> int:
    return fetch_index_from_road_option(opt, False)


def fetch_nmc_onehot_vector_from_index(index: int) -> torch.Tensor:
    return fetch_onehot_vector_from_index(index, False)


def fetch_nmc_onehot_vector_from_road_option(opt: RoadOption):
    index = fetch_nmc_index_from_road_option(opt)
    return fetch_nmc_onehot_vector_from_index(index)


def fetch_num_high_level_commands() -> int:
    return len(HIGH_LEVEL_COMMAND_NAMES)


def fetch_high_level_command_from_index(index: int) -> str:
    return HIGH_LEVEL_COMMAND_NAMES[index]


def fetch_onehot_index_from_high_level_str(name: str) -> int:
    return INDEX_FROM_HIGH_LEVEL_COMMAND_NAMES[name.lower()]


def fetch_onehot_vector_from_high_level_str(name: str) -> torch.Tensor:
    index = fetch_onehot_index_from_high_level_str(name)
    dim = fetch_num_high_level_commands()
    tensor = torch.zeros((dim,), dtype=torch.float32)
    tensor[index] = 1.0
    return tensor


def fetch_index_tensor_from_high_level_str(name: str) -> torch.Tensor:
    index = fetch_onehot_index_from_high_level_str(name)
    return torch.tensor([index], dtype=torch.float32).view((1,))


def fetch_num_sentence_commands() -> int:
    return len(SENTENCE_COMMAND_NAMES)


def fetch_sentence_command_from_index(index: int) -> str:
    return SENTENCE_COMMAND_NAMES[index]


def fetch_onehot_index_from_sentence_command(name: str) -> int:
    return INDEX_FROM_SENTENCE_COMMAND_NAMES[name.lower()]


def fetch_onehot_vector_from_sentence_command(name: str) -> torch.Tensor:
    index = fetch_onehot_index_from_sentence_command(name)
    dim = fetch_num_sentence_commands()
    tensor = torch.zeros((dim,), dtype=torch.float32)
    tensor[index] = 1.0
    return tensor


def is_road_option_action(opt: RoadOption) -> bool:
    return opt in ROAD_OPTION_ACTIONS
