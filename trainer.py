import json
import re
from collections import defaultdict, OrderedDict
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Any, Dict

import torch
from torch.nn import L1Loss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from data.dataset import fetch_dataset, fetch_dataset_pair, fetch_dataset_list, fetch_dataset_list_pair, \
    fetch_data_loader, fetch_grouped_data_loaders
from model import fetch_agent, init_hidden_states
from parameter import Parameter
from util.common import get_logger, set_random_seed, fetch_node_name
from util.directory import fetch_checkpoint_dir, fetch_checkpoint_path, fetch_checkpoint_meta_path

logger = get_logger(__name__)


def find_latest_file(root_dir: Path, suffix: str) -> Path:
    def fetch_step_from_path(path: Path) -> int:
        result = list(map(lambda x: int(x), re.findall('step([0-9]+)', str(path))))
        return result[0] if result else -1
    path_step_list = [(path, fetch_step_from_path(path)) for path in root_dir.glob('*{}'.format(suffix))]
    path_step_list = list(sorted(path_step_list, key=lambda x: x[1]))
    return path_step_list[-1][0] if path_step_list else None


class CheckpointSchedulerMixin:
    def __init__(self):
        self.scheduler_steps = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        self.max_scheduled_step = self.scheduler_steps[-1]
        self.scheduler_step_size = 2500

    def save(self, step: int):
        raise NotImplementedError

    def save_scheduled(self, step: int):
        scheduled = step <= self.max_scheduled_step and step in self.scheduler_steps
        regular = step > self.max_scheduled_step and step % self.scheduler_step_size == 0
        if scheduled or regular:
            self.save(step)
            logger.info('checkpoint was saved at step {}'.format(step))


class EvaluateSchedulerMixin:
    def __init__(self):
        self.eval_scheduler_steps = [5000, 10000]
        self.eval_max_scheduled_step = self.eval_scheduler_steps[-1]
        self.eval_scheduler_step_size = 10000

    def evaluate(self, step: int):
        raise NotImplementedError

    def eval_scheduled(self, step: int):
        scheduled = step <= self.eval_max_scheduled_step and step in self.eval_scheduler_steps
        regular = step > self.eval_max_scheduled_step and step % self.eval_scheduler_step_size == 0
        if scheduled or regular:
            self.evaluate(step)
            logger.info('evaluation was requested at step {}'.format(step))


class CheckpointBase(CheckpointSchedulerMixin, EvaluateSchedulerMixin):
    def __init__(self, param: Parameter):
        self.modes = ['train', 'valid'] if param.split_train else ['train']
        self.epochs, self.steps = dict(), dict()
        for mode in self.modes:
            self.epochs[mode] = 0
            self.steps[mode] = 0

        CheckpointSchedulerMixin.__init__(self)
        EvaluateSchedulerMixin.__init__(self)

        self.exp_index = param.exp_index
        self.exp_name = param.exp_name
        self.shuffle = param.shuffle
        self.device_type = param.device_type
        self.model = fetch_agent(param)

        set_random_seed(param.random_seed)

    @property
    def step(self):
        return self.steps['train']

    @property
    def checkpoint_dir(self):
        return fetch_checkpoint_dir(self.exp_index, self.exp_name)

    def epoch_step_str(self, keyword: str):
        return 'epoch {:05d}, step {:06d}'.format(self.epochs[keyword], self.steps[keyword])

    def model_path(self, step: int):
        return fetch_checkpoint_path(self.exp_index, self.exp_name, step)

    def meta_path(self, step: int):
        return fetch_checkpoint_meta_path(self.exp_index, self.exp_name, step)

    def save_model(self, step: int):
        torch.save(self.model.state_dict(), str(self.model_path(step)))

    def save_meta(self, step: int):
        meta = dict()
        for mode in self.modes:
            meta['{}_epoch'.format(mode)] = self.epochs[mode]
            meta['{}_step'.format(mode)] = self.steps[mode]
        with open(str(self.meta_path(step)), 'w') as file:
            json.dump(meta, file, indent=2)

    def save(self, step: int):
        self.save_model(step)
        self.save_meta(step)

    def load_meta(self, step=None):
        if step is not None and step > 0:
            meta_path = self.meta_path(step)
        else:
            meta_path = find_latest_file(self.checkpoint_dir, '.json')
        if meta_path is None or not meta_path.exists():
            logger.error('could not find the meta path {}'.format(meta_path))
            return

        with open(str(meta_path), 'r') as file:
            data = json.load(file)
        logger.info('found meta data {}'.format(data))
        for mode in self.modes:
            self.epochs[mode] = data['{}_epoch'.format(mode)]
            self.steps[mode] = data['{}_step'.format(mode)]

    def load_model(self, step=None):
        if step is not None and step > 0:
            model_path = self.model_path(step)
        else:
            model_path = find_latest_file(self.checkpoint_dir, '.pth')
        if model_path is None or not model_path.exists():
            logger.error('could not find the model path {}'.format(model_path))
            return

        state_dict = torch.load(str(model_path), map_location=self.device_type)
        new_state_dict = OrderedDict()
        logger.info('checkpoint loaded from {}'.format(model_path))
        for key in state_dict.keys():
            if key.startswith('decoder.stop'):
                continue
            elif key.endswith('stop_linear.weight') or key.endswith('stop_linear.bias'):
                continue
            else:
                new_state_dict[key] = state_dict[key]
        self.model.load_state_dict(new_state_dict)

    def load(self, step=None):
        self.load_meta(step)
        self.load_model(step)


class LossAccumulator:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    @property
    def loss(self):
        if self.count == 0:
            return 0.0
        else:
            value = self.value / self.count
            self.value = 0.0
            self.count = 0
            return value

    def __add__(self, other: tuple):
        value, count = other
        self.value += value * count
        self.count += count
        return self


class Reporter:
    def __init__(self, exp_name: str, mode: str, keyword: str):
        self.name = '{}/{}/{}/{}'.format(exp_name.replace('+', '_'), keyword, mode, fetch_node_name())
        self.writer = SummaryWriter('resources/runs/{}'.format(self.name))

    def report(self, value: float, step: int):
        if self.writer is None:
            return
        self.writer.add_scalar(tag=self.name, scalar_value=value, global_step=step)


class WeightComputer:
    def __init__(self, param: Parameter):
        self.out_final = param.output_size
        self.loss_weight_type = param.loss_weight_type
        self.loss_weight_min = param.loss_weight_min
        self.loss_weight_max = param.loss_weight_max
        self.device_type = param.device_type

    def __call__(self, target_actions: torch.Tensor):
        abs_actions = torch.abs(target_actions)
        if self.loss_weight_type == 'max':
            scale = (self.loss_weight_max - self.loss_weight_min) / 2.0
            sum_abs_actions = (abs_actions[:, :, 0] + abs_actions[:, :, 1]) * scale  # [0, M - m]
            weight = sum_abs_actions + self.loss_weight_min  # [m, M]
        elif self.loss_weight_type == 'med':
            abs_steering = 1.0 - torch.abs(0.5 - abs_actions[:, :, 0])  # min: 0.5, max: 1.0
            abs_throttle = torch.tensor(abs_actions[:, :, 1] > 0.5, dtype=torch.float32).to(self.device_type) * 0.5
            abs_throttle = torch.max(abs_actions[:, :, 1], abs_throttle) * 2.0  # min: 0, max: 1.0
            sum_abs_actions = abs_throttle + abs_steering  # min: 0.5, max: 2
            weight = 0.5 * sum_abs_actions  # min: 0.25, max: 1
        elif self.loss_weight_type == 'throttle':
            scale = self.loss_weight_max - self.loss_weight_min
            abs_throttle = abs_actions[:, :, 1].to(self.device_type) * scale
            weight = abs_throttle + self.loss_weight_min
        elif self.loss_weight_type == 'nonlinear':
            TH = 0.3
            scale = (self.loss_weight_max - self.loss_weight_min) / TH
            fmask1 = (abs_actions[:, :, 1] >= TH).to(dtype=torch.float32, device=self.device_type)
            fmask2 = (abs_actions[:, :, 1] < TH).to(dtype=torch.float32, device=self.device_type)
            weight = torch.max(fmask1 * TH, abs_actions[:, :, 1] * fmask2) * scale + self.loss_weight_min
        elif self.loss_weight_type == 'steer':
            weight = torch.ones_like(abs_actions)
            weight[:, :, 1] = 0.0
        else:
            raise TypeError('invalid loss weight option {}'.format(self.loss_weight_type))
        if self.loss_weight_type != 'steer':
            weight = weight.unsqueeze(-1).expand(-1, -1, self.out_final)
        return weight


class LearningRateScheduler:
    def __init__(self, model: torch.nn.Module, step: int, param: Parameter):
        self.model = model
        self.step = step
        self.lr_init = param.lr_init
        self.lr_step = param.lr_step
        self.lr_rate = param.lr_rate

        self.optimizer = None
        self.update_optimizer_scheduled()

    @property
    def lr(self):
        value = self.lr_init
        for i in range(self.step // self.lr_step):
            value *= self.lr_rate
        return value

    # https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949
    def update_optimizer_scheduled(self):
        logger.info('optimizer was updated with lr {} at step {}'.format(self.lr, self.step))
        self.optimizer = Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)


class TrainerInstance(LearningRateScheduler):
    def __init__(
            self,
            param: Parameter,
            model: torch.nn.Module,
            sample_weights: List[float],
            sample_counts: List[int],
            train: bool,
            print_only: bool,
            epoch: int,
            step: int):

        self.train = train
        self.print_only = print_only
        self.epoch = epoch
        self.step = step

        self.exp_index = param.exp_index
        self.exp_name = param.exp_name

        self.model_type = param.model_type
        self.model_level = param.model_level
        self.ablation_type = param.ablation_type
        self.device_type = param.device_type
        self.iterative = param.decoder_iterative

        self.sample_weights = sample_weights
        self.sample_counts = sample_counts

        self.param = param

        self.eval_keyword = param.eval_keyword
        self.max_grad_norm = param.max_grad_norm

        self.model = model
        self.is_control = param.is_control
        self.output_name = 'high' if self.model_level == 'high' else ('control' if self.is_control else 'stop')
        self.loss_computer = LossAccumulator()
        self.reporter = Reporter(self.exp_name, self.mode, self.output_name)

        self.weight_computer = WeightComputer(param) if param.loss_weight_type != 'none' else None
        self.criterion, self.stop_criterion = None, None
        self.initialize_criterion()

        LearningRateScheduler.__init__(self, model, step, param)

        self.step_elapsed = 0
        self.print_count = 0
        self.loss = 0.0
        self.opt_count_dict = defaultdict(int)

    def initialize_criterion(self):
        if self.model_level == 'low':
            if self.is_control:
                self.criterion = L1Loss(reduction='sum')
            else:
                self.criterion = BCEWithLogitsLoss(reduction='sum')
        else:
            self.criterion = CrossEntropyLoss()

    def run_loss(self, data: dict, output: dict) -> Tuple[Any, int]:
        """
        custom loss function
        :param data: data dictionary from the dataset
        :param output: output dictionary from the model (batch_size, sequence_length, dim)
        :return: loss value
        """
        if self.model_level == 'low':
            if self.is_control:
                control, control_predict = data['actions'], output['output']
                control_loss_count = control_predict.numel()
                control_loss = self.criterion(control_predict, control) / control_loss_count
                return control_loss, control_loss_count
            else:
                stop_target, stop_predict = data['stops'], output['output']
                stop_loss_count = stop_predict.numel()
                stop_loss = self.criterion(stop_predict, stop_target) / stop_loss_count
                return stop_loss, stop_loss_count
        else:
            control, control_predict = data['onehots'], output['output'].transpose(1, 2)
            control_loss_count = 1
            control_loss = self.criterion(control_predict, control)
            return control_loss, control_loss_count

    @property
    def mode(self):
        return 'train' if self.train else 'valid'

    def init_epoch(self):
        self.step_elapsed = 0
        self.print_count = 0
        self.loss = 0.0

    def run_batch(self, data: dict) -> Dict[bool, int]:
        self.model.train() if self.train else self.model.eval()
        self.optimizer.zero_grad()

        if 'type' in data:
            for opt in data['type']:
                self.opt_count_dict[opt] += 1

        encoder_hidden, decoder_hidden = init_hidden_states(self.param)
        model_output = self.model(data, encoder_hidden, decoder_hidden, self.step)

        loss, loss_count = self.run_loss(data, model_output)

        if self.train and not self.print_only:
            loss.backward()  # calculates the gradients
            if self.max_grad_norm > 0:
                params = chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.optimizer.step()

        if self.step % self.lr_step == 0:
            self.update_optimizer_scheduled()

        local_stop_counter = dict()
        if 'stops' in data:
            local_stop_counter[True] = int(torch.sum(data['stops']).detach().cpu().item())
            local_stop_counter[False] = data['stops'].numel() - local_stop_counter[True]

        self.loss_computer += loss.item(), loss_count
        self.step_elapsed += 1
        self.step += 1
        return local_stop_counter

    def print_loss(self, weights: List[float] = list()):
        loss_value = self.loss_computer.loss
        loss_str = '{} {:+5.3f}'.format(self.output_name, loss_value)
        train_str = 'epoch {:05d}, step {:06d}, {}, {}'.format(self.epoch, self.step, self.mode, loss_str)
        if weights:
            train_str += ', weight min {:+5.3f}, max {:+5.3f}, avg {:+5.3f}'.format(
                min(weights), max(weights), sum(weights) / len(weights))
        count = sum(self.opt_count_dict.values())
        if count > 0:
            train_str += ', ' + ', '.join(['{:+5.3f}'.format(v / count) for k, v in self.opt_count_dict.items()])
        logger.info(train_str)
        if not self.print_only:
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            self.reporter.report(loss_value, self.step)
        self.print_count += 1


class Trainer(CheckpointBase):
    def __init__(self, param: Parameter, print_only: bool):
        CheckpointBase.__init__(self, param)

        self.stop_counter = defaultdict(int)
        self.grouped_batch = param.grouped_batch
        self.model_level = param.model_level
        self.is_control = param.is_control
        dataset_pair_func = fetch_dataset_list_pair if self.grouped_batch else fetch_dataset_pair
        dataset_func = fetch_dataset_list if self.grouped_batch else fetch_dataset

        if self.is_stop:
            self.stop_counter[True] += 1
            self.stop_counter[False] += 1

        if param.split_train:
            self.datasets = {mode: dataset
                             for mode, dataset in zip(self.modes, dataset_pair_func(param, self.is_control))}
        else:
            self.datasets = {self.modes[0]: dataset_func(param, self.is_control)}

        if self.grouped_batch:
            self.sample_weights = {mode: None for mode in self.modes}
            self.sample_counts = {mode: None for mode in self.modes}
        else:
            self.sample_weights = {mode: [1.0 for _ in range(len(self.datasets[mode]))] for mode in self.modes}
            self.sample_counts = {mode: [1 for _ in range(len(self.datasets[mode]))] for mode in self.modes}

        self.debug = param.debug
        self.print_only = print_only
        self.split_train = param.split_train

        self.instances = dict()
        for mode in self.modes:
            self.instances[mode] = TrainerInstance(
                param, self.model,
                self.sample_weights[mode],
                self.sample_counts[mode],
                True if mode == 'train' else False,
                self.print_only, self.epochs[mode], self.steps[mode])

        self.print_every = param.print_every
        self.valid_every = param.valid_every
        self.num_workers = param.num_workers
        self.batch_size = param.batch_size
        self.num_epoch = param.num_epoch

    @property
    def is_stop(self):
        return self.model_level == 'low' and not self.is_control

    def save(self, step: int):
        for mode in self.modes:
            self.epochs[mode] = self.instances[mode].epoch
            self.steps[mode] = self.instances[mode].step
        self.save_model(step)
        self.save_meta(step)

    def call_batch_function(self, mode: str, batch: dict):
        local_stop_counter = self.instances[mode].run_batch(batch)
        for key in local_stop_counter.keys():
            self.stop_counter[key] += local_stop_counter[key]

        if self.instances[mode].step % self.print_every == 0 and self.instances[mode].step_elapsed > 0:
            self.instances[mode].print_loss(self.normalized_weight_list(mode))

        if mode == 'train' and not self.print_only:
            self.save_scheduled(self.instances[mode].step)
            self.eval_scheduled(self.instances[mode].step)

    def normalized_weight_list(self, mode: str) -> List[float]:
        if self.sample_weights[mode] is None:
            return []
        else:
            return [w / c for w, c in zip(self.sample_weights[mode], self.sample_counts[mode])]

    def run_epoch(self, mode: str):
        self.instances[mode].init_epoch()

        if self.grouped_batch:
            for dataset in self.datasets[mode]:
                assert len(dataset) >= self.batch_size
            loaders = fetch_grouped_data_loaders(
                self.datasets[mode],
                self.batch_size,
                self.num_workers,
                self.device_type,
                stop_counter=self.stop_counter)
            for batches in zip(*loaders):
                for batch in batches:
                    self.call_batch_function(mode, batch)
                if self.debug:
                    break
        else:
            loader = fetch_data_loader(
                self.datasets[mode],
                self.batch_size,
                self.num_workers,
                self.device_type,
                stop_counter=self.stop_counter)
            for batch in loader:
                self.call_batch_function(mode, batch)
                if self.debug:
                    break

        # if self.instances[mode].step_elapsed > 0 and self.instances[mode].print_count == 0:
        #     self.instances[mode].print_loss(self.normalized_weight_list(mode))
        self.instances[mode].epoch += 1

    def train(self):
        for mode in self.modes:
            self.instances[mode].update_optimizer_scheduled()
        while self.epochs['train'] < self.num_epoch:
            self.run_epoch('train')
            if self.split_train and self.epochs['train'] % self.valid_every == 0:
                self.run_epoch('valid')

    def load_meta(self, step=None):
        CheckpointBase.load_meta(self, step)
        for mode in self.modes:
            self.instances[mode].step = self.steps[mode]
            self.instances[mode].epoch = self.epochs[mode]


def main():
    param = Parameter()
    param.save()
    trainer = Trainer(param=param, print_only=False)
    trainer.load()
    trainer.train()


if __name__ == '__main__':
    main()
