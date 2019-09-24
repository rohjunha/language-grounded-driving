import os
import tarfile
from functools import partial
from pathlib import Path
from random import uniform
from typing import List, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf

import torch
from PIL import Image
from six.moves import urllib
from torch.nn import Module, ModuleList, ELU, Linear, BatchNorm1d, Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_TENSOR_HEIGHT, IMAGE_TENSOR_WIDTH, STOP_DIM
from parameter import Parameter
from util.common import get_logger
from util.directory import fetch_word_embeddings_dir, mkdir_if_not_exists
from util.road_option import fetch_onehot_vector_dim, fetch_num_sentence_commands

logger = get_logger(__name__)


__key_check_str_dict__ = {
    'images': 'bs-hw',
    'actions': 'bs-',
    'stops': 'bs{}'.format(STOP_DIM),
    'onehots': 'bs'
}
__split_keys__ = set(filter(lambda x: 's' in __key_check_str_dict__[x], __key_check_str_dict__.keys()))
__non_split_keys__ = set(filter(lambda x: 's' not in __key_check_str_dict__[x], __key_check_str_dict__.keys()))


def check_tensor_size(x: torch.Tensor, size_str: str, size_dict: Dict[str, int]) -> None:
    if x.dim() != len(size_str):
        raise ValueError('dimension mismatches {}, {}'.format(x.dim(), size_str))
    for i in range(x.dim()):
        if size_str[i] == '-':
            continue
        si = x.size(i)
        sj = int(size_str[i]) if '0' <= size_str[i] < '9' else size_dict[size_str[i]]
        if si != sj:
            raise ValueError('size mismatches at {}: {}, {}'.format(i, si, sj))


def split_batch(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    valid_split_keys = list(filter(lambda x: x in batch, __key_check_str_dict__.keys()))
    seq_len = batch['images'].size(1)

    splits = dict()
    for key in valid_split_keys:
        if key not in batch:
            continue
        if batch[key] is None:
            splits[key] = [None for _ in range(seq_len)]
        else:
            splits[key] = torch.split(batch[key], 1, 1)

    batches = [{key: splits[key][s] if key in __split_keys__ else batch[key]
                for key in valid_split_keys} for s in range(seq_len)]

    if 'action_index' in batch:
        for i in range(len(batches)):
            batches[i]['action_index'] = batch['action_index']
    if 'sequence_length' in batch:
        for i in range(len(batches)):
            batches[i]['sequence_length'] = torch.tensor(batch['sequence_length'] - i > 0, dtype=torch.long)

    return batches


def merge_output(output_list: List[Dict[str, torch.Tensor]]):
    if not output_list:
        return dict()
    output = {
        'encoder_hidden': output_list[-1]['encoder_hidden'],  # 1be
        'decoder_hidden': output_list[-1]['decoder_hidden'],   # 1bd
        # save the last attention value
        'attention': output_list[-1]['attention'].view(-1) if output_list[-1]['attention'] is not None else None
    }
    for key in ['output', 'decoder_output']:
        output[key] = torch.cat(list(map(lambda x: x[key], output_list)), 1)  # bs2, bsd
    return output


class FullyConnectedLayersBase(Module):
    def __init__(self, dims: List[int], dropout_probability: float = 0.0, concat_dim: int = 0):
        super().__init__()
        n_dim = len(dims)
        self.concat_dim = concat_dim
        self.use_dropout = dropout_probability > 0.0
        if concat_dim == 0:
            self.fcl = ModuleList([Linear(dims[i], dims[i + 1]) for i in range(n_dim - 1)])
        else:
            in_dims = [dims[i] + concat_dim for i in range(n_dim - 1)]
            out_dims = [dims[i] for i in range(1, n_dim)]
            self.fcl = ModuleList([Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.ell = ModuleList([ELU() for _ in dims[1:-1]])
        if self.use_dropout:
            self.dol = ModuleList([Dropout(p=dropout_probability) for _ in dims[1:-1]])

    def process(self, x, i):
        raise NotImplementedError

    def concat(self, x, y):
        if self.concat_dim > 0 and y is not None:
            return torch.cat((x, y), -1)
        else:
            return x

    def forward(self, x, y: torch.Tensor = None):
        for i in range(len(self.ell)):
            x = self.concat(x, y)
            x = self.fcl[i](x)
            x = self.process(x, i)
            x = self.ell[i](x)
            if self.use_dropout:
                x = self.dol[i](x)
        x = self.concat(x, y)
        x = self.fcl[-1](x)
        return x


class FullyConnectedLayersNoBatchNorm(FullyConnectedLayersBase):
    def __init__(self, dims: List[int], dropout_probability: float = 0.0, concat_dim: int = 0):
        FullyConnectedLayersBase.__init__(self, dims, dropout_probability, concat_dim)

    def process(self, x, i):
        return x


class FullyConnectedLayersBatchNormFeature(FullyConnectedLayersBase):
    def __init__(self, dims: List[int], dropout_probability: float = 0.0, concat_dim: int = 0):
        FullyConnectedLayersBase.__init__(self, dims, dropout_probability, concat_dim)
        self.bnl = ModuleList([BatchNorm1d(v) for v in dims[1:-1]])

    def process(self, x, i):
        if x.dim() == 2:
            return self.bnl[i](x)
        elif x.dim() == 3:
            for s in range(x.shape[1]):
                # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-
                # gradient-computation-has-been-modified-by-an-inplace-operation/836
                x[:, s, :] = self.bnl[i](x[:, s, :].clone())
            return x
        else:
            raise ValueError('dimension of the input must be two or three')


class FullyConnectedLayersBatchNormSequence(FullyConnectedLayersBase):
    def __init__(self, dims: List[int], num_seq: int, dropout_probability: float = 0.0, concat_dim: int = 0):
        FullyConnectedLayersBase.__init__(self, dims, dropout_probability, concat_dim)
        self.num_seq = num_seq
        self.bnl = ModuleList([BatchNorm1d(self.num_seq) for _ in dims[1:-1]])

    def process(self, x, i):
        if x.dim() == 3:
            if x.shape[1] != self.num_seq:
                raise ValueError('the length of the sequence is different from the parameter: {}, {}'.format(
                    x.shape[1], self.num_seq))
            return self.bnl[i](x)
        else:
            raise ValueError('dimension of the input must be three')


def fetch_fully_connected_layers(batch_norm_option: str, dims: List[int], num_seq: int = -1,
                                 dropout_probability: float = 0.0, concat_dim: int = 0):
    if batch_norm_option == 'none':
        return FullyConnectedLayersNoBatchNorm(dims, dropout_probability, concat_dim)
    elif batch_norm_option == 'sequence':
        if num_seq < 1:
            raise ValueError('the number of sequence should be positive: {}'.format(num_seq))
        return FullyConnectedLayersBatchNormSequence(dims, num_seq, dropout_probability, concat_dim)
    elif batch_norm_option == 'feature':
        return FullyConnectedLayersBatchNormFeature(dims, dropout_probability, concat_dim)
    else:
        raise NotImplementedError('no other option is possible for batch norm')


class ConvLayers(torch.nn.Module):
    def __init__(self, channels: List[int]):
        torch.nn.Module.__init__(self)

        # image processing
        self.conv1 = torch.nn.Conv2d(channels[0], channels[1], kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2)
        self.image_feat_channel = channels[-1]

    def forward(self, data: dict) -> torch.Tensor:
        # Get the image representation
        images = data['images']
        batch_size = images.size(0)
        size_dict = {
            'b': batch_size,
            'f': self.image_feat_channel,
            'h': IMAGE_TENSOR_HEIGHT,
            'w': IMAGE_TENSOR_WIDTH
        }
        if images.dim() == 4:
            x = torch.nn.functional.elu(self.conv1(images))
            x = torch.nn.functional.elu(self.conv2(x))
            x_image_rep = torch.nn.functional.elu(self.conv3(x))  # B x F x IH x IW
            check_tensor_size(x, 'bfhw', size_dict)
        elif images.dim() == 5:
            xs = []
            size_dict['s'] = images.size(1)
            c = images.size(2)
            for s in range(images.size(1)):
                image = images[:, s, :, :, :].squeeze().view(batch_size, c, IMAGE_HEIGHT, IMAGE_WIDTH)
                x = torch.nn.functional.elu(self.conv1(image))
                x = torch.nn.functional.elu(self.conv2(x))
                x = torch.nn.functional.elu(self.conv3(x))
                xs.append(x.view(batch_size, self.image_feat_channel, IMAGE_TENSOR_HEIGHT, IMAGE_TENSOR_WIDTH))
            x_image_rep = torch.stack(xs, dim=1)  # B x S x F x IH x IW
            check_tensor_size(x_image_rep, 'bsfhw', size_dict)
        else:
            raise ValueError('invalid dim of images: {}'.format(images.dim()))
        return x_image_rep


class EarlyFusionConvLayers(torch.nn.Module):
    def __init__(self, channels: List[int], encoder_hidden_size: int):
        torch.nn.Module.__init__(self)

        # image processing
        self.encoder_hidden_size = encoder_hidden_size
        self.conv1 = torch.nn.Conv2d(channels[0], channels[1], kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(channels[1] + self.encoder_hidden_size, channels[2], kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2)
        self.image_feat_channel = channels[-1]

    def forward_single_sequence_batch(self, single_step_batch: torch.Tensor, x_encode: torch.Tensor):
        """
        processes a single step of image batch and generates an image feature
        :param single_step_batch: B x 3 x H x W
        :param x_encode: 1 x B x E
        :return: image representation B x F x IH x IW
        """
        batch_size = single_step_batch.size(0)
        size_dict = {
            'b': batch_size,
            'f': self.image_feat_channel,
            'e': self.encoder_hidden_size,
            'h': IMAGE_HEIGHT,
            'w': IMAGE_WIDTH,
            'i': IMAGE_TENSOR_HEIGHT,
            'j': IMAGE_TENSOR_WIDTH
        }
        check_tensor_size(single_step_batch, 'b3hw', size_dict)
        check_tensor_size(x_encode, '1be', size_dict)
        batch_size = x_encode.size(1)

        x = torch.nn.functional.elu(self.conv1(single_step_batch))

        # concatenate tiled encoded tensor to the image representation from the first conv layers
        x_encode = x_encode.view(batch_size, self.encoder_hidden_size, 1, 1)
        x_encode = x_encode.expand(batch_size, self.encoder_hidden_size, x.size(2), x.size(3))
        x = torch.cat((x, x_encode), dim=1)  # B, E, IH, IW

        x = torch.nn.functional.elu(self.conv2(x))
        x_image_rep = torch.nn.functional.elu(self.conv3(x))
        check_tensor_size(x_image_rep, 'bfij', size_dict)
        return x_image_rep

    def forward(self, data: dict, x_encode: torch.Tensor) -> torch.Tensor:
        """
        generates image feature representation
        :param data: input data
        :param x_encode: 1 x B x E
        :return: image representation B (x S) x F x IH x IW
        """
        # Get the image representation
        images = data['images']
        batch_size = images.size(0)
        size_dict = {
            'b': batch_size,
            'f': self.image_feat_channel,
            'e': self.encoder_hidden_size,
            'h': IMAGE_HEIGHT,
            'w': IMAGE_WIDTH,
            'i': IMAGE_TENSOR_HEIGHT,
            'j': IMAGE_TENSOR_WIDTH
        }
        dims = [batch_size, self.image_feat_channel, IMAGE_TENSOR_HEIGHT, IMAGE_TENSOR_WIDTH]
        if images.dim() == 4:
            x_image_rep = self.forward_single_sequence_batch(images, x_encode)
        elif images.dim() == 5:
            xs = []
            size_dict['s'] = images.size(1)
            for s in range(images.size(1)):
                image = images[:, s, :, :, :].squeeze().view(batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
                x_single_image_rep = self.forward_single_sequence_batch(image, x_encode)
                x_single_image_rep = x_single_image_rep.view(*dims)
                xs.append(x_single_image_rep.view(*dims))
            x_image_rep = torch.stack(xs, dim=1)
            check_tensor_size(x_image_rep, 'bsfij', size_dict)
        else:
            raise ValueError('invalid dim of images: {}'.format(images.dim()))
        return x_image_rep


def glove_word_embedding_layer() -> torch.nn.Module:
    out_parameter = fetch_word_embeddings_dir() / 'param.pth'
    num_embeddings, embedding_dim = 100, 50
    embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    assert out_parameter.exists()
    embedding_layer.load_state_dict(torch.load(str(out_parameter)))
    return embedding_layer


class Encoder(torch.nn.Module):
    def __init__(self, param: Parameter, num_words: int):
        torch.nn.Module.__init__(self)
        self.num_words = num_words
        self.encoder_type = param.encoder_type
        self.encoder_embedding_size = param.encoder_embedding_size
        self.encoder_hidden_size = param.encoder_hidden_size
        self.use_glove_embedding = param.use_glove_embedding
        self.use_low_level_segment = param.use_low_level_segment
        if param.model_level == 'low':
            self.onehot_dim = fetch_onehot_vector_dim(param.use_low_level_segment)
        else:
            self.onehot_dim = fetch_num_sentence_commands()

        # encoder
        if self.encoder_type == 'gru':
            if self.use_glove_embedding:
                self.embedding = glove_word_embedding_layer()
            else:
                self.embedding = torch.nn.Embedding(self.num_words, self.encoder_embedding_size, padding_idx=0)
            self.encoder_gru = torch.nn.GRU(
                input_size=self.encoder_embedding_size,
                hidden_size=self.encoder_hidden_size,
                batch_first=True)
        elif self.encoder_type == 'onehot':
            self.encoder_linear = torch.nn.Linear(self.onehot_dim, self.encoder_hidden_size)
        else:
            raise TypeError('invalid encoder_type {}'.format(self.encoder_type))

    def forward(self, data: dict, encoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == 'gru':
            embedded = self.embedding(data['word_indices'])  # embedding (batch_size, sequence_length, hidden_size)
            if embedded.dim() == 1:
                embedded = embedded.view(1, 1, -1)
            elif embedded.dim() == 2:
                embedded.unsqueeze_(0)
            assert embedded.dim() == 3
            embedded = pack_padded_sequence(
                input=embedded, lengths=data['length'], batch_first=True, enforce_sorted=False)
            encoder_output, encoder_hidden = self.encoder_gru(embedded, encoder_hidden)
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
            batch_size, seq_len = encoder_output.size(0), encoder_output.size(1)
            x_instr_rep = encoder_hidden.view(-1, batch_size, encoder_output.size(2))  # 1, batch_size, hidden_size
        else:
            encoder_output = encoder_hidden = self.encoder_linear(data['onehot'])  # batch_size, onehot_dim
            batch_size, onehot_dim = encoder_output.size(0), encoder_output.size(1)
            encoder_output = encoder_output.view(batch_size, 1, onehot_dim)
            encoder_hidden = encoder_hidden.view(1, batch_size, onehot_dim)
            x_instr_rep = encoder_hidden  # 1, batch_size, hidden_size
        return encoder_output, x_instr_rep


class Decoder(torch.nn.Module):
    def __init__(self, param: Parameter, in_decoder: int):
        torch.nn.Module.__init__(self)
        self.decoder_type = param.decoder_type
        self.is_control = param.is_control
        self.out_final = param.output_size
        self.in_decoder = in_decoder
        self.out_decoder = param.out_decoder
        self.use_history = param.use_history  # if param.model_type != 'se' else False

        if self.use_history:
            self.out_extra = param.history_size
            self.extra_linear = torch.nn.Linear(self.out_final, self.out_extra)
            self.in_decoder += self.out_extra

        if self.recurrent:
            self.decoder_gru = torch.nn.GRU(self.in_decoder, self.out_decoder, batch_first=True)
        self.final_linear = torch.nn.Linear(self.in_final, self.out_final)

    @property
    def recurrent(self):
        return self.decoder_type in ['gru', 'lstm']

    @property
    def in_final(self):
        return self.out_decoder if self.recurrent else self.in_decoder

    def forward(
            self,
            x: torch.Tensor,
            decoder_hidden: torch.Tensor,
            extra_action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        decoder_output = None
        if self.recurrent:
            if self.use_history:
                assert extra_action is not None
                x = torch.cat((x, self.extra_linear(extra_action)), -1)
            decoder_output, decoder_hidden = self.decoder_gru(x, decoder_hidden)
            x = decoder_output
        output = self.final_linear(x)
        return {
            'output': output,
            'attention': None,
            'decoder_output': decoder_output,
            'decoder_hidden': decoder_hidden
        }


def check_hidden_tensor_sizes(
        encoder_output: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_hidden_size: int,
        decoder_hidden_size: int) -> Dict[str, int]:
    assert encoder_output.size(0) == decoder_hidden.size(1)
    batch_size, seq_len = encoder_output.size(0), encoder_output.size(1)
    size_dict = {
        'b': batch_size,
        's': seq_len,
        'e': encoder_hidden_size,
        'd': decoder_hidden_size}
    check_tensor_size(encoder_output, 'bse', size_dict)
    check_tensor_size(decoder_hidden, '1bd', size_dict)
    return batch_size, seq_len, size_dict


class AttentionLuong(torch.nn.Module):
    def __init__(self, param: Parameter):
        torch.nn.Module.__init__(self)
        self.encoder_hidden_size = param.encoder_hidden_size
        self.decoder_hidden_size = param.decoder_hidden_size
        self.score_linear = torch.nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

    def align(self, encoder_output: torch.Tensor, decoder_hidden: torch.Tensor) -> torch.Tensor:
        # encoder_output: batch, seq_len, num_directions * encoder_hidden_size
        # decoder_hidden: num_layers * num_directions, batch, decoder_hidden_size
        batch_size, seq_len = encoder_output.size(0), encoder_output.size(1)
        x = self.score_linear(encoder_output)  # batch, seq_len, decoder_hidden_size
        x = torch.bmm(x, decoder_hidden.squeeze().view(-1, self.decoder_hidden_size, 1)).squeeze()  # score: B, S
        x = x.view(batch_size, seq_len)
        x = torch.nn.functional.softmax(x, dim=1)  # weight: B, S
        return x

    def context(self, aligned_weight: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # aligned_weight: batch, seq_len
        # encoder_output: batch, seq_len, num_directions * encoder_hidden_size
        x = aligned_weight.unsqueeze(-1).expand(*encoder_output.shape)
        x = x * encoder_output
        x = torch.sum(x, dim=1)  # batch, encoder_hidden_size
        return x

    def forward(self, encoder_output: torch.Tensor, decoder_hidden: torch.Tensor):
        # encoder_output: batch, seq_len, num_directions * encoder_hidden_size
        # decoder_hidden: num_layers * num_directions, batch, decoder_hidden_size
        batch_size, seq_len, size_dict = check_hidden_tensor_sizes(
            encoder_output, decoder_hidden, self.encoder_hidden_size, self.decoder_hidden_size)
        attention_weight = self.align(encoder_output, decoder_hidden)
        check_tensor_size(attention_weight, 'bs', size_dict)
        context = self.context(attention_weight, encoder_output)
        context = context.view(1, batch_size, self.encoder_hidden_size)
        check_tensor_size(context, '1be', size_dict)
        return context, attention_weight


class ModelBase(torch.nn.Module):
    def __init__(self, param: Parameter):
        torch.nn.Module.__init__(self)

        self.output_size = param.output_size
        self.out_linear = param.out_linear
        self.channels = param.cv_visual_dims
        self.image_feat_channel = self.channels[-1]
        self.decoder_type = param.decoder_type
        self.decoder_rollout = param.decoder_rollout
        self.out_decoder = param.out_decoder
        self.use_history = param.use_history
        self.is_control = param.is_control
        self.scheduled_sampling = param.scheduled_sampling
        self.scheduled_iteration = param.scheduled_iteration
        self.model_level = param.model_level
        self.ablation_type = param.ablation_type

    @property
    def decoder_hidden_size(self):
        return self.out_decoder

    @property
    def in_decoder(self) -> int:
        return self.out_linear

    @property
    def image_vec_dim(self) -> int:
        return self.image_feat_channel * IMAGE_TENSOR_HEIGHT * IMAGE_TENSOR_WIDTH


class ModelSeparated(ModelBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelBase.__init__(self, param)

        self.use_low_level_segment: bool = param.use_low_level_segment
        self.onehot_dim: int = fetch_onehot_vector_dim(self.use_low_level_segment)

        self.image_processor = ConvLayers(self.channels)
        self.linear = torch.nn.Linear(self.image_vec_dim, self.out_linear)
        self.decoders = ModuleList([Decoder(param, self.in_decoder) for _ in range(self.onehot_dim)])

    def subforward(
            self,
            data: dict,
            decoder_hidden: torch.Tensor,
            extra_action: torch.Tensor = None):
        """this function assumes that all the data in the batch belong to the same label"""
        x = self.image_processor.forward(data)  # bs*
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, -1)  # batch_size, seq_len, 9216
        x = torch.nn.functional.elu(self.linear(x))  # batch_size, seq_len, in_decoder_channel
        decoder_index = data['action_index'][0]
        out_dict = self.decoders[decoder_index].forward(x, decoder_hidden, extra_action)
        out_dict['encoder_hidden'] = None
        return out_dict

    def forward(self, data: dict, encoder_hidden: torch.Tensor, decoder_hidden: torch.Tensor, iteration: int = -1):
        # decoder_hidden: num_layers * num_directions, batch, decoder_hidden_size
        if self.decoder_type == 'gru' and self.decoder_rollout:
            teacher_probability = min(1.0, max(0.0, 1.0 - iteration / self.scheduled_iteration))
            single_data_list = split_batch(data)
            single_output_list = []
            for i, d in enumerate(single_data_list):
                if self.use_history:
                    if not single_output_list:
                        extra_action = torch.zeros((decoder_hidden.size(1), 1, self.output_size),
                                                   dtype=torch.float32, device=decoder_hidden.device)
                    else:
                        if iteration >= 0 and \
                                (not self.scheduled_sampling or
                                 (self.scheduled_sampling and uniform(0.0, 1.0) < teacher_probability)):
                            assert i > 0
                            extra_action_key = 'actions' if self.output_size == 2 else 'stops'
                            extra_action = single_data_list[i - 1][extra_action_key].view(-1, 1, self.output_size)
                        else:
                            extra_action = single_output_list[-1]['output'].view(-1, 1, self.output_size)
                    single_output = self.subforward(d, decoder_hidden, extra_action)
                else:
                    single_output = self.subforward(d, decoder_hidden)
                decoder_hidden = single_output['decoder_hidden']
                single_output_list.append(single_output)
            return merge_output(single_output_list)
        else:
            return self.subforward(data, decoder_hidden)


class ModelIntegratedBase(ModelBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelBase.__init__(self, param)

        self.attention_type = param.attention_type
        if self.use_attention:
            self.attention_module = AttentionLuong(param)
        if self.use_attention and not self.decoder_rollout:
            raise ValueError('currently decoder_rollout is required when attention is used: {}, {}'.format(
                self.use_attention, self.decoder_rollout
            ))

        self.encoder = Encoder(param, num_words)
        self.decoder = Decoder(param, self.in_decoder)

    @property
    def use_attention(self):
        return self.attention_type != 'none' and self.decoder_type == 'gru'

    @property
    def encoder_hidden_size(self):
        return self.encoder.encoder_hidden_size

    def subforward(
            self,
            data: dict,
            encoder_hidden: torch.Tensor,
            decoder_hidden: torch.Tensor,
            extra_action: torch.Tensor = None):
        x = self.process(encoder_hidden, data)
        out_dict = self.decoder.forward(x, decoder_hidden, extra_action)
        out_dict['encoder_hidden'] = encoder_hidden
        return out_dict

    def forward(self, data: dict, encoder_hidden: torch.Tensor, decoder_hidden: torch.Tensor, iteration: int = -1):
        # encoder_output: batch, seq_len, num_directions * encoder_hidden_size
        # decoder_hidden: num_layers * num_directions, batch, decoder_hidden_size
        encoder_output, encoder_hidden = self.encoder.forward(data, encoder_hidden)
        batch_size, seq_len, size_dict = check_hidden_tensor_sizes(
            encoder_output, decoder_hidden, self.encoder_hidden_size, self.decoder_hidden_size)
        check_tensor_size(encoder_hidden, '1be', size_dict)

        if self.decoder_type == 'gru':
            teacher_probability = min(1.0, max(0.0, 1.0 - iteration / self.scheduled_iteration))
            single_data_list = split_batch(data)
            single_output_list = []
            for i, d in enumerate(single_data_list):
                if self.use_attention:
                    context, attention = self.attention_module.forward(encoder_output, decoder_hidden)  # 1be, bs
                else:
                    context, attention = encoder_hidden, None
                check_tensor_size(context, '1be', size_dict)
                if self.use_history:
                    if not single_output_list:
                        extra_action = torch.zeros((encoder_output.size(0), 1, self.output_size),
                                                   dtype=torch.float32, device=encoder_output.device)
                    else:
                        if iteration >= 0 and \
                                (not self.scheduled_sampling or
                                 (self.scheduled_sampling and uniform(0.0, 1.0) < teacher_probability)):
                            assert i > 0
                            if self.model_level == 'high':
                                extra_action_index = single_data_list[i - 1]['onehots'].view(-1, 1)
                                extra_action = torch.zeros(extra_action_index.size(0), self.output_size,
                                                           dtype=torch.float32, device=encoder_output.device)
                                extra_action.scatter_(1, extra_action_index, 1)
                                extra_action = extra_action.view(-1, 1, self.output_size)
                            elif self.model_level == 'low':
                                key = 'actions' if self.is_control else 'stops'
                                extra_action = single_data_list[i - 1][key].view(-1, 1, self.output_size)
                        else:
                            extra_action = single_output_list[-1]['output'].view(-1, 1, self.output_size)
                    single_output = self.subforward(d, context, decoder_hidden, extra_action)
                else:
                    single_output = self.subforward(d, context, decoder_hidden)
                single_output['attention'] = attention
                decoder_hidden = single_output['decoder_hidden']
                single_output_list.append(single_output)
            return merge_output(single_output_list)
        else:
            return self.subforward(data, encoder_hidden, decoder_hidden)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        raise NotImplementedError


class ModelIntegratedGatedAttention(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelIntegratedBase.__init__(self, param, num_words)

        self.image_processor = ConvLayers(self.channels)
        self.attn_linear = torch.nn.Linear(self.encoder_hidden_size, self.image_feat_channel)
        self.linear = torch.nn.Linear(self.image_vec_dim, self.out_linear)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        x_image_rep = self.image_processor.forward(data)
        assert x_instr_rep.size(1) == x_image_rep.size(0)
        batch_size, seq_len = x_image_rep.size(0), x_image_rep.size(1)
        # Get the attention vector from the instruction representation
        x_attention = torch.sigmoid(self.attn_linear(x_instr_rep))  # batch_size, image_channels
        x_attention = x_attention.view(batch_size, 1, self.image_feat_channel, 1, 1)
        x_attention = x_attention.expand(
            batch_size, seq_len, self.image_feat_channel, IMAGE_TENSOR_HEIGHT, IMAGE_TENSOR_WIDTH)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep * x_attention
        x = x.view(batch_size, seq_len, -1)  # batch_size, seq_len, 9216
        x = torch.nn.functional.elu(self.linear(x))  # batch_size, seq_len, in_decoder_channel
        return x


class ModelIntegratedAblationNMC(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        param.attention_type = 'none'
        ModelIntegratedBase.__init__(self, param, num_words)
        assert not self.use_attention

        self.image_processor = ConvLayers(self.channels)
        self.linear = torch.nn.Linear(self.image_vec_dim, self.out_linear)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        x_image_rep = self.image_processor.forward(data)
        assert x_instr_rep.size(1) == x_image_rep.size(0)
        batch_size, seq_len = x_image_rep.size(0), x_image_rep.size(1)
        x = x_image_rep.view(batch_size, seq_len, -1)  # batch_size, seq_len, 9216
        x = torch.nn.functional.elu(self.linear(x))  # batch_size, seq_len, in_decoder_channel
        return x


class ModelIntegratedAblationWithoutImage(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelIntegratedBase.__init__(self, param, num_words)
        self.linear = torch.nn.Linear(self.encoder_hidden_size, self.out_linear)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        batch_size, seq_len = x_instr_rep.size(1), data['images'].size(1)
        x_instr_rep = x_instr_rep.view(batch_size, 1, self.encoder_hidden_size)
        x_instr_rep = x_instr_rep.expand(batch_size, seq_len, self.encoder_hidden_size)
        x = torch.nn.functional.elu(self.linear(x_instr_rep))  # batch_size, seq_len, in_decoder_channel
        return x


class ModelIntegratedLanguageAttention(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelIntegratedBase.__init__(self, param, num_words)

        self.image_processor = ConvLayers(self.channels)
        self.attn_linear = torch.nn.Linear(self.image_vec_dim, self.encoder_hidden_size)
        self.linear = torch.nn.Linear(self.encoder_hidden_size, self.out_linear)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        x_image_rep = self.image_processor.forward(data)
        assert x_instr_rep.size(1) == x_image_rep.size(0)
        batch_size, seq_len = x_image_rep.size(0), x_image_rep.size(1)

        # Get the attention vector from the last image representation
        last_only = False
        if last_only:
            last_image_rep = x_image_rep[:, -1, :, :, :]
            x_attention = torch.sigmoid(self.attn_linear(last_image_rep.view(batch_size, -1)))  # B x E
            x = x_attention * x_instr_rep.view(batch_size, self.encoder_hidden_size)  # B x E
            x = torch.nn.functional.elu(self.linear(x))  # B X O
            x = x.view(batch_size, 1, -1).expand(batch_size, seq_len, self.out_linear)  # B x S x O
        else:
            xs = []
            for i in range(x_image_rep.size(1)):
                x_attention = torch.sigmoid(self.attn_linear(x_image_rep[:, i, :, :, :].view(batch_size, -1)))  # B x E
                xs.append(x_attention * x_instr_rep.view(batch_size, self.encoder_hidden_size))  # B x E
            x = torch.stack(xs, dim=1)  # B x S x E
            x = torch.nn.functional.elu(self.linear(x))  # B x S x O
        return x


class ModelIntegratedEarlyFusion(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelIntegratedBase.__init__(self, param, num_words)

        self.image_processor = EarlyFusionConvLayers(self.channels, self.encoder_hidden_size)
        self.linear = torch.nn.Linear(self.image_vec_dim, self.out_linear)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        x_image_rep = self.image_processor.forward(data, x_instr_rep)
        batch_size, seq_len = x_image_rep.size(0), x_image_rep.size(1)
        x = x_image_rep.view(batch_size, seq_len, -1)
        x = torch.nn.functional.elu(self.linear(x))  # batch_size, seq_len, in_decoder_channel
        return x


class ModelIntegratedLateFusion(ModelIntegratedBase):
    def __init__(self, param: Parameter, num_words: int):
        ModelIntegratedBase.__init__(self, param, num_words)

        self.out_image_rep = self.out_linear // 2
        self.out_instr_rep = self.out_linear // 2

        self.image_processor = ConvLayers(self.channels)
        self.instr_linear = torch.nn.Linear(self.encoder_hidden_size, self.out_instr_rep)
        self.image_linear = torch.nn.Linear(self.image_vec_dim, self.out_image_rep)

    def process(self, x_instr_rep: torch.Tensor, data: dict) -> torch.Tensor:
        """
        :param x_instr_rep: 1 x B x E
        :param data: data dict
        :return: representation just before the decoder B x S x ID
        """
        x_image_rep = self.image_processor.forward(data)
        assert x_instr_rep.size(1) == x_image_rep.size(0)
        batch_size, seq_len = x_image_rep.size(0), x_image_rep.size(1)

        x_image_rep = x_image_rep.view(batch_size, seq_len, -1)
        x_image_rep = torch.nn.functional.elu(self.image_linear(x_image_rep))
        x_instr_rep = torch.nn.functional.elu(self.instr_linear(x_instr_rep))
        x_instr_rep = x_instr_rep.view(batch_size, 1, self.out_instr_rep)
        x_instr_rep = x_instr_rep.expand(batch_size, seq_len, self.out_instr_rep)
        x = torch.cat((x_image_rep, x_instr_rep), -1)
        return x


model_dict = {
    'ga': ModelIntegratedGatedAttention,
    'ef': ModelIntegratedEarlyFusion,
    'lf': ModelIntegratedLateFusion,
    'la': ModelIntegratedLanguageAttention,
    'nmc': ModelIntegratedAblationNMC,
    'wi': ModelIntegratedAblationWithoutImage,
    'se': ModelSeparated
}


def fetch_agent(param: Parameter):
    if param.model_type not in model_dict:
        raise KeyError('invalid model type was given {}'.format(param.model_type))
    model = model_dict[param.model_type](param, param.encoder_num_words)
    model = model.to(param.device_type)
    return model


def fetch_hidden_dimensions(param: Parameter):
    encoder_size = torch.Size([param.encoder_hidden_layers, param.batch_size, param.encoder_hidden_size])
    decoder_size = torch.Size([param.decoder_hidden_layers, param.batch_size, param.decoder_hidden_size])
    return encoder_size, decoder_size


def init_hidden_states(param: Parameter):
    encoder_hidden = torch.zeros(param.encoder_hidden_layers, param.batch_size, param.encoder_hidden_size,
                                 device=param.device_type, requires_grad=False)
    decoder_hidden = torch.zeros(param.decoder_hidden_layers, param.batch_size, param.decoder_hidden_size,
                                 device=param.device_type, requires_grad=False)
    return encoder_hidden, decoder_hidden


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path: str, device_str: str):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        with tf.device('/device:{}'.format(device_str)):
            graph_def = None
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break
            tar_file.close()

            if graph_def is None:
                raise RuntimeError('Cannot find inference graph in tar archive.')

            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')

        self.config = tf.ConfigProto()
        self.config.allow_soft_placement = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.08
        self.sess = tf.Session(graph=self.graph, config=self.config)

    def _run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run(self, img: np.ndarray) -> np.ndarray:
        img_ = Image.fromarray(img[:, :, ::-1])
        _, seg = self._run(img_)
        seg = (cv2.resize(seg, (200, 88), interpolation=cv2.INTER_NEAREST) == 0).astype(dtype=np.uint8) * 255
        return seg


def prepare_deeplab_model(index: int = 0):
    model_dir = mkdir_if_not_exists(Path('./resources/models'))
    model_path = model_dir / 'deeplab_model.tar.gz'
    if not model_path.exists():
        urllib.request.urlretrieve(
            'http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
            model_path)
    return DeepLabModel(str(model_path), 'cuda:{}'.format(index))
