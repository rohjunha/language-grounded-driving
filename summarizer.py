import json
import re
from argparse import ArgumentParser
from itertools import chain, groupby
from math import sqrt
from multiprocessing.pool import ThreadPool
from operator import attrgetter
from pathlib import Path
from typing import List

from data.types import DriveDataFrame
from evaluator import load_evaluation_dataset
from parameter import Parameter
from util.common import get_logger, timethis
from util.directory import EvaluationDirectory, fetch_evaluation_dir, fetch_evaluation_summary_dir
from util.serialize import parse_bool

logger = get_logger(__name__)


class EvaluationUnitInfo:
    """(atomic) evaluation information class with data and model information"""

    def __init__(self, data_keyword: str, exp_index: int, exp_name: str, exp_step: int, data_traj_index: int):
        self.data_keyword = data_keyword
        self.exp_index = exp_index
        self.exp_name = exp_name
        self.exp_step = exp_step
        self.data_traj_index = data_traj_index

    def __str__(self):
        return 'EvaluationUnitInfo {}:{}:{}:step{}:traj{}'.format(
            self.data_keyword, self.exp_index, self.exp_name, self.exp_step, self.data_traj_index)


class EvaluationInfoBase:
    def __init__(self, info: EvaluationUnitInfo):
        self.info = info

    @property
    def exp_index(self):
        return self.info.exp_index

    @property
    def exp_name(self):
        return self.info.exp_name

    @property
    def exp_step(self):
        return self.info.exp_step

    @property
    def data_traj_index(self):
        return self.info.data_traj_index

    @property
    def data_keyword(self):
        return self.info.data_keyword


class EvaluationTrajectory(EvaluationInfoBase):
    def __init__(self, info: EvaluationUnitInfo, points: List[DriveDataFrame], collided: bool):
        EvaluationInfoBase.__init__(self, info)
        self.points = points
        self.collided = collided
        if not self.points:
            raise ValueError('points was empty')

    def __str__(self):
        return 'EvaluationTrajectory {}, collided {}, len {}\n'.format(self.info, self.collided, len(self.points))

    def __repr__(self):
        return str(self)


class EvaluationTrajectoryGroup(EvaluationInfoBase):
    def __init__(self, info: EvaluationUnitInfo, trajectories: List[EvaluationTrajectory]):
        EvaluationInfoBase.__init__(self, info)
        self.trajectories = trajectories
        self.trajectory_indices = list(map(attrgetter('data_traj_index'), self.trajectories))

    def __str__(self):
        return '{}, len {}'.format(str(self.info), len(self))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.trajectories)


def load_expert_trajectory(data_name: str, info_name: str, data_keyword: str) -> EvaluationTrajectoryGroup:
    param = Parameter()
    param.model_level = 'high'
    param.eval_keyword = data_keyword
    param.eval_data_name = data_name
    param.eval_info_name = info_name

    data_frame_list, sentences = load_evaluation_dataset(param)
    info_list = [EvaluationUnitInfo(data_keyword, -1, '', -1, i) for i in range(len(data_frame_list))]
    trajectories = [EvaluationTrajectory(i, d, False) for i, d in zip(info_list, data_frame_list)]
    info = EvaluationUnitInfo(data_keyword, -1, '', -1, -1)
    traj_group = EvaluationTrajectoryGroup(info, trajectories)
    return traj_group


def load_model_single_trajectory(eval_dir: EvaluationDirectory, traj_index: int) -> EvaluationTrajectory:
    try:
        with open(str(eval_dir.state_path(traj_index)), 'r') as file:
            data = json.load(file)
    except:
        raise FileNotFoundError('failed to load {}'.format(eval_dir.state_path(traj_index)))
    collided = False if 'collided' not in data else parse_bool(data['collided'])
    data_frames = [DriveDataFrame.load_from_str(f) for f in data['data_frames']]
    info = EvaluationUnitInfo(
        eval_dir.data_keyword, eval_dir.exp_index, eval_dir.exp_name, eval_dir.exp_step, traj_index)
    return EvaluationTrajectory(info, data_frames, collided)


def load_model_single_trajectory_group(eval_dir: EvaluationDirectory) -> EvaluationTrajectoryGroup:
    group_info = EvaluationUnitInfo(eval_dir.data_keyword, eval_dir.exp_index, eval_dir.exp_name, eval_dir.exp_step, -1)
    trajectories = []
    for traj_index in eval_dir.traj_indices_from_state_dir():
        trajectories.append(load_model_single_trajectory(eval_dir, traj_index))
    return EvaluationTrajectoryGroup(group_info, trajectories)


def evaluation_directory_from_path(eval_path: Path) -> EvaluationDirectory:
    pattern = 'exp([\d]+)\/([\w-]+)\/step([\d]+)/([\w,]+)'
    res = re.findall(pattern, str(eval_path))
    if not res or len(res[0]) < 4:
        return None
    else:
        exp_index = int(res[0][0])
        exp_name = res[0][1]
        exp_step = int(res[0][2])
        data_keyword = res[0][3]
        return EvaluationDirectory(exp_index, exp_name, exp_step, data_keyword)


def load_model_trajectory_groups(exp_index: int, data_keyword: str, exp_name: str = '') -> List[
    EvaluationTrajectoryGroup]:
    logger.info(data_keyword)
    root_dir = fetch_evaluation_dir() / 'exp{:02d}'.format(exp_index)
    if not root_dir.exists():
        raise ValueError('root directory {} does not exist'.format(root_dir))
    exp_dirs = sorted(list(filter(lambda x: x.is_dir(), root_dir.glob('*'))))
    if exp_name:
        exp_dirs = list(filter(lambda x: exp_name == x.stem, exp_dirs))
    step_dirs = [list(filter(lambda x: x.is_dir(), exp_dir.glob('*'))) for exp_dir in exp_dirs]
    step_dirs = list(chain.from_iterable(step_dirs))
    data_dirs = list(filter(lambda x: x.exists(), [d / data_keyword for d in step_dirs]))
    eval_dirs = list(map(evaluation_directory_from_path, data_dirs))
    eval_dirs = list(filter(lambda x: x is not None, eval_dirs))

    pool = ThreadPool(8)
    groups = pool.map(load_model_single_trajectory_group, eval_dirs)
    return groups


def sort_trajectories(expert_trajectories: List[EvaluationTrajectory], model_trajectories: List[EvaluationTrajectory]):
    expert_group = groupby(expert_trajectories, lambda x: x.data_keyword)
    model_group = groupby(model_trajectories, lambda x: (x.data_keyword, x.exp_index, x.exp_name, x.exp_step))
    return expert_group, model_group


EVAL_ENDPOINT_TH = 10.0
EVAL_TRAJACTORY_TH = 10.0


def compute_dist(p1, p2):
    x1 = p1.state.transform.location.x
    y1 = p1.state.transform.location.y
    x2 = p2.state.transform.location.x
    y2 = p2.state.transform.location.y
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class TrajectoryComparator(EvaluationInfoBase):
    """evaluate a single pair of trajectories of the same index"""

    def __init__(self, expert: EvaluationTrajectory, model: EvaluationTrajectory):
        assert expert.data_keyword == model.data_keyword
        assert expert.data_traj_index == model.data_traj_index
        EvaluationInfoBase.__init__(self, model.info)
        self.valid = True
        self.expert = expert
        self.model = model

        self.endpoint_dist = min([compute_dist(self.expert.points[-1], src) for src in self.model.points])
        self.endpoint_success = self.endpoint_dist < EVAL_ENDPOINT_TH
        self.trajectory_dists = [min([compute_dist(dst, src) for src in self.model.points]) for dst in
                                 self.expert.points]
        self.trajectory_dist_avg = sum(self.trajectory_dists) / len(self.trajectory_dists)
        trajectory_successes = list(map(lambda d: d < EVAL_TRAJACTORY_TH, self.trajectory_dists))
        self.trajectory_rate = float(sum(trajectory_successes)) / len(self.trajectory_dists)
        self.trajectory_success = all(trajectory_successes)
        self.collision_success = not self.model.collided
        self.all_success = self.collision_success and self.trajectory_success and self.endpoint_success

    def to_dict(self):
        return {
            'info': str(self.info),
            'endpoint_dist': self.endpoint_dist,
            'endpoint_success': self.endpoint_success,
            'trajectory_dists': self.trajectory_dists,
            'trajectory_dist_avg': self.trajectory_dist_avg,
            'trajectory_rate': self.trajectory_rate,
            'trajectory_success': self.trajectory_success,
            'collision_success': self.collision_success,
            'all_success': self.all_success,
        }

    @classmethod
    def header(cls):
        return '{traj:>4s}, {e1:>6s}, {e2:>5s}, {t1:>5s}, {t2:>5s}, {c1:>5s}, {a1:>5s}'.format(
            traj='traj', e1='edist', e2='esucc', t1='trate', t2='tsucc', c1='csucc', a1='asucc')

    def __str__(self):
        return '{info:>4s}, {e1:>6.3f}, {e2:>5s}, {t1:>5.3f}, {t2:>5s}, {c1:>5s}, {a1:>5s}'.format(
            info='{:02d}'.format(self.data_traj_index),
            e1=self.endpoint_dist, e2=str(self.endpoint_success),
            t1=self.trajectory_rate, t2=str(self.trajectory_success),
            c1=str(self.collision_success),
            a1=str(self.all_success))

    def __repr__(self):
        return str(self)


class TrajectoryGroupComparator(EvaluationInfoBase, EvaluationDirectory):
    """collapse multiple evaluations of the same step"""

    def __init__(
            self,
            expert_group: EvaluationTrajectoryGroup,
            model_group: EvaluationTrajectoryGroup,
            overwrite: bool):
        assert len(expert_group.trajectory_indices) == len(expert_group.trajectories)
        assert len(model_group.trajectory_indices) == len(model_group.trajectories)

        EvaluationInfoBase.__init__(self, model_group.info)
        self.eval_dir = EvaluationDirectory(self.exp_index, self.exp_name, self.exp_step, self.data_keyword)
        if self.has_summary:
            if overwrite:
                self.summary_path.unlink()
            else:
                logger.info('skipped the summarization: {}'.format(self.summary_path))
                return

        original_indices = expert_group.trajectory_indices
        common_indices = sorted(list(set(expert_group.trajectory_indices).intersection(
            set(model_group.trajectory_indices))))
        if len(common_indices) < len(original_indices):
            logger.info('not all of the evaluation trajectories were evaluated at {}'.format(self.info))
        if not common_indices:
            logger.error('no valid trajectories were found in the model trajectory {}'.format(model_group.info))
            return
        self.indices = common_indices

        for index in self.indices:
            if index >= len(expert_group.trajectories) or index >= len(model_group.trajectories):
                logger.error(
                    'invalid trajectory {}, {}'.format(len(expert_group.trajectories), len(model_group.trajectories)))
                return

        self.expert_trajectories = [expert_group.trajectories[i] for i in self.indices]
        self.model_trajectories = [model_group.trajectories[i] for i in self.indices]
        self.metrics = [TrajectoryComparator(e, m) for e, m in zip(self.expert_trajectories, self.model_trajectories)]

        self.endpoint_dist_avg = sum(map(attrgetter('endpoint_dist'), self.metrics)) / len(self)
        self.endpoint_success_rate = sum(map(attrgetter('endpoint_success'), self.metrics)) / len(self)
        self.trajectory_dist_avg_avg = sum(map(attrgetter('trajectory_dist_avg'), self.metrics)) / len(self)
        self.trajectory_dist_rate_avg = sum(map(attrgetter('trajectory_rate'), self.metrics)) / len(self)
        self.trajectory_success_rate = sum(map(attrgetter('trajectory_success'), self.metrics)) / len(self)
        self.collision_success_rate = sum(map(attrgetter('collision_success'), self.metrics)) / len(self)
        self.all_success_rate = sum(map(attrgetter('all_success'), self.metrics)) / len(self)

        logger.info(self)
        # self.send_files()
        self.write_summary()

    def to_dict(self):
        return {
            'endpoint_dist_avg': self.endpoint_dist_avg,
            'endpoint_success_rate': self.endpoint_success_rate,
            'trajectory_dist_avg_avg': self.trajectory_dist_avg_avg,
            'trajectory_dist_rate_avg': self.trajectory_dist_rate_avg,
            'trajectory_success_rate': self.trajectory_success_rate,
            'collision_success_rate': self.collision_success_rate,
            'all_success_rate': self.all_success_rate,
            'num_trajectories': len(self)
        }

    def __len__(self):
        return len(self.metrics)

    def __str__(self):
        str_multiple = '''
{exp_info}
{s1:<{indent_words}}: {eval_info:>19s}
{s2:<{indent_words}}: {e1:>12.3f}, {e2:>5.3f}
{s3:<{indent_words}}: {t1:>5.3f}, {t2:>5.3f}, {t3:>5.3f}
{s4:<{indent_words}}: {c1:>19.3f}
{s5:<{indent_words}}: {a1:>19.3f}
'''.format(
            indent_words=60, exp_info=self.info,
            s1='data information',
            s2='endpoint (dist_avg, success_rate)',
            s3='trajectory (dist_avg_avg, dist_rate_avg, success_rate)',
            s4='collision (success_rate)',
            s5='all (success_rate)',
            eval_info='{}, {}'.format(len(self), 'position (3d)'),
            e1=self.endpoint_dist_avg,
            e2=self.endpoint_success_rate,
            t1=self.trajectory_dist_avg_avg,
            t2=self.trajectory_dist_rate_avg,
            t3=self.trajectory_success_rate,
            c1=self.collision_success_rate,
            a1=self.all_success_rate)

        str_header = TrajectoryComparator.header()
        str_singles = list(map(lambda x: str(x), filter(lambda m: m.valid, self.metrics)))
        return '\n'.join([str_multiple, str_header] + str_singles)

    @property
    def summary_path(self):
        return self.eval_dir.summary_path

    @property
    def has_summary(self):
        return self.summary_path.exists()

    def write_summary(self):
        summary = {
            'group': self.to_dict(),
            'individual': list(map(lambda x: x.to_dict(), filter(lambda x: x.valid, self.metrics)))
        }
        with open(str(self.summary_path), 'w') as file:
            json.dump(summary, file, indent=2)


class EvaluationRow:
    def __init__(self, exp_name: str, exp_index: int, step: int, values: list):
        self.exp_name = exp_name
        self.exp_index = exp_index
        self.step = step
        self.values = values
        self.ratios = [0.5 for _ in range(len(self))]
        self.ascendings = [True for _ in range(len(self))]
        for i in [0, 2]:
            self.ascendings[i] = False

    @property
    def to_list(self):
        return [self.exp_name, str(self.exp_index), str(self.step)] + \
               ['{:5.3f}'.format(v) if isinstance(v, float) else str(v) for v in self.values]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return ', '.join(self.to_list)

    def __repr__(self):
        return str(self)


def sorted_rows(rows: List[EvaluationRow]):
    return list(sorted(rows, key=lambda x: (x.exp_index, x.exp_name, x.step)))


def summarize(
        exp_index: int,
        data_name: str,
        info_name: str,
        data_keyword: str,
        overwrite: bool,
        exp_name: str = '') -> List[EvaluationRow]:
    rows = []
    expert_group = load_expert_trajectory(data_name, info_name, data_keyword)
    print(len(expert_group))

    def compare_group(model_group: EvaluationTrajectoryGroup):
        return TrajectoryGroupComparator(expert_group, model_group, overwrite)

    model_group_list = load_model_trajectory_groups(exp_index, data_keyword, exp_name)
    pool = ThreadPool(8)
    group_comparators = pool.map(compare_group, model_group_list)
    group_comparators = list(filter(lambda x: x.has_summary, group_comparators))
    group_comparators = sorted(group_comparators, key=lambda x: (x.exp_index, x.exp_name, x.exp_step))
    for gc in group_comparators:
        logger.info('{}:{}:{:>6d}'.format(gc.exp_index, gc.exp_name, gc.exp_step))
        with open(str(gc.summary_path), 'r') as file:
            data = json.load(file)
        rows.append(
            EvaluationRow(gc.exp_name, gc.exp_index, gc.exp_step, [v for k, v in data['group'].items()]))
    return sorted_rows(rows)


def update_ratios(rows: List[EvaluationRow]):
    num_col = len(rows[0])
    for col in range(num_col):
        col_values = [r.values[col] for r in rows]
        min_value = min(col_values)
        max_value = max(col_values)
        if min_value == max_value:
            continue
        ratios = [(v - min_value) / (max_value - min_value) for v in col_values]
        for index, ratio in enumerate(ratios):
            rows[index].ratios[col] = ratio


def fetch_table_header():
    return '''
<tr class="table-active">
    <th scope="col" rowspan="2">experiment-name</th>
    <th scope="col" rowspan="2">index</th>
    <th scope="col" rowspan="2">iter</th>
    <th scope="col" colspan="2">endpoint</th>
    <th scope="col" colspan="3">trajectory</th>
    <th scope="col">cols.</th>
    <th scope="col">all</th>
    <th scope="col" rowspan="2">#traj</th>
</tr>
<tr class="table-active">
    <th scope="col">avg</th>
    <th scope="col">rate</th>
    <th scope="col">avg</th>
    <th scope="col">point</th>
    <th scope="col">rate</th>
    <th scope="col">rate</th>
    <th scope="col">rate</th>
</tr>'''


def fetch_document_header():
    return '''
<head>
    <link rel="stylesheet" href="https://bootswatch.com/4/flatly/bootstrap.min.css" />
    <style type="text/css">
        .tr-border {
            border-top: 1px solid #000;
        }
    </style>
</head>
'''


hexstr_from_int = {i: '{:02x}'.format(i) for i in range(256)}
int_from_hexstr = {v: k for k, v in hexstr_from_int.items()}


def indices_from_hexcode(hexcode: str) -> List[int]:
    result = re.findall('([0-9a-fA-F]+)', hexcode)
    if not result:
        raise ValueError('could not found hex value in the string {}'.format(hexcode))
    code = result[0]
    if len(code) % 2 != 0:
        raise ValueError('invalid length of the code {}'.format(len(code)))
    return [int_from_hexstr[code[i:i + 2]] for i in range(0, len(code), 2)]


def hexcode_from_indices(indices: List[int]) -> str:
    return ''.join([hexstr_from_int[i] for i in indices])


RED = 'e67c73'
WHITE = 'ffffff'
GREEN = '57bb8a'


def interp_color_indices(ratio: float, c1: List[int], c2: List[int], c3: List[int]):
    def interp(ratio: float, l1: List[int], l2: List[int]):
        return [round((1.0 - ratio) * v1 + ratio * v2) for v1, v2 in zip(l1, l2)]

    return interp(2.0 * ratio, c1, c2) if ratio < 0.5 else interp(2.0 * (ratio - 0.5), c2, c3)


def interp_color_codes(ratio: float, s1: str, s2: str, s3: str):
    return hexcode_from_indices(
        interp_color_indices(ratio, indices_from_hexcode(s1), indices_from_hexcode(s2), indices_from_hexcode(s3)))


def fetch_interp_color(ratio: float, ascending: bool = True) -> str:
    c1 = RED if ascending else GREEN
    c2 = GREEN if ascending else RED
    return interp_color_codes(ratio, c1, WHITE, c2)


def write_th(value):
    return '<th>{}</th>'.format(str(value))


def write_tds(data: EvaluationRow) -> List[str]:
    return ['<td style="background: #{color}">{value}</td>'.format(
        value='{:5.3f}'.format(v) if isinstance(v, float) else str(v),
        color=fetch_interp_color(r, a))
        for v, r, a in zip(data.values, data.ratios, data.ascendings)]


def write_tr(data: EvaluationRow, border: bool):
    tr_header = '<tr class="tr-border">' if border else '<tr>'
    return '{}{}</tr>'.format(
        tr_header,
        '\n'.join([write_th(data.exp_name), write_th(data.exp_index), write_th(data.step)] + write_tds(data)))


def summarize_by_topic(
        exp_index: int,
        data_name: str,
        info_name: str,
        keyword: str,
        overwrite: bool,
        output_filename: str = ''):
    data = summarize(exp_index, data_name, info_name, keyword, overwrite)
    if not output_filename:
        return
    update_ratios(data)

    # find the first row indices of the same experiment names, and set border lines
    borders = [False]
    for i in range(len(data) - 1):
        borders.append(data[i].exp_name != data[i + 1].exp_name)

    dst_name = '{}.html'.format(output_filename)
    page_path = fetch_evaluation_summary_dir() / dst_name
    table_header = fetch_table_header()
    trs = [write_tr(d, b) for d, b in zip(data, borders)]
    body = '<body><div class="table-responsive"><table class="table-sm">{}\n{}</table></div></body>'.format(
        table_header, '\n'.join(trs))
    document = '<html>\n{}\n{}\n</html>'.format(fetch_document_header(), body)

    with open(str(page_path), 'w') as file:
        file.write(document)


def summarize_by_experiment(
        exp_index: int,
        exp_name: str,
        data_name: str,
        info_name: str,
        overwrite: bool,
        output_filename: str = ''):
    if not info_name.endswith('40'):
        keywords = [
            'left', 'right', 'straight',
            'left,left', 'left,right', 'left,straight',
            'right,left', 'right,right', 'right,straight', 'straight,straight',
            'firstleft', 'firstright', 'secondleft', 'secondright']
    else:
        keywords = ['extrastraight', 'extraleft', 'extraright', 'extrastraight,extrastraight', 'extrastraight,right',
                    'right,extrastraight', 'extrastraight,extraleft', 'extraleft,right', 'right,extraleft',
                    'extrastraight,left', 'left,extraright', 'extraright,extraright', 'extraright,left', 'right,extraright',
                    'extraright,extrastraight', 'left,extrastraight', 'left,extraleft', 'extraleft,extraleft',
                    'extrastraight,extraright', 'extraleft,extrastraight']
        if 'town2' in exp_name:
            keywords += ['extraleft,left']

    data = []
    for keyword in keywords:
        item = summarize(exp_index, data_name, info_name, keyword, overwrite, exp_name)
        assert len(item) == 1
        data.append(item[0])
    update_ratios(data)

    # find the first row indices of the same experiment names, and set border lines
    borders = [False]
    for i in range(len(data) - 1):
        borders.append(data[i].exp_name != data[i + 1].exp_name)

    dst_name = '{}.html'.format(output_filename)
    page_path = fetch_evaluation_summary_dir() / dst_name
    table_header = fetch_table_header()
    trs = [write_tr(d, b) for d, b in zip(data, borders)]
    body = '<body><div class="table-responsive"><table class="table-sm">{}\n{}</table></div></body>'.format(
        table_header, '\n'.join(trs))
    document = '<html>\n{}\n{}\n</html>'.format(fetch_document_header(), body)

    with open(str(page_path), 'w') as file:
        file.write(document)


@timethis
def main():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('keyword', type=str)
    parser.add_argument('town', type=int)
    parser.add_argument('--exp_index', type=int, default=40)
    parser.add_argument('--data_name', type=str, default='semantic1')
    parser.add_argument('--info_name', type=str, default='semantic1-v37')
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.filename = args.keyword.replace(' ', '-')
    args.data_name = 'semantic{}'.format(args.town)
    args.info_name = 'semantic{}-v37'.format(args.town)
    if args.mode == 'topic':
        summarize_by_topic(args.exp_index, args.data_name, args.info_name, args.keyword, args.overwrite, args.filename)
    elif args.mode == 'experiment':
        summarize_by_experiment(args.exp_index, args.keyword, args.data_name, args.info_name, args.overwrite,
                                args.filename)
    else:
        raise ValueError('invalid mode {}'.format(args.mode))


if __name__ == '__main__':
    main()
