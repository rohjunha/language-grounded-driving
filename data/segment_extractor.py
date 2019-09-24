from collections import defaultdict
from itertools import chain
from math import sqrt, atan2, pi
from operator import attrgetter, itemgetter
from typing import List, Tuple, Dict, Any

from data.types import FrameInfo, SegmentInfo, DriveDataFrame
from util.common import unique_with_islices, get_logger

logger = get_logger(__name__)


def angle_diff(a1: float, a2: float) -> float:
    a = a2 - a1
    da = -360 if a > 180 else (360 if a < -180 else 0)
    return a + da


def dict_from_segment_info(s: SegmentInfo, has_stop: bool) -> Dict[str, Any]:
    return {
        'option': s.option.upper(),
        'indices': s.index_range,
        'split_index': s.split_index,
        'has_stop': has_stop
    }


class SegmentExtractor:
    def __init__(
            self,
            infos: List[FrameInfo],
            unique: bool = False,
            augment: bool = True):
        self.verbose = False
        self.infos = infos
        self.unique = unique
        self.augment = augment

        self.angle_threshold = 10
        self.cluster_dist_threshold = 30
        self.cluster_point_list: List[Tuple[float, float]] = []

        self.segment_list: List[SegmentInfo] = []
        self.intersection_dist_enter_threshold = 30
        self.intersection_dist_exit_threshold = 30
        self.intersection_split_threshold = 10
        self.lanefollow_dist_threshold = 5
        self.lanefollow_split_threshold_long = 18

        self.segment_length_threshold = 60
        self.inner_radius = 15
        self.outer_radius = 25
        self.goal_radius = 7

    def angle_from_segment(self, index_range: Tuple[int, int]) -> float:
        i1, i2 = index_range[0], index_range[1] - 1
        t1, t2 = self.infos[i1].theta, self.infos[i2].theta
        return angle_diff(t1, t2)

    def determine_action_from_angle(self, index_range: Tuple[int, int]) -> str:
        angle = self.angle_from_segment(index_range)
        if abs(angle) < self.angle_threshold:
            return 'straight'
        elif angle >= self.angle_threshold:
            return 'right'
        else:
            return 'left'

    def compute_trajectory_length(self, index_range) -> float:
        ps = [(self.infos[i].x, self.infos[i].y) for i in range(*index_range)]
        return sum([sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for (x1, y1), (x2, y2) in zip(ps[:-1], ps[1:])])

    def update_label_from_raw_data(self) -> None:
        inter_list = list(map(attrgetter('is_intersection'), self.infos))
        option_list = list(map(attrgetter('option'), self.infos))

        # update is_intersection if the option is action-related
        update_intersection_indices = list(map(itemgetter(0), filter(
            lambda x: x[1][1] != 'lanefollow' and not x[1][0], enumerate(zip(inter_list, option_list)))))
        for i in update_intersection_indices:
            self.infos[i].is_intersection = True
            inter_list[i] = True
        inter_index_list = unique_with_islices(inter_list)

        # update lanefollow labels with intersection to neighboring action labels
        for intersection, index_range in inter_index_list:
            if intersection:
                option_set = set([self.infos[j].option for j in range(*index_range)])
                if len(option_set) > 1:
                    action_options = list(filter(lambda x: x != 'lanefollow', option_set))
                    assert len(action_options) == 1
                    action_option = action_options[0]
                    action_update_indices = list(
                        filter(lambda j: self.infos[j].option == 'lanefollow', range(*index_range)))
                    for j in action_update_indices:
                        self.infos[j].option = action_option
                        assert len(self.infos[j].possible_options) == 1
                        self.infos[j].possible_options = [action_option]
                        option_list[j] = action_option

        # merge two intersection segments with specific conditions: lanefollow followed by action, similar dists
        option_inter_index_list = unique_with_islices(zip(option_list, inter_list))
        for i, ((option, intersection), index_range) in enumerate(option_inter_index_list):
            if intersection and option == 'lanefollow':
                if i + 2 < len(option_inter_index_list):
                    action_orig = option_inter_index_list[i + 2][0][0]
                    inter_range = option_inter_index_list[i + 1][1]
                    new_index_range = index_range[0], option_inter_index_list[i + 2][1][1]
                    action_pred = self.determine_action_from_angle(new_index_range)
                    dist_curr = self.compute_trajectory_length(index_range)
                    dist_new = self.compute_trajectory_length(new_index_range)
                    dist_ratio = dist_new / dist_curr
                    if dist_ratio < 3.5 and action_orig == action_pred:
                        extend_range = index_range[0], inter_range[1]
                        for j in range(*extend_range):
                            self.infos[j].option = action_orig
                            self.infos[j].is_intersection = True
                            assert len(self.infos[j].possible_options) == 1 and \
                                   self.infos[j].possible_options[0] == 'lanefollow'
                            self.infos[j].possible_options = [action_orig]
                            option_list[j] = action_orig
                            inter_list[j] = True

        # make possible options unique
        for i in range(len(self.infos)):
            if not self.infos[i].possible_options:
                print('error at {}!'.format(i))
            if len(self.infos[i].possible_options) < 2:
                continue
            self.infos[i].possible_options = list(set(self.infos[i].possible_options))

        # update labels based on angles
        option_inter_index_list = unique_with_islices(zip(option_list, inter_list))
        for i, ((option, intersection), index_range) in enumerate(option_inter_index_list):
            option_pred = self.determine_action_from_angle(index_range)
            if intersection and option != option_pred:
                angle = self.angle_from_segment(index_range)
                if option_pred == 'left' and -115 <= angle <= -65 or option_pred == 'right' and 65 <= angle <= 115:
                    for j in range(*index_range):
                        self.infos[j].option = option_pred
                        if len(self.infos[j].possible_options) == 1:
                            self.infos[j].possible_options = [option_pred]
                        else:
                            self.infos[j].possible_options = \
                                list(
                                    set(list(filter(lambda x: x != option, self.infos[j].possible_options)) + [
                                        option_pred]))
                        option_list[j] = option_pred

        # check integrity and make a segment_list
        option_inter_index_list = unique_with_islices(zip(option_list, inter_list))
        segment_list = []
        for i, ((option, intersection), index_range) in enumerate(option_inter_index_list):
            option_list, possible_option_list = zip(
                *[(self.infos[j].option, self.infos[j].possible_options) for j in range(*index_range)])
            possible_option_list = chain.from_iterable(possible_option_list)
            option_list = list(set(option_list))
            possible_option_list = list(set(possible_option_list))
            assert len(option_list) == 1 and option == option_list[0]
            assert option in possible_option_list
            # logger.info((i, index_range, intersection, option, possible_option_list))
            segment_list.append(SegmentInfo(index_range, intersection, option, possible_option_list, True))

        for i in range(10):
            logger.info(segment_list[i].index_range)
        self.segment_list = segment_list

    def compute_dist_from_index(self, i1: int, i2: int) -> float:
        x1, y1, x2, y2 = self.infos[i1].x, self.infos[i1].y, self.infos[i2].x, self.infos[i2].y
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def find_intersection_points(self) -> None:
        intersection_mid_list = list(
            map(attrgetter('mid_index'), filter(attrgetter('is_intersection'), self.segment_list)))
        mid_cluster_list = []
        cluster_index_dict = dict()

        for qry in intersection_mid_list:
            valid = True
            for ref in mid_cluster_list:
                if self.compute_dist_from_index(qry, ref) < self.cluster_dist_threshold:
                    valid = False
                    cluster_index_dict[qry] = ref
                    break
            if valid:
                mid_cluster_list.append(qry)
                cluster_index_dict[qry] = qry

        clustered_index_list_dict = defaultdict(list)
        for qry, ref in cluster_index_dict.items():
            clustered_index_list_dict[ref].append(qry)
        cluster_point_list = []
        for ref, indices in clustered_index_list_dict.items():
            xs, ys = zip(*[(self.infos[i].x, self.infos[i].y) for i in indices])
            cluster_point_list.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        if self.verbose:
            import matplotlib.pyplot as plt
            xs, ys = zip(*list(map(lambda x: (x.x, x.y), self.infos)))
            plt.plot(xs, ys, 'C1-')
            for x, y in cluster_point_list:
                circle = plt.Circle((x, y), self.cluster_dist_threshold, color='r', fill=False)
                plt.gcf().gca().add_artist(circle)
            plt.show()

        self.cluster_point_list = cluster_point_list

    def dist_to_cluster(self, ci: int, si: int) -> float:
        x1, y1 = self.cluster_point_list[ci]
        x2, y2 = self.infos[si].x, self.infos[si].y
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # update segments based on the distance to the cluster and delete the last (partial) segment
    def find_closest_cluster_index(self, query_index: int) -> int:
        cluster_dists = [(c, self.dist_to_cluster(c, query_index)) for c in range(len(self.cluster_point_list))]
        cluster_index, best_dist = sorted(cluster_dists, key=lambda x: x[1])[0]
        return cluster_index

    def extend_intersection_segment(self, i: int) -> None:
        s = self.segment_list[i]
        assert s.is_intersection
        cluster_index = self.find_closest_cluster_index(s.mid_index)
        min_index = self.segment_list[i - 1].index_range[0] if i > 0 else s.index_range[0]
        max_index = self.segment_list[i + 1].index_range[1] if i + 1 < len(self.segment_list) else s.index_range[1]
        left_index = s.mid_index
        right_index = s.mid_index
        split_index = -1
        while left_index > min_index and \
                self.dist_to_cluster(cluster_index, left_index) < self.intersection_dist_enter_threshold:
            left_index -= 1
        while right_index < max_index:
            dist = self.dist_to_cluster(cluster_index, right_index)
            if dist > self.intersection_dist_exit_threshold:
                break
            if dist < self.intersection_split_threshold:
                split_index = right_index
            right_index += 1
        self.segment_list[i].index_range = left_index, right_index
        self.segment_list[i].split_index = split_index
        assert left_index < split_index < right_index

    def extend_lanefollow_segment(self, i: int) -> None:
        if i + 1 >= len(self.segment_list):  # ignore the last case
            return
        curr = self.segment_list[i]
        next = self.segment_list[i + 1]
        assert not curr.is_intersection
        assert next.is_intersection
        assert self.lanefollow_dist_threshold < self.lanefollow_split_threshold_long
        if i == 0:  # left most case
            min_index = left_index = curr.index_range[0]
        else:
            min_cluster_index = self.find_closest_cluster_index(self.segment_list[i - 1].mid_index)
            min_index = self.segment_list[i - 1].index_range[0]
            left_index = curr.mid_index
            while left_index > min_index and \
                    self.dist_to_cluster(min_cluster_index, left_index) > self.lanefollow_dist_threshold:
                left_index -= 1
        if i + 1 == len(self.segment_list):  # right most case
            right_index = curr.index_range[1]
            max_cluster_index = self.find_closest_cluster_index(self.segment_list[i].last_index)
        else:
            max_cluster_index = self.find_closest_cluster_index(self.segment_list[i + 1].mid_index)
            # cluster_index_dict[i] = max_cluster_index
            max_index = self.segment_list[i + 1].index_range[1]
            right_index = curr.mid_index
            while right_index < max_index and \
                    self.dist_to_cluster(max_cluster_index, right_index) > self.lanefollow_dist_threshold:
                right_index += 1
        split_index = right_index - 1
        while split_index > min_index and \
                self.dist_to_cluster(max_cluster_index, split_index) < self.lanefollow_split_threshold_long:
            split_index -= 1
        self.segment_list[i].index_range = left_index, right_index
        self.segment_list[i].split_index = split_index
        assert left_index < split_index < right_index

    def extend_segment_list(self):
        # find intersection points
        for i, s in enumerate(self.segment_list):
            if s.is_intersection:
                self.extend_intersection_segment(i)
            else:
                self.extend_lanefollow_segment(i)
        del self.segment_list[-1]
        # for i in range(10):
        #     logger.info(self.segment_list[i].index_range)

    def fetch_segment_piece_boundary_index(self, s: SegmentInfo, first: bool) -> int:
        i1, i2 = s.index_range
        assert i1 < i2
        if i2 - i1 <= self.segment_length_threshold:
            return i1 if first else i2
        else:
            num_segment = (i2 - i1) // self.segment_length_threshold + 1
            sub_segment_length = (i2 - i1) / num_segment
            index_range_list = []
            for n in range(num_segment):
                j1 = i1 + round(n * sub_segment_length)
                j2 = i1 + round((n + 1) * sub_segment_length)
                index_range_list.append((j1, j2))
            return index_range_list[-1][0] if first else index_range_list[0][1]

    def dist_to_goal(self, ci: int, gi: int) -> float:
        x1, y1 = self.infos[ci].x, self.infos[ci].y
        x2, y2 = self.infos[gi].x, self.infos[gi].y
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def compute_high_level_segments(self) -> List[Dict[str, Any]]:
        collpased_segments = []
        for i in range(1, len(self.segment_list) - 1):
            if self.segment_list[i - 1].option == 'lanefollow' and \
                    self.segment_list[i].option != 'lanefollow' and \
                    self.segment_list[i + 1].option == 'lanefollow':
                s = self.segment_list[i]
                i1 = self.fetch_segment_piece_boundary_index(self.segment_list[i - 1], True)
                i2 = self.fetch_segment_piece_boundary_index(self.segment_list[i + 1], False)
                index_range = i1, i2
                collpased_segments.append(SegmentInfo(index_range, True, s.option, s.possible_options, True))

        segment_infos = []
        for i, s in enumerate(collpased_segments):
            cluster_index = self.find_closest_cluster_index(s.mid_index)
            min_index, max_index = s.index_range
            left_index2 = s.mid_index
            right_index1 = s.mid_index
            while left_index2 > min_index and self.dist_to_cluster(cluster_index, left_index2) < self.inner_radius:
                left_index2 -= 1
            while right_index1 < max_index and self.dist_to_cluster(cluster_index, right_index1) < self.inner_radius:
                right_index1 += 1
            left_index1 = left_index2
            right_index2 = right_index1
            while left_index1 > min_index and self.dist_to_cluster(cluster_index, left_index1) < self.outer_radius:
                left_index1 -= 1
            while right_index2 < max_index and self.dist_to_cluster(cluster_index, right_index2) < self.outer_radius:
                right_index2 += 1
            goal_index = right_index2
            while goal_index < max_index and self.dist_to_goal(goal_index, max_index) > self.goal_radius:
                goal_index += 1
            conditions = [min_index < left_index1, left_index1 < left_index2,
                          right_index1 < right_index2 <= goal_index, goal_index < max_index]
            if not all(conditions):
                continue
            segment_infos.append({
                'sentence': s.option.upper(),
                'sequence': [
                    (min_index, left_index1, 'LANEFOLLOW'),
                    (left_index1, left_index2, s.option.upper()),
                    (right_index1, right_index2, 'LANEFOLLOW'),
                    (goal_index, max_index, 'STOP')]
            })
        return segment_infos

    def augment_segment_list(self):
        # replicate segments based on possible road options
        logger.info('augment was called')
        duplicated_segment_list = []
        for i, s in enumerate(self.segment_list):
            if s.option == 'straight' and len(s.possible_options) == 2:
                available_option = list(filter(lambda x: x in ['left', 'right'], s.possible_options))[0]
                actual_option = 'extraleft' if available_option == 'right' else 'extraright'
                duplicated_segment_list.append((i, actual_option))
            if len(s.possible_options) == 2 and \
                (s.option == 'left' and 'right' in s.possible_options or
                 s.option == 'right' and 'left' in s.possible_options):
                actual_option = 'extrastraight'
                duplicated_segment_list.append((i, actual_option))

        for i, option in duplicated_segment_list:
            s = self.segment_list[i]
            new_segment = SegmentInfo(
                s.index_range, s.is_intersection, option, s.possible_options, s.has_stop, s.split_index, i)
            self.segment_list.append(new_segment)

    # split long lanefollow segments
    def finalize_segment_list(self) -> List[Dict[str, Any]]:
        new_segment_list: List[dict] = []
        for i, s in enumerate(self.segment_list):
            if s.option in ['left', 'right', 'straight', 'extraleft', 'extraright', 'extrastraight',
                            'extrastraightleft', 'extrastraightright']:
                new_segment_list.append(dict_from_segment_info(s, True))
            elif s.option == 'lanefollow':
                has_stop = i < len(self.segment_list) - 1 and self.segment_list[i + 1].option != 'lanefollow'
                i1, i2 = s.index_range
                assert i1 < i2
                if i2 - i1 <= self.segment_length_threshold:
                    new_segment_list.append(dict_from_segment_info(s, has_stop))
                else:
                    num_segment = (i2 - i1) // self.segment_length_threshold + 1
                    sub_segment_length = (i2 - i1) / num_segment

                    def make_pair(n):
                        return i1 + round(n * sub_segment_length), i1 + round((n + 1) * sub_segment_length)

                    index_pairs = list(map(make_pair, range(num_segment)))
                    stops = list(map(lambda n: n == num_segment - 1, range(num_segment)))
                    split_indices = []
                    for index_pair in index_pairs:
                        if s.split_index >= index_pair[1]:
                            split_indices.append(-1)
                        elif s.split_index < index_pair[0]:
                            split_indices.append(index_pair[0])
                        else:
                            split_indices.append(s.split_index)
                    for stop, index_pair, split_index in zip(stops, index_pairs, split_indices):
                        new_segment = SegmentInfo(index_pair, s.is_intersection, s.option, s.possible_options,
                                                  stop, split_index)
                        new_segment_list.append(dict_from_segment_info(new_segment, stop))
            else:
                raise KeyError('invalid road_option was given {}'.format(s.option))
        return new_segment_list

    def export_segment_dict(
            self,
            low_level_segment_list: List[Dict[str, Any]],
            high_level_segment_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # update the segment indices with actual data/image frame indices
        new_low_level_segment_list = []
        for s in low_level_segment_list:
            split_index = self.infos[s['split_index']].index if s['split_index'] >= 0 else -1
            indices = list(map(lambda i: self.infos[i].index, range(*s['indices'])))
            for i, (j, k) in enumerate(zip(indices[:-1], indices[1:])):
                if j + 1 != k:
                    logger.error('invalid indices at {}: {}, {}'.format(i, j, k))
            new_segment = {
                'option': s['option'],
                'indices': (indices[0], indices[-1] + 1),
                'split_index': split_index,
                'has_stop': s['has_stop']
            }
            new_low_level_segment_list.append(new_segment)

        info_dict = dict()
        for i, info in enumerate(self.infos):
            info_dict[info.index] = i

        # update the high-level segment indices
        new_high_level_segment_list = []
        for s in high_level_segment_list:
            sequence = s['sequence']
            new_sequence = []
            for i1, i2, opt in sequence:
                indices = list(map(lambda i: self.infos[i].index, range(i1, i2)))
                new_sequence.append((indices[0], indices[-1], opt))
            new_high_level_segment_list.append({'sentence': s['sentence'], 'sequence': new_sequence})

        segment_dict = {
            'low_level_segments': new_low_level_segment_list,
            'high_level_segments': new_high_level_segment_list,
            'clusters': self.cluster_point_list,
            'info_dict': info_dict
        }
        return segment_dict

    def extract_segments(self) -> Dict[str, List[Any]]:
        self.update_label_from_raw_data()
        if self.unique:
            self.make_segment_list_unique()
        self.find_intersection_points()
        self.extend_segment_list()
        if not self.unique:
            if self.augment:
                self.augment_segment_list()
            high_level_segment_list = self.compute_high_level_segments()
            low_level_segment_list = self.finalize_segment_list()
            segment_dict = self.export_segment_dict(low_level_segment_list, high_level_segment_list)
        else:
            low_level_segment_list = [dict_from_segment_info(s, False) for s in self.segment_list]
            segment_dict = self.export_segment_dict(low_level_segment_list, [])
            logger.info('final low-level segments {}'.format(len(self.segment_list)))
        return segment_dict

    def make_segment_list_unique(self):
        num_segment = len(self.segment_list)
        xya_list = []
        for s, segment in enumerate(self.segment_list):
            locations = [(self.infos[f].x, self.infos[f].y) for f in range(*segment.index_range)]
            xs, ys = zip(*locations)
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            a1 = atan2(my - ys[0], mx - xs[0]) * 180 / pi
            a2 = atan2(ys[-1] - my, xs[-1] - mx) * 180 / pi
            xya_list.append((mx, my, a1, a2))

        def compute_dist(v1, v2):
            return sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

        dist_threshold = 10.0
        edge_dict = defaultdict(list)
        for i, v1 in enumerate(xya_list):
            ivl = sorted([(j, compute_dist(v1, xya_list[j])) for j in range(i + 1, len(xya_list))], key=lambda x: x[1])
            edge_dict[i] = list(map(itemgetter(0), filter(lambda x: x[1] < dist_threshold, ivl)))
            print(i, *v1, edge_dict[i])

        visited = [False for _ in range(num_segment)]
        unique_indices = []
        for i in range(num_segment):
            if visited[i]:
                continue
            visited[i] = True
            for n in edge_dict[i]:
                visited[n] = True
            unique_indices.append(i)

        print(len(unique_indices))
        road_options = list(map(lambda i: self.segment_list[i].option, unique_indices))
        print(list(zip(unique_indices, road_options)))
        self.segment_list = [self.segment_list[i] for i in unique_indices]


class SegmentExtractorv18(SegmentExtractor):
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractor.__init__(self, infos, unique, augment)
        self.intersection_dist_threshold = 30

    def extend_intersection_segment(self, i: int):
        s = self.segment_list[i]
        assert s.is_intersection
        cluster_index = self.find_closest_cluster_index(s.mid_index)
        min_index = self.segment_list[i - 1].index_range[0] if i > 0 else s.index_range[0]
        max_index = self.segment_list[i + 1].index_range[1] if i + 1 < len(self.segment_list) else s.index_range[1]
        left_index = s.mid_index
        right_index = s.mid_index
        while left_index > min_index and \
                self.dist_to_cluster(cluster_index, left_index) < self.intersection_dist_threshold:
            left_index -= 1
        while right_index < max_index and \
                self.dist_to_cluster(cluster_index, right_index) < self.intersection_dist_threshold:
            right_index += 1
        self.segment_list[i].index_range = left_index, right_index

    def extend_lanefollow_segment(self, i: int):
        s = self.segment_list[i]
        assert not s.is_intersection
        if i == 0:  # left most case
            left_index = s.index_range[0]
        else:
            min_cluster_index = self.find_closest_cluster_index(self.segment_list[i - 1].mid_index)
            min_index = self.segment_list[i - 1].index_range[0]
            left_index = s.mid_index
            while left_index > min_index and self.dist_to_cluster(min_cluster_index, left_index) > self.lanefollow_dist_threshold:
                left_index -= 1
        if i + 1 == len(self.segment_list):  # right most case
            right_index = s.index_range[1]
        else:
            max_cluster_index = self.find_closest_cluster_index(self.segment_list[i + 1].mid_index)
            # cluster_index_dict[i] = max_cluster_index
            max_index = self.segment_list[i + 1].index_range[1]
            right_index = s.mid_index
            while right_index < max_index and self.dist_to_cluster(max_cluster_index, right_index) > self.lanefollow_dist_threshold:
                right_index += 1
        self.segment_list[i].index_range = left_index, right_index


class SegmentExtractorv17(SegmentExtractorv18):  # to reproduce the dataset version 17/18
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv18.__init__(self, infos, unique, augment)
        self.lanefollow_dist_threshold = 15


class SegmentExtractorv26(SegmentExtractor):  # to fix weird control values (too many lanefollow controls)
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractor.__init__(self, infos, unique, augment)
        self.intersection_dist_enter_threshold = 20
        self.intersection_dist_exit_threshold = 20


class SegmentExtractorv27(SegmentExtractorv26):  # for high-level dataset
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv26.__init__(self, infos, unique, augment)
        self.lanefollow_dist_threshold = 15


class SegmentExtractorv28(SegmentExtractor):  # to fix control problems more than 100k
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractor.__init__(self, infos, unique, augment)
        self.intersection_dist_enter_threshold = 18
        self.intersection_dist_exit_threshold = 25


class SegmentExtractorv29(SegmentExtractorv27):  # enlarged region for high-level dataset
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv27.__init__(self, infos, unique, augment)
        self.inner_radius = 12
        self.outer_radius = 27
        self.goal_radius = 10


class SegmentExtractorv30(SegmentExtractor):  # make the stopping decision earlier
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractor.__init__(self, infos, unique, augment)
        self.lanefollow_split_threshold = 20


class SegmentExtractorv31(SegmentExtractorv26):  # do not augment straight to left and right
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv26.__init__(self, infos, unique, False)


class SegmentExtractorv32(SegmentExtractor):  # make the stopping decision even earlier
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractor.__init__(self, infos, unique, augment)
        self.lanefollow_split_threshold = 23


class SegmentExtractorv33(SegmentExtractorv29):  # make the stopping decision even earlier
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv29.__init__(self, infos, unique, False)


class ExtendedSegmentExtractor(SegmentExtractorv29):  # same with v33
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        SegmentExtractorv29.__init__(self, infos, unique, augment)
        self.num_turns = [1, 2]

    def compute_indices_from_turn_segment_index(self, index: int):
        assert self.segment_list[index - 1].option == 'lanefollow'
        assert self.segment_list[index + 1].option == 'lanefollow'
        assert self.segment_list[index].option != 'lanefollow'
        curr = self.segment_list[index]
        i1 = self.fetch_segment_piece_boundary_index(self.segment_list[index - 1], True)
        i2 = self.fetch_segment_piece_boundary_index(self.segment_list[index + 1], False)
        extended = SegmentInfo((i1, i2), True, curr.option, curr.possible_options, True)
        cluster_index = self.find_closest_cluster_index(extended.mid_index)
        min_index, max_index = extended.index_range
        left_index2 = extended.mid_index
        right_index1 = extended.mid_index
        while left_index2 > min_index and self.dist_to_cluster(cluster_index, left_index2) < self.inner_radius:
            left_index2 -= 1
        while right_index1 < max_index and self.dist_to_cluster(cluster_index, right_index1) < self.inner_radius:
            right_index1 += 1
        left_index1 = left_index2
        right_index2 = right_index1
        while left_index1 > min_index and self.dist_to_cluster(cluster_index, left_index1) < self.outer_radius:
            left_index1 -= 1
        while right_index2 < max_index and self.dist_to_cluster(cluster_index, right_index2) < self.outer_radius:
            right_index2 += 1
        goal_index = right_index2
        while goal_index < max_index and self.dist_to_goal(goal_index, max_index) > self.goal_radius:
            goal_index += 1
        conditions = [min_index < left_index1, left_index1 < left_index2,
                      right_index1 < right_index2 <= goal_index, goal_index < max_index]
        if all(conditions):
            return {
                'sentence': extended.option.upper(),
                'sequence': [
                    (min_index, left_index1, 'LANEFOLLOW'),
                    (left_index1, left_index2, extended.option.upper()),
                    (right_index1, right_index2, 'LANEFOLLOW'),
                    (goal_index, max_index, 'STOP')]
            }
        else:
            logger.info('failed to generate a segment from {}'.format(index))
            return None

    def merge_turn_segment_info(self, segment_info_list: List[Dict[str, Any]]):
        sentence = ','.join([s['sentence'] for s in segment_info_list])  # merge keywords
        mid_sequences = list(chain.from_iterable([s['sequence'][1:3] for s in segment_info_list]))  # action, lanefollow
        # complete the sequence: [lf] + [action, lf, action, lf, ..., action, lf] + [stop]
        sequences = [segment_info_list[0]['sequence'][0]] + mid_sequences + [segment_info_list[-1]['sequence'][-1]]
        return {
            'sentence': sentence,
            'sequence': sequences
        }

    def compute_high_level_segments(self) -> List[Dict[str, Any]]:
        simple_turn_indices = []
        turn_segment_info_dict = dict()
        logger.info('called high')

        extra_turn_index_dict = dict()
        for i in range(1, len(self.segment_list) - 1):
            if self.segment_list[i - 1].option == 'lanefollow' and \
                    self.segment_list[i + 1].option == 'lanefollow' and \
                    self.segment_list[i].option != 'lanefollow':
                simple_turn_indices.append(i)
                turn_segment_info_dict[i] = self.compute_indices_from_turn_segment_index(i)

        # update extra options differently because it does not have neighboring lanefollow segments
        for i in range(len(self.segment_list)):
            if self.segment_list[i].option.startswith('extra'):
                ref_index = self.segment_list[i].ref_index
                if ref_index < 0:
                    continue
                info_dict = self.compute_indices_from_turn_segment_index(ref_index)
                left_index1 = info_dict['sequence'][1][0]
                left_index2 = info_dict['sequence'][1][1]
                info_dict['sequence'][1] = left_index1, left_index2, self.segment_list[i].option.upper()
                info_dict['sentence'] = self.segment_list[i].option.upper()  # put extra keyword
                turn_segment_info_dict[i] = info_dict
                extra_turn_index_dict[ref_index] = i
                print(info_dict)

        turn_indices = []
        for num_turn in self.num_turns:
            for i in range(len(simple_turn_indices)):
                valid = True
                for j in range(num_turn):
                    if not (i + j < len(simple_turn_indices) and
                            simple_turn_indices[i] + 2 * j == simple_turn_indices[i + j]):
                        valid = False
                        break
                if valid:
                    turn_indices.append([simple_turn_indices[i + j] for j in range(num_turn)])

        new_turn_indices = []
        for turn_index_list in turn_indices:
            has_extra_turns = False
            new_turn_index_list = [index for index in turn_index_list]
            for i, index in enumerate(turn_index_list):
                if index in extra_turn_index_dict:
                    has_extra_turns = True
                    new_turn_index_list[i] = extra_turn_index_dict[index]
            if has_extra_turns:
                new_turn_indices.append(new_turn_index_list)
        turn_indices += new_turn_indices

        raw_turn_segment_infos = list(map(lambda indices: [turn_segment_info_dict[i] for i in indices], turn_indices))
        candidates = list(map(self.merge_turn_segment_info, raw_turn_segment_infos))
        return candidates


class SegmentExtractorv35(SegmentExtractor):
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        self.lanefollow_split_threshold_short = 15.0
        SegmentExtractor.__init__(self, infos, unique, False)
        assert self.lanefollow_dist_threshold < self.lanefollow_split_threshold_short  # moved from each extend call

    def extend_lanefollow_segment(self, i: int) -> None:
        if i + 1 >= len(self.segment_list):  # ignore the last case
            return
        curr = self.segment_list[i]
        next = self.segment_list[i + 1]
        assert not curr.is_intersection
        assert next.is_intersection
        long = 'straight' in next.possible_options
        split_threshold = self.lanefollow_split_threshold_long if long else self.lanefollow_split_threshold_short

        if i == 0:  # left most case
            min_index = left_index = curr.index_range[0]
        else:
            min_cluster_index = self.find_closest_cluster_index(self.segment_list[i - 1].mid_index)
            min_index = self.segment_list[i - 1].index_range[0]
            left_index = curr.mid_index
            while left_index > min_index and \
                    self.dist_to_cluster(min_cluster_index, left_index) > self.lanefollow_dist_threshold:
                left_index -= 1
        if i + 1 == len(self.segment_list):  # right most case
            right_index = curr.index_range[1]
            max_cluster_index = self.find_closest_cluster_index(self.segment_list[i].last_index)
        else:
            max_cluster_index = self.find_closest_cluster_index(self.segment_list[i + 1].mid_index)
            # cluster_index_dict[i] = max_cluster_index
            max_index = self.segment_list[i + 1].index_range[1]
            right_index = curr.mid_index
            while right_index < max_index and \
                    self.dist_to_cluster(max_cluster_index, right_index) > self.lanefollow_dist_threshold:
                right_index += 1
        split_index = right_index - 1
        while split_index > min_index and \
                self.dist_to_cluster(max_cluster_index, split_index) < split_threshold:
            split_index -= 1
        self.segment_list[i].index_range = left_index, right_index
        self.segment_list[i].split_index = split_index
        assert left_index < split_index < right_index


class ExtendedSegmentExtractorv36(ExtendedSegmentExtractor):  # from v34, removed lanefollow before stop, #turns = 1
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        ExtendedSegmentExtractor.__init__(self, infos, unique, augment)
        self.num_turns = [1]

    def compute_indices_from_turn_segment_index(self, index: int):
        assert self.segment_list[index - 1].option == 'lanefollow'
        assert self.segment_list[index + 1].option == 'lanefollow'
        assert self.segment_list[index].option != 'lanefollow'
        curr = self.segment_list[index]
        i1 = self.fetch_segment_piece_boundary_index(self.segment_list[index - 1], True)
        i2 = self.fetch_segment_piece_boundary_index(self.segment_list[index + 1], False)
        extended = SegmentInfo((i1, i2), True, curr.option, curr.possible_options, True)
        cluster_index = self.find_closest_cluster_index(extended.mid_index)
        min_index, max_index = extended.index_range
        left_index2 = extended.mid_index
        right_index1 = extended.mid_index
        while left_index2 > min_index and self.dist_to_cluster(cluster_index, left_index2) < self.inner_radius:
            left_index2 -= 1
        while right_index1 < max_index and self.dist_to_cluster(cluster_index, right_index1) < self.inner_radius:
            right_index1 += 1
        left_index1 = left_index2
        right_index2 = right_index1
        while left_index1 > min_index and self.dist_to_cluster(cluster_index, left_index1) < self.outer_radius:
            left_index1 -= 1
        while right_index2 < max_index and self.dist_to_cluster(cluster_index, right_index2) < self.outer_radius:
            right_index2 += 1
        conditions = [min_index < left_index1, left_index1 < left_index2, right_index1 < right_index2]
        if all(conditions):
            return {
                'sentence': extended.option.upper(),
                'sequence': [
                    (min_index, left_index1, 'LANEFOLLOW'),
                    (left_index1, left_index2, extended.option.upper()),
                    (right_index1, right_index2, 'LANEFOLLOW')]
            }
        else:
            logger.info('failed to generate a segment from {}'.format(index))
            return None

    def merge_turn_segment_info(self, segment_info_list: List[Dict[str, Any]]):
        sentence = ','.join([s['sentence'] for s in segment_info_list])
        mid_sequences = list(chain.from_iterable([s['sequence'][1:] for s in segment_info_list]))
        sequences = [segment_info_list[0]['sequence'][0]] + mid_sequences
        sequences[-1] = (sequences[-1][0], sequences[-1][1], 'STOP')
        return {
            'sentence': sentence,
            'sequence': sequences
        }


class ExtendedSegmentExtractorv37(ExtendedSegmentExtractorv36):  # #turns = [1, 2]
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        ExtendedSegmentExtractorv36.__init__(self, infos, unique, False)
        self.num_turns = [1, 2]


class ExtendedSegmentExtractorv38(ExtendedSegmentExtractorv36):
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        ExtendedSegmentExtractorv36.__init__(self, infos, unique, True)
        self.num_turns = [1, 2]


class ExtendedSegmentExtractorv39(ExtendedSegmentExtractorv36):
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        ExtendedSegmentExtractorv36.__init__(self, infos, unique, True)
        self.num_turns = [1, 2]

    def augment_segment_list(self):
        # replicate segments based on possible road options
        logger.info('augment was called')
        duplicated_segment_list = []
        for i, s in enumerate(self.segment_list):
            if s.option == 'straight' and len(s.possible_options) == 2:
                available_option = list(filter(lambda x: x in ['left', 'right'], s.possible_options))[0]
                actual_option = 'extraleft' if available_option == 'right' else 'extraright'
                duplicated_segment_list.append((i, actual_option))
            if len(s.possible_options) == 2 and \
                (s.option == 'left' and 'right' in s.possible_options or
                 s.option == 'right' and 'left' in s.possible_options):
                actual_option = 'extrastraight' + s.option
                duplicated_segment_list.append((i, actual_option))

        for i, option in duplicated_segment_list:
            s = self.segment_list[i]
            new_segment = SegmentInfo(
                s.index_range, s.is_intersection, option, s.possible_options, s.has_stop, s.split_index, i)
            self.segment_list.append(new_segment)


class ExtendedSegmentExtractorv40(ExtendedSegmentExtractorv36):
    def __init__(self, infos: List[FrameInfo], unique: bool, augment: bool):
        ExtendedSegmentExtractorv36.__init__(self, infos, unique, True)
        self.num_turns = [1, 2]

    def refine_high_level_segments_with_extra_straight(self, data: List[dict]):
        sentence_dict = defaultdict(list)
        for i, d in enumerate(data):
            if 'extra' in d['sentence'].lower():
                sentence_dict[d['sentence'].lower()].append(i)
        sequences = [d['sequence'] for d in data]

        def cut_sequence_with_extra_straight(sequences: List[Tuple[int, int, str]]):
            index = -1
            for i, item in enumerate(sequences):
                if item[-1].lower() == 'extrastraight':
                    index = i
                    break
            if index >= 0:
                return sequences[:index + 1]
            else:
                return sequences

        sequences = list(map(cut_sequence_with_extra_straight, sequences))

        def fetch_location(drive_frame: FrameInfo):
            return drive_frame.x, drive_frame.y

        def fetch_mean_location(drive_frames: List[FrameInfo]):
            xs, ys = zip(*list(map(fetch_location, drive_frames)))
            return sum(xs) / len(xs), sum(ys) / len(ys)

        def compute_average_dist(l1, l2):
            dist = 0.0
            for (x1, y1), (x2, y2) in zip(l1, l2):
                dist += (x2 - x1) ** 2 + (y2 - y1) ** 2
            return sqrt(dist)

        def compute_clusters(segment_indices: List[int]) -> List[List[int]]:
            # indices = [0, 2, 4] if len(segment_indices) == 5 else [0, 2]
            locations = []
            for i in segment_indices:
                indices = list(map(itemgetter(0),
                                   filter(lambda x: x[1][-1].lower() == 'lanefollow', enumerate(sequences[i]))))
                sequence = [sequences[i][j] for j in indices]
                infos = [[self.infos[i] for i in range(*x[:2])] for x in sequence]
                locations.append(list(map(fetch_mean_location, infos)))

            cluster_index_dict = dict()
            cluster_index_list_dict = defaultdict(list)
            cluster_index = 0
            dist_threshold = 10.0
            for i, l1 in zip(range(len(segment_indices)), locations):
                valid = True
                for j, k in cluster_index_dict.items():
                    dist = compute_average_dist(l1, locations[k])
                    if dist < dist_threshold:
                        valid = False
                        cluster_index_list_dict[j].append(i)
                        break
                if valid:
                    cluster_index_dict[cluster_index] = i
                    cluster_index_list_dict[cluster_index].append(i)
                    cluster_index += 1

            cluster_indices = []
            for indices in cluster_index_list_dict.values():
                cluster_indices.append([segment_indices[i] for i in indices])
            return cluster_indices

        cluster_dict = defaultdict(list)
        for key, value in sentence_dict.items():
            cluster_dict[key] = compute_clusters(value)

        for key, value in cluster_dict.items():
            print('found {} clusters from {}'.format(len(value), key))

        segments = []
        for key, cluster_indices in cluster_dict.items():
            for i, cluster_index in enumerate(cluster_indices):
                print(key, i, cluster_index)
                sequence = sequences[cluster_index[0]]
                segments.append({'sentence': key.upper(), 'sequence': sequence})
        return segments

    def extract_segments(self) -> Dict[str, List[Any]]:
        self.update_label_from_raw_data()
        if self.unique:
            self.make_segment_list_unique()
        self.find_intersection_points()
        self.extend_segment_list()
        if not self.unique:
            if self.augment:
                self.augment_segment_list()
            high_level_segment_list = self.compute_high_level_segments()
            low_level_segment_list = self.finalize_segment_list()
            high_level_segment_list = self.refine_high_level_segments_with_extra_straight(
                high_level_segment_list)
            segment_dict = self.export_segment_dict(low_level_segment_list, high_level_segment_list)
        else:
            low_level_segment_list = [dict_from_segment_info(s, False) for s in self.segment_list]
            segment_dict = self.export_segment_dict(low_level_segment_list, [])
            logger.info('final low-level segments {}'.format(len(self.segment_list)))
        return segment_dict


__segment_extractor_dict_by_version__ = {
    40: ExtendedSegmentExtractorv40,
    39: ExtendedSegmentExtractorv39,
    38: ExtendedSegmentExtractorv38,
    37: ExtendedSegmentExtractorv37,
    36: ExtendedSegmentExtractorv36,
    35: SegmentExtractorv35
}


def fetch_segment_extractor_by_version(version: int):
    if version in __segment_extractor_dict_by_version__:
        return __segment_extractor_dict_by_version__[version]
    else:
        raise ValueError('version is too low {}: minimum version is 35'.format(version))
