import json
import re
from operator import itemgetter
from pathlib import Path
from moviepy import editor

from util.common import unique_with_islices


def generate_online_video():
    map_index = 1
    traj_name = 'ls-town2'
    # seq_dir = Path.home() / '.carla/evaluations/exp37/gs1-town1-new1/step100000/online/'
    seq_dir = Path.home() / 'Downloads/lgd-video/{}'.format(traj_name)
    in_vid_file = seq_dir / '{}e.mp4'.format(traj_name)
    out_vid_file = seq_dir / 'test-{}e.mp4'.format(traj_name)
    state_path = seq_dir / '{}-state.json'.format(traj_name)
    image_dir = seq_dir / 'images'
    # seg_dir = seq_dir / 'segments'
    # seg_fmt = '{:08d}.png'
    image_fmt = '{:08d}e.png'
    # seq_path = Path.home() / '.carla/evaluations/exp37/gs1-town1-new1/step100000/online/states/traj01.json'

    def fetch_color_code(option: str) -> str:
        option_code = {
            'left': 'C0',
            'right': 'C3',
            'straight': 'C1',
            'newfollow': 'C2',
            'extraleft': 'C2',
            'extraright': 'C2',
            'extrastraight': 'b',
            'extrastraightleft': 'g',
            'extrastraightright': 'r',
            'lanefollow': 'k',
            'stop': 'y'}
        intersection = option in ['left', 'right', 'straight']
        return option_code[option]

    def fetch_default_trajectory(map_index: int, num_traj: int):
        rawdata = '1561065444962896' if map_index == 1 else '1560998639947491'
        data_path = Path.home() / '.carla/rawdata/{}/data.txt'.format(rawdata)
        with open(str(data_path), 'r') as file:
            lines = file.read().splitlines()[:num_traj]
        lines = list(map(lambda x: x.split(':')[1], lines))
        xs = list(map(lambda x: float(x.split(',')[0]), lines))
        ys = list(map(lambda x: -float(x.split(',')[1]), lines))
        return xs, ys

    def fetch_sequences(map_index: int):
        data_path = Path.home() / '.carla/dataset/info/semantic{}-v37/segment.json'.format(map_index)
        with open(str(data_path), 'r') as file:
            data = json.load(file)
        return data['low_level_segments']

    def fetch_sequences_online(seq_path: Path):
        with open(str(seq_path), 'r') as file:
            data = json.load(file)
        xs, ys = zip(*list(map(lambda x: tuple([float(v) for v in x.split(',')[:2]]), data['data_frames'])))
        frame_range = data['frame_range']
        sentences = data['sentences']
        subgoals = list(map(itemgetter(1), data['stop_frames']))
        return xs, ys, frame_range, sentences, subgoals

    ds = DataStorage(True, Path.home() / '.carla/dataset/data/semantic{}'.format(map_index), True)
    dxs, dys = fetch_default_trajectory(map_index, 50000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dxs, dys, fetch_color_code('lanefollow'), linewidth=4)

    xs, ys, frame_range, sentences, subgoals = fetch_sequences_online(state_path)
    ys = [-v for v in ys]
    traj_dir = mkdir_if_not_exists(Path.home() / '.tmp/traj_online')
    sentence_index_list = unique_with_islices(sentences)
    subtask_index_list = unique_with_islices(subgoals)
    # image_file_list = sorted(image_dir.glob('*.jpg'))
    # image_list = [cv2.imread(str(f)) for f in image_file_list]
    # print(len(image_file_list), len(list(range(*frame_range))), len(sentences))
    print(sentence_index_list)
    print(subtask_index_list)

    def generate_text_clips(text_index_list, prefix, fontsize, pos, color='white'):
        clips = []
        for sentence, frame_range in text_index_list:
            text_clip = TextClip(
                prefix + sentence + ' ', fontsize=fontsize, color=color, bg_color='black', font='Lato-Medium')
            start_time = frame_range[0] / clip.fps
            end_time = frame_range[1] / clip.fps
            print(sentence, start_time)
            text_clip = text_clip.set_start(start_time, False).set_end(end_time)
            text_clip = text_clip.set_position(pos)
            clips.append(text_clip)
        return clips

    clip = VideoFileClip(str(in_vid_file))
    clips = [clip] + \
            generate_text_clips(sentence_index_list, ' sentence: ', 27, (30, 360)) + \
            generate_text_clips(subtask_index_list, ' sub-task : ', 27, (30, 393), 'yellow')

    final = CompositeVideoClip(clips)
    final.write_videofile(str(out_vid_file))


def generate_single_video_with_audio(root_dir, traj_index: int):
    video_dir = root_dir / 'videos'
    audio_dir = root_dir / 'audios'
    state_dir = root_dir / 'states'
    image_dir = root_dir / 'images'
    timing_path = audio_dir / 'timing{:02d}.json'.format(traj_index)
    state_path = state_dir / 'traj{:02d}.json'.format(traj_index)
    out_vid_path = video_dir / 'traj{:02d}r.mp4'.format(traj_index)
    if out_vid_path.exists():
        return

    def image_path_func(frame: int):
        return image_dir / '{:08d}e.png'.format(frame)

    def generate_text_clips(state_path: Path, frame_timing_dict):
        def fetch_sequences_online(seq_path: Path):
            with open(str(seq_path), 'r') as file:
                data = json.load(file)
            xs, ys = zip(*list(map(lambda x: tuple([float(v) for v in x.split(',')[:2]]), data['data_frames'])))
            frame_range = data['frame_range']
            sentences = data['sentences']
            subgoals = list(map(itemgetter(1), data['stop_frames']))
            return xs, ys, frame_range, sentences, subgoals

        xs, ys, frame_range, sentences, subgoals = fetch_sequences_online(state_path)
        sentence_index_list = unique_with_islices(sentences)
        subtask_index_list = unique_with_islices(subgoals)

        def _generate_text_clips(text_index_list, prefix, fontsize, pos, color='white'):
            clips = []
            for sentence, local_frame_range in text_index_list:
                local_frame_range = [v + frame_range[0] for v in local_frame_range]
                start_time, end_time = None, None
                for frame in range(*local_frame_range):
                    if frame not in frame_timing_dict:
                        continue
                    timing = frame_timing_dict[frame]
                    if start_time is None or timing < start_time:
                        start_time = timing
                    if end_time is None or timing > end_time:
                        end_time = timing
                if start_time is None or end_time is None or start_time > end_time:
                    print('failed to find proper timing from {}, {}'.format(sentence, local_frame_range))
                text_clip = editor.TextClip(
                    prefix + sentence + ' ', fontsize=fontsize, color=color, bg_color='black', font='Lato-Medium')
                text_clip = text_clip.set_start(start_time, False).set_end(end_time)
                text_clip = text_clip.set_position(pos)
                clips.append(text_clip)
            return clips

        # fsz, w1, h1, h2 = 27, 30, 360, 393  # 480
        fsz, w1, h1, h2 = 17, 20, 160, 180  # 240
        clips = _generate_text_clips(sentence_index_list, ' sentence: ', fsz, (w1, h1)) + \
                _generate_text_clips(subtask_index_list, ' sub-task : ', fsz, (w1, h2), 'yellow')
        return clips

    with open(str(timing_path), 'r') as file:
        data = json.load(file)
    frame_timing_dict = {int(key): data[key] for key in data.keys()}
    frame_list = sorted(frame_timing_dict.keys())
    frame_list = list(filter(lambda x: image_path_func(x).exists(), frame_list))
    raw_timing_list = [frame_timing_dict[f] for f in frame_list]
    t1, t2 = raw_timing_list[0], raw_timing_list[-1]

    audio_file_list = sorted(audio_dir.glob('*.wav'))
    audio_file_list = list(filter(lambda x: t1 < int(x.stem) < t2, audio_file_list))
    audio_timing_list = list(map(lambda x: int(x.stem), audio_file_list))
    print(audio_timing_list[0], audio_timing_list[-1])

    timing_offset = frame_timing_dict[frame_list[0]]
    image_timing_list = [(frame_timing_dict[f] - timing_offset) / 1e3 for f in frame_list]
    audio_timing_list = [(t - timing_offset) / 1e3 for t in audio_timing_list]
    for key in frame_timing_dict.keys():
        frame_timing_dict[key] = (frame_timing_dict[key] - timing_offset) / 1e3

    image_duration_list = [t2 - t1 for t1, t2 in zip(image_timing_list[:-1], image_timing_list[1:])]
    image_duration_list.append(image_duration_list[-1])
    image_clip_list = []
    for frame, image_timing, image_duration in zip(frame_list, image_timing_list, image_duration_list):
        clip = editor.ImageClip(str(image_path_func(frame)))
        clip = clip.set_duration(image_duration)
        clip = clip.set_start(image_timing, True)
        image_clip_list.append(clip)
    audio_clip_list = []
    for audio_file, audio_timing in zip(audio_file_list, audio_timing_list):
        try:
            clip = editor.AudioFileClip(str(audio_file))
            clip = clip.set_start(audio_timing, True)
            audio_clip_list.append(clip)
        except:
            print('failed in {}'.format(audio_file))
            continue

    text_clips = generate_text_clips(state_path, frame_timing_dict)
    cvc = editor.CompositeVideoClip(image_clip_list + text_clips)
    cac = editor.CompositeAudioClip(audio_clip_list)
    cvc = cvc.set_fps(30)
    cvc = cvc.set_audio(cac)
    cvc.write_videofile(str(out_vid_path))


def generate_video_with_audio(root_dir: Path):
    vid_dir = root_dir / 'videos'
    in_vid_list = sorted(vid_dir.glob('*e.mp4'))
    out_vid_list = sorted(vid_dir.glob('*r.mp4'))

    def fetch_traj_index(s: str, c: str):
        res = re.findall(r'traj([\d]+){}.mp4'.format(c), str(s))
        return int(res[0]) if res else -1

    in_index_list = list(filter(lambda x: x >= 0, [fetch_traj_index(s, 'e') for s in in_vid_list]))
    out_index_list = list(filter(lambda x: x >= 0, [fetch_traj_index(s, 'r') for s in out_vid_list]))
    index_list = list(filter(lambda x: x not in out_index_list, in_index_list))

    for t in index_list:
        generate_single_video_with_audio(root_dir, t)


if __name__ == '__main__':
    root_dir = Path.home() / 'projects/language-grounded-driving/.carla/evaluations/exp40/ls-town2/step072500/online'
    # generate_video_with_audio(root_dir)
    for t in [6, 8]:
        generate_single_video_with_audio(root_dir, t)
