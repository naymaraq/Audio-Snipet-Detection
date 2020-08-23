import os
import numpy as np
import json
from numpy.random import randint
from pydub import AudioSegment

from vggish.vggish_feature_extractor import *


class Snipet:
    def __init__(self, start, end, name, audio):
        self.name = name
        self.start = start
        self.end = end
        self.audio = audio

    def export(self, format):
        self.audio.export(self.name, format)


def has_intersection(s1, e1, s2, e2):
    assert s1 < e1 and s2 < e2
    return not (e1 < s2 or e2 < s1)


def miliseconds(second):
    return second * 1000


def negative_samples(pos_s, pos_e, pattern_len, low, high, n):
    samples = []
    for i in range(n):
        neg_s = randint(low=low, high=high)
        neg_e = neg_s + pattern_len

        neg_s = miliseconds(neg_s)
        neg_e = miliseconds(neg_e)

        while has_intersection(pos_s, pos_e, neg_s, neg_e):
            neg_s = randint(low=low, high=high)
            neg_e = neg_s + pattern_len
            neg_s = miliseconds(neg_s)
            neg_e = miliseconds(neg_e)

        samples.append((neg_s, neg_e))
    return samples


def generate_audio_segments(audo_file, n_segments, min_len, max_len, pattern_len, save_folder, n_negatives):
    """
    min_len and max_len must be in seconds
    """

    # Read audio file
    audio = AudioSegment.from_wav(audo_file)
    duration_in_seconds = audio.duration_seconds

    # generate random start/end points
    starts = list(randint(low=0, high=duration_in_seconds - max_len, size=n_segments))
    ends = [s + randint(low=min_len, high=max_len) for s in starts]

    labels, patterns, snipets, negative_snipets = [], [], [], []
    for i in range(n_segments):

        # Reference snipet
        s, e = miliseconds(starts[i]), miliseconds(ends[i])
        snipet_name = f'snipet_{i}.wav'
        snipet = Snipet(s, e, snipet_name, audio[s:e])
        snipets.append(snipet)

        # Pattern snipet
        s_ = randint(low=starts[i], high=ends[i] - pattern_len)
        e_ = s_ + pattern_len
        s_, e_ = miliseconds(s_), miliseconds(e_)
        pattern_name = f'pattern_{i}.wav'
        pattern = Snipet(s_, e_, pattern_name, audio[s_:e_])
        patterns.append(pattern)

        # negative snipets
        neg_samples = negative_samples(pos_s=s, pos_e=e, pattern_len=pattern_len, low=0,
                                       high=duration_in_seconds - pattern_len, n=n_negatives)
        neg_snips = [Snipet(s, e, f'neg_pattern_{i}_{j}.wav', audio[s:e]) for j, (s, e) in enumerate(neg_samples)]
        negative_snipets.append(neg_snips)

        labels.append((pattern.name, snipet.name, 1))
        for n_s in neg_snips:
            labels.append((n_s.name, snipet.name, 0))

    # save snipet waves
    if save_folder:
        # save patterns
        for p in patterns:
            p.name = f'{save_folder}/{p.name}'
            p.export(format="wav")

        # save snipets
        for s in snipets:
            s.name = f'{save_folder}/{s.name}'
            s.export(format="wav")

        # save negative patterns
        for neg_snips in negative_snipets:
            for n in neg_snips:
                n.name = f'{save_folder}/{n.name}'
                n.export(format="wav")

        # save meta info
        with open(f'{save_folder}/dataset.txt', "w") as f:
            for p, s, y in labels:
                f.write('\t'.join([p, s, str(y)]) + '\n')

    return labels, patterns, snipets, negative_snipets


SAVE_FOLDER = 'tracks/snipets'
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

labels, patterns, snipets, negative_snipets = generate_audio_segments('tracks/reference.wav',
                                                                      n_segments=2,
                                                                      min_len=5,
                                                                      max_len=10,
                                                                      pattern_len=2,
                                                                      save_folder=SAVE_FOLDER,
                                                                      n_negatives=1)

#retraive segments paths
snipet_paths = [s.name for s in snipets]
pattern_paths = [p.name for p in patterns]
neg_snipet_paths = [n.name for n_list in negative_snipets for n in n_list]

#get features
snipet_features = get_features(snipet_paths)
pattern_features = get_features(pattern_paths)
negative_snipets = get_features(neg_snipet_paths)

# pad snipet features
max_len = max([f.shape[0] for p, f in snipet_features.items()])
for path, feature in snipet_features.items():
    l = feature.shape[0]
    if l < max_len:
        feature = np.pad(feature, pad_width=((0, max_len - l), (0, 0)), mode='edge')
        snipet_features[path] = feature

#initialize placeholders
patterns_numpy = np.zeros(shape=(len(labels), *list(pattern_features.items())[0][1].shape))
snipets_numpy = np.zeros(shape=(len(labels), *list(snipet_features.items())[0][1].shape))
y_true = np.zeros(shape=(len(labels, )))

#numpify
pindex2wav = {}
sindex2wav = {}

for index, (p_name, s_name, y) in enumerate(labels):

    pindex2wav[index] = p_name
    sindex2wav[index] = s_name

    p_name = f'{SAVE_FOLDER}/{p_name}'
    s_name = f'{SAVE_FOLDER}/{s_name}'

    p_feature = pattern_features.get(p_name, [])
    if len(p_feature) == 0:
        p_feature = negative_snipets[p_name]
    s_feature = snipet_features[s_name]

    y_true[index] = y
    patterns_numpy[index, :] = p_feature
    snipets_numpy[index, :] = s_feature

#save
y_true.dump(f'{SAVE_FOLDER}/labels.npy')
patterns_numpy.dump(f'{SAVE_FOLDER}/patterns.npy')
snipets_numpy.dump(f'{SAVE_FOLDER}/snipets.npy')
json.dump(pindex2wav, open(f"{SAVE_FOLDER}/pindex2wav", 'w'))
json.dump(sindex2wav, open(f"{SAVE_FOLDER}/sindex2wav", 'w'))
