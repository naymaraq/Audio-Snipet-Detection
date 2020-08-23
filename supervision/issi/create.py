import json
import os

from supervision.issi.snipets import *
from vggish.vggish_feature_extractor import *

SAVE_FOLDER = 'tracks/snipets'
if not os.path.exists(SAVE_FOLDER):
    os.mkdir(SAVE_FOLDER)

labels, patterns, snipets, negative_snipets = generate_audio_segments('tracks/reference.wav',
                                                                      n_segments=10000,
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
