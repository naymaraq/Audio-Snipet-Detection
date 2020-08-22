from pydub import AudioSegment
from numpy.random import randint
import os

class Snipet:

    def __init__(self, start, end, audio):

        self.start = start
        self.end = end
        self.audio = audio

    def export(self, file_name, format):
        self.audio.export(file_name, format)
        

def has_intersection(s1, e1, s2, e2):

    assert s1 < e1 and s2 < e2
    return not (e1 < s2 or e2 < s1)

def miliseconds(second):
    return second*1000

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


def generate_audio_segments(audo_file, n_segments, min_len, max_len, pattern_len):
    '''
    min_len and max_len must be in seconds
    '''

    audio = AudioSegment.from_wav(audo_file)
    duration_in_seconds = audio.duration_seconds
    starts = list(randint(low=0, high=duration_in_seconds-max_len, size=n_segments))
    ends = [s + randint(low=min_len, high=max_len) for s in starts]

    SAVE_FOLDER = 'tracks/snipets'
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    labels = []
    for i in range(n_segments):

        #Reference snipet
        s, e = miliseconds(starts[i]), miliseconds(ends[i])
        snipet = Snipet(s, e, audio[s:e])

        #Pattern snipet
        s_ = randint(low=starts[i], high=ends[i]-pattern_len)
        e_ = s_ + pattern_len
        s_, e_ = miliseconds(s_), miliseconds(e_)
        pattern = Snipet(s_, e_, audio[s_:e_])

        #negative snipets
        neg_samples = negative_samples(pos_s=s_, pos_e=e_, pattern_len=pattern_len, low=0, high=duration_in_seconds-pattern_len, n=2)
        negative_snipets = [Snipet(s, e, audio[s:e]) for s,e in neg_samples] 
        
        #save snipet waves
        labels.append((f'pattern_{i}.wav', f'snipet_{i}.wav', 1))
        for j, neg_snip in enumerate(negative_snipets):
            neg_snip.export(f'{SAVE_FOLDER}/neg_pattern_{i}_{j}.wav', format="wav")
            labels.append((f'{SAVE_FOLDER}/neg_pattern_{i}_{j}.wav', f'snipet_{i}.wav', 0))
        snipet.export(f'{SAVE_FOLDER}/snipet_{i}.wav', format="wav")
        pattern.export(f'{SAVE_FOLDER}/pattern_{i}.wav', format="wav")

    #save meta info
    with open(f'{SAVE_FOLDER}/dataset.txt', "w") as f:
        for p, s, y in labels:
            f.write('\t'.join([p,s, str(y)]) + '\n')

generate_audio_segments('tracks/reference.wav', 
                        n_segments=5, 
                        min_len=5, 
                        max_len=10, 
                        pattern_len=2)


