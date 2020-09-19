# Tools to load and save midi files for the rnn-gan-project.
# 
# Written by Olof Mogren, http://mogren.one/
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from random import randint

base_tones = {'C':   0,
              'C#':  1, 
              'D':   2,
              'D#':  3,
              'E':   4,
              'F':   5,
              'F#':  6,
              'G':   7,
              'G#':  8,
              'A':   9,
              'A#': 10,
              'B':  11}

base_c_tones = {'C':   0}

scale = {}

# Major scale:
scale['major'] = [0, 2, 4, 5, 7, 9, 11]
# (W-W-H-W-W-W-H)
# (2 2 1 2 2 2 1)

# Natural minor scale:
scale['natural_minor'] = [0, 2, 3, 5, 7, 8, 10]
# (W-H-W-W-H-W-W)
# (2 1 2 2 1 2 2)
 
# Harmonic minor scale:
scale['harmonic_minor'] = [0, 2, 3, 5, 7, 8, 11]
# (W-H-W-W-H-WH-H)
# (2 1 2 2 1 3 1)

tone_names = {}
for tone_name in base_tones:
    tone_names[base_tones[tone_name]] = tone_name


def tones_to_scales(tones):
    """
    Midi to tone name (octave: -5):
    0: C
    1: C#
    2: D
    3: D#
    4: E
    5: F
    6: F#
    7: G
    8: G#
    9: A
    10: A#
    11: B

    Melodic minor scale is ignored.

    One octave is 12 tones.
    """
    counts = {}
    for base_tone in base_tones:
        counts[base_tone] = {}
        counts[base_tone]['major'] = 0
        counts[base_tone]['natural_minor'] = 0
        counts[base_tone]['harmonic_minor'] = 0

    if not len(tones):
        frequencies = {}
        for base_tone in base_tones:
            frequencies[base_tone] = {}
        for scale_label in scale:
            frequencies[base_tone][scale_label] = 0.0
        return frequencies
    for tone in tones:
        for base_tone in base_tones:
            for scale_label in scale:
                if tone%12-base_tones[base_tone] in scale[scale_label]:
                    counts[base_tone][scale_label] += 1
    frequencies = {}
    for base_tone in counts:
        frequencies[base_tone] = {}
        for scale_label in counts[base_tone]:
            frequencies[base_tone][scale_label] = float(counts[base_tone][scale_label])/float(len(tones))
    return frequencies


def tones_to_c_scales(tones):
    """
    Midi to tone name (octave: -5):
    0: C
    1: C#
    2: D
    3: D#
    4: E
    5: F
    6: F#
    7: G
    8: G#
    9: A
    10: A#
    11: B

    Melodic minor scale is ignored.

    One octave is 12 tones.
    """
    counts = {}
    for base_tone in base_c_tones:
        counts[base_tone] = {}
        counts[base_tone]['major'] = 0
        counts[base_tone]['natural_minor'] = 0
        counts[base_tone]['harmonic_minor'] = 0

    if not len(tones):
        frequencies = {}
        for base_tone in base_c_tones:
            frequencies[base_tone] = {}
        for scale_label in scale:
            frequencies[base_tone][scale_label] = 0.0
        return frequencies
    for tone in tones:
        for base_tone in base_c_tones:
            for scale_label in scale:
                if tone%12-base_c_tones[base_tone] in scale[scale_label]:
                    counts[base_tone][scale_label] += 1
    frequencies = {}
    for base_tone in counts:
        frequencies[base_tone] = {}
        for scale_label in counts[base_tone]:
            frequencies[base_tone][scale_label] = float(counts[base_tone][scale_label])/float(len(tones))
    return frequencies


def repetitions(tones):
  rs = {}
  #print(tones)
  #print(len(tones)/2)
  for l in range(2, min(len(tones) // 2, 10)):
      # print (l)
      rs[l] = 0
      cnt = 0
      grams = []
      index = {}
      for i in range(len(tones) - l + 1):
          value = tuple(tones[i:i + l])
          grams.append(value)
          if value not in index:
              index[value] = -1
      for i in grams:
          index[i] += 1
      for i in index:
          if index[i]:
              cnt += index[i]
      rs[l] = cnt
  rs2 = {}
  for r in rs:
    if rs[r]:
      rs2[r] = rs[r]
  return rs2
      

def tone_to_tone_name(tone):
    """
    Midi to tone name (octave: -5):
    0: C
    1: C#
    2: D
    3: D#
    4: E
    5: F
    6: F#
    7: G
    8: G#
    9: A
    10: A#
    11: B

    One octave is 12 tones.
    """

    base_tone = tone_names[tone%12]
    octave = tone//12-1
    return '{} {}'.format(base_tone, octave)


def c_tone_to_c_tone_name(tone):
    """
    Midi to tone name (octave: -5):
    0: C
    1: C#
    2: D
    3: D#
    4: E
    5: F
    6: F#
    7: G
    8: G#
    9: A
    10: A#
    11: B

    One octave is 12 tones.
    """

    base_tone = c_tone_names[tone%12]
    octave = tone//12-1
    return '{} {}'.format(base_tone, octave)


def max_likelihood_scale(tones):
    scale_statistics = tones_to_scales(tones)
    stat_list = []
    for base_tone in scale_statistics:
        for scale_label in scale_statistics[base_tone]:
            stat_list.append((base_tone, scale_label, scale_statistics[base_tone][scale_label]))
    stat_list.sort(key=lambda e: e[2], reverse=True)

    return stat_list[0][0]+' '+stat_list[0][1], stat_list[0][2]
    

def max_likelihood_c_scale(tones):
    scale_statistics = tones_to_c_scales(tones)
    stat_list = []
    for base_tone in scale_statistics:
        for scale_label in scale_statistics[base_tone]:
            stat_list.append((base_tone, scale_label, scale_statistics[base_tone][scale_label]))
    stat_list.sort(key=lambda e: e[2], reverse=True)

    return stat_list[0][0]+' '+stat_list[0][1], stat_list[0][2]


def get_all_stats(midi_pattern):
    stats = {}
    if not midi_pattern:
        print('Failed to read midi pattern.')
        return None
    tones = []
    note_type = []
    rest = []
    for i in range(len(midi_pattern)):
        tones.append(midi_pattern[i][0])
    for i in range(len(midi_pattern)):
        note_type.append(midi_pattern[i][1])
    for i in range(len(midi_pattern)):
        rest.append(midi_pattern[i][2])
    if len(tones) == 0:
        print('This is an empty song.')
        return None
    stats['num_tones'] = len(tones)
    stats['tone_min'] = min(tones)
    stats['tone_max'] = max(tones)
    stats['tone_span'] = max(tones)-min(tones)
    stats['tones_unique'] = len(set(tones))
    rs = repetitions(tones)
    for r in range(2, 10):
        if r in rs:
          stats['repetitions_{}'.format(r)] = rs[r]
        else:
          stats['repetitions_{}'.format(r)] = 0
    ml = max_likelihood_scale(tones)
    stats['scale'] = ml[0]
    stats['scale_score'] = ml[1]
    stats['rest_max'] = max(rest)
    stats['average_rest'] = np.mean(rest)
    stats['num_null_rest'] = rest.count(0)
    stats['songlength'] = sum(rest) + sum(note_type)

    return stats


def print_stats(stats):
    if stats is None:
        print('Could not extract stats.')
    else:
        print('ML scale estimate: {}: {:.2f}'.format(stats['scale'], stats['scale_score']))
        print('Min tone: {}'.format(tone_to_tone_name(stats['tone_min'])))
        print('Max tone: {}'.format(tone_to_tone_name(stats['tone_max'])))
        print('Span: {} tones'.format(stats['tone_span']))
        print('Unique tones: {}'.format(stats['tones_unique']))
        for r in range(2, 10):
            print('Repetitions of len {}: {}'.format(r, stats['repetitions_{}'.format(r)]))
        print('Longest rest: {}'.format(stats['rest_max']))
        print('Average rest: {}'.format(stats['average_rest']))
        print('Number of null rests: {}'.format(stats['num_null_rest']))
        print('Average song length: {}'.format(stats['songlength']))


def tune_song(midi_pattern):
    tones = []
    for i in range(len(midi_pattern)):
        tones.append(midi_pattern[i][0])
    ml = max_likelihood_scale(tones)
    detected_scale = ml[0]
    scale_score = ml[1]

    if detected_scale[1] == '#':
        base_tone = detected_scale[0] + detected_scale[1]
        scale_type = detected_scale[3:]
    else:
        base_tone = detected_scale[0]
        scale_type = detected_scale[2:]

    if scale_score < 1:
        for i in range(len(tones)):
            if tones[i] % 12 - base_tones[base_tone] not in scale[scale_type]:
                tones[i] = tones[i] - (tones[i] % 12 - base_tones[base_tone]) + \
                           min(scale[scale_type], key=lambda x: abs(x-(tones[i] % 12-base_tones[base_tone])))
            midi_pattern[i][0] = tones[i]
        return midi_pattern
    else:
        return midi_pattern


def tune_song_c_scale(midi_pattern):
    tones = []
    for i in range(len(midi_pattern)):
        tones.append(midi_pattern[i][0])
    ml = max_likelihood_c_scale(tones)
    detected_scale = ml[0]
    scale_score = ml[1]

    if detected_scale[1] == '#':
        base_tone = detected_scale[0] + detected_scale[1]
        scale_type = detected_scale[3:]
    else:
        base_tone = detected_scale[0]
        scale_type = detected_scale[2:]

    if scale_score < 1:
        for i in range(len(tones)):
            if tones[i] % 12 - base_c_tones[base_tone] not in scale[scale_type]:
                tones[i] = tones[i] - (tones[i] % 12 - base_c_tones[base_tone]) + \
                           min(scale[scale_type], key=lambda x: abs(x-(tones[i] % 12-base_c_tones[base_tone])))
            midi_pattern[i][0] = tones[i]
        return midi_pattern
    else:
        return midi_pattern


def main():
    test_data = np.load('./data/processed_dataset_matrices/test_data.npy')
    midi_pattern = []
    stats_scale_tot = 0
    stats_repetitions_2_tot = 0
    stats_repetitions_3_tot = 0
    stats_tone_span_tot = 0
    stats_unique_tones_tot = 0
    stats_rest_value = 0
    stats_songlength = 0
    num_null_rest = 0
    best_scale_score = 0
    best_repetitions_2 = 0
    best_repetitions_3 = 0
    longest_rest = 0
    num_perfect_scale = 0
    print(len(test_data))
    for j in range(0, len(test_data)):
        for iters in range(20):
            midi_pattern.append([test_data[j][3 * iters], test_data[j][3 * iters+1], test_data[j][3 * iters+2]])
            #midi_pattern.append(pattern[i][j])
        #stats = get_all_stats(tune_song(midi_pattern))
        stats = get_all_stats(midi_pattern)
        stats_scale_tot += stats['scale_score']
        stats_repetitions_2_tot += stats['repetitions_2']
        stats_repetitions_3_tot += stats['repetitions_3']
        stats_tone_span_tot += stats['tone_span']
        num_null_rest += stats['num_null_rest']
        stats_rest_value += stats['average_rest']
        stats_songlength += stats['songlength']
        stats_unique_tones_tot += float(stats['tones_unique'])
        best_scale_score = max(stats['scale_score'], best_scale_score)
        best_repetitions_2 = max(stats['repetitions_2'], best_repetitions_2)
        best_repetitions_3 = max(stats['repetitions_3'], best_repetitions_3)
        longest_rest = max(stats['rest_max'], longest_rest)
        midi_pattern = []
        if stats['scale_score'] == 1.0:
            num_perfect_scale += 1
    print('Average scale score :', stats_scale_tot/len(test_data))
    print('Average repetitions of len 2 :', stats_repetitions_2_tot/len(test_data))
    print('Average repetitions of len 3 :', stats_repetitions_3_tot/len(test_data))
    print('Average unique tones :', stats_unique_tones_tot / len(test_data))
    print('Average tone span :', stats_tone_span_tot/len(test_data))
    print('num_null_rest :', num_null_rest/len(test_data))
    print('avg rest :', stats_rest_value/len(test_data))
    print('avg songlength :', stats_songlength/len(test_data))

    print("=================================================================================================\n")
    print('Best scale score :', best_scale_score)
    print('Longest repetition(s) of len 2 :', best_repetitions_2)
    print('Longest repetition(s) of len 3 :', best_repetitions_3)
    print('Number of perfect scale scores :', num_perfect_scale)
    print('Longest rest :', longest_rest)


if __name__ == "__main__":
  main()

