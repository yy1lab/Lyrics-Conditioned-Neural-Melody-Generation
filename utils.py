import math
import midi
import pretty_midi
import json
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def get_syll_data(full_data, path_model, songlength, num_songs):

    syllModel = Word2Vec.load(path_model)
    syll2Vec = syllModel.wv['yo']
    full_data = full_data[:, 0]
    syll_matrix = np.zeros(shape=(num_songs, len(syll2Vec) * songlength))

    for i in range(0, num_songs):
        for j in range(0, songlength):
            try:
                syll_matrix[i][j * len(syll2Vec):(j + 1) * len(syll2Vec)] = syllModel.wv[full_data[i][0][j][0]]
            except:
                a = 0

    return syll_matrix


def get_midi_data(full_data, songlength, num_songs, num_midi_features):
    full_data = full_data[:, 0]
    data_matrix = np.zeros(shape=(num_songs, num_midi_features * songlength))
    for i in range(0, num_songs):
        for j in range(0, songlength):
            for feat in range(0, num_midi_features):
                try:
                    data_matrix[i][j * num_midi_features + feat] = full_data[i][0][j][1][feat]
                except:
                    data_matrix[i][j * num_midi_features + feat] = 0

    return data_matrix


def load_settings_from_file(settings):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    # check for settings missing in file
    for key in settings.keys():
        if not key in settings_loaded:
            print(key, 'not found in loaded settings - adopting value from command line defaults: ', settings[key])
            # overwrite parsed/default settings with those read from file, allowing for
    # (potentially new) default settings not present in file
    settings.update(settings_loaded)
    return settings


def cents_to_pitchwheel_units(cents):
    return int(40.96*(float(cents)))


def tone_to_freq(tone):
    """
      returns the frequency of a tone.

      formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
    """
    return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0


def freq_to_tone(freq):
    """
      returns a dict d where
      d['tone'] is the base tone in midi standard
      d['cents'] is the cents to make the tone into the exact-ish frequency provided.
                 multiply this with 8192 to get the midi pitch level.

      formulas from
        * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
        * https://en.wikipedia.org/wiki/Cent_(music)
    """
    if freq <= 0.0:
        return None
    float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
    int_tone = int(float_tone)
    cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
    return {'tone': int_tone, 'cents': cents}


def save_midi_pattern(filename, midi_pattern):
    if filename is not None:
        midi.write_midifile(filename, midi_pattern)


def get_batch_multi(batch_size, pointer, train, validate, test, num_midi_features, num_syllables_per_sentence,
                    part='train'):
    batch_songs = []
    batch_meta = []
    batch_meta_wrong = []
    if part == 'train':
        batch_size = min(batch_size, np.shape(train[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(train[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(train[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(train[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(train[i-1][num_midi_features*num_syllables_per_sentence:None])
            except:
                break

    elif part == 'validation':
        batch_size = min(batch_size, np.shape(validate[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(validate[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(validate[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(validate[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(validate[i-1][num_midi_features*num_syllables_per_sentence:None])
            except:
                break

    else:
        batch_size = min(batch_size, np.shape(test[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(test[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(test[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(test[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(test[i-1][num_midi_features*num_syllables_per_sentence:None])

            except:
                break

    try:
        batch_songs_list = np.split(np.asarray(batch_songs), indices_or_sections=num_syllables_per_sentence, axis=1)
        batch_songs_list = np.transpose(batch_songs_list, axes=(1, 0, 2))
        batch_meta_list = np.split(np.asarray(batch_meta), indices_or_sections=num_syllables_per_sentence, axis=1)
        batch_meta_list = np.transpose(batch_meta_list, axes=(1, 0, 2))
        batch_meta_wrong_list = np.split(np.asarray(batch_meta_wrong), indices_or_sections=num_syllables_per_sentence,
                                         axis=1)
        batch_meta_wrong_list = np.transpose(batch_meta_wrong_list, axes=(1, 0, 2))

    except:
        batch_songs_list = None
        batch_meta_list = None
        batch_meta_wrong_list = None

    pointer = pointer + batch_size
    return batch_songs_list, batch_meta_list, batch_meta_wrong_list, pointer


def get_batch(batch_size, pointer, train, validate, test, num_midi_features, num_syllables_per_sentence,
              num_syllables_features, part='train'):
    batch_songs = []
    batch_meta = []
    batch_meta_wrong = []
    if part == 'train':
        batch_size = min(batch_size, np.shape(train[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(train[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(train[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(train[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(train[i-1][num_midi_features*num_syllables_per_sentence:None])
            except:
                break

    elif part == 'validation':
        batch_size = min(batch_size, np.shape(validate[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(validate[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(validate[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(validate[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(validate[i-1][num_midi_features*num_syllables_per_sentence:None])
            except:
                break

    else:
        batch_size = min(batch_size, np.shape(test[pointer:None])[0])
        for i in range(pointer, pointer+batch_size):
            try:
                batch_songs.append(test[i][0:num_midi_features*num_syllables_per_sentence])
                batch_meta.append(test[i][num_midi_features*num_syllables_per_sentence:None])
                if i == pointer:
                    batch_meta_wrong.append(test[i+batch_size-1][num_midi_features*num_syllables_per_sentence:None])
                else:
                    batch_meta_wrong.append(test[i-1][num_midi_features*num_syllables_per_sentence:None])

            except:
                break

    try:
        batch_songs_list = np.split(np.asarray(batch_songs), indices_or_sections=num_syllables_per_sentence, axis=1)
        batch_songs_list = np.transpose(batch_songs_list, axes=(1, 0, 2))
        batch_meta_list = np.split(np.asarray(batch_meta),
                                   indices_or_sections=num_syllables_per_sentence*num_syllables_features, axis=1)
        batch_meta_wrong_list = np.split(np.asarray(batch_meta_wrong),
                                         indices_or_sections=num_syllables_per_sentence*num_syllables_features, axis=1)

    except:
        batch_songs_list = None
        batch_meta = None
        batch_meta_wrong = None

    pointer = pointer + batch_size
    return batch_songs_list, batch_meta, batch_meta_wrong, pointer


def get_batch_no_cond(batch_size, pointer, train, validate, test, num_midi_features, num_syllables_per_sentence,
                      num_syllables_features, part='train'):
    batch_songs = []
    if part == 'train':
        batch_size = min(batch_size, np.shape(train[pointer:None])[0])
        try:
            for i in range(pointer, pointer+batch_size):
                batch_songs.append(train[i][0:num_midi_features*num_syllables_per_sentence])
        except:
            batch_songs = None

    elif part == 'validation':
        batch_size = min(batch_size, np.shape(validate[pointer:None])[0])
        try:
            for i in range(pointer, pointer+batch_size):
                batch_songs.append(validate[i][0:num_midi_features*num_syllables_per_sentence])
        except:
            batch_songs = None

    else:
        batch_size = min(batch_size, np.shape(test[pointer:None])[0])
        try:
            for i in range(pointer, pointer+batch_size):
                batch_songs.append(test[i][0:num_midi_features*num_syllables_per_sentence])
        except:
            batch_songs = None

    try:
        batch_songs_list = np.split(np.asarray(batch_songs), indices_or_sections=num_syllables_per_sentence, axis=1)
        batch_songs_list = np.transpose(batch_songs_list, axes=(1, 0, 2))

    except:
        batch_songs_list = None

    pointer = pointer + batch_size
    return batch_songs_list, pointer


def generate_fake_data(num_midi_features, num_syllables_features, num_syllables_per_sentence, num_songs,
                       dataset_type='Sentence level'):
    labels = []
    if dataset_type == 'Sentence level':
        for i in range(0, num_syllables_per_sentence):
            labels.append('time' + str(i))
            labels.append('duration' + str(i))
            labels.append('freq' + str(i))
            labels.append('velocity' + str(i))
        for i in range(0, num_syllables_per_sentence):
            for j in range(0, num_syllables_features):
                labels.append('feature' + str(j) + 'syllable' + str(i))
        print(num_songs,num_syllables_per_sentence,num_midi_features,num_syllables_features)
        df = pd.DataFrame(120*np.random.random(size=(num_songs, num_syllables_per_sentence *
                                               (num_midi_features + num_syllables_features))), columns=labels)

    return df


def normalise_data(train, vali, test, low=-1, high=1):
    """ Apply some sort of whitening procedure
    """
    mean = np.mean(np.vstack([train, vali]), axis=(0, 1))
    std = np.std(np.vstack([train-mean, vali-mean]), axis=(0, 1))

    normalised_train = (train - mean)/std
    normalised_vali = (vali - mean)/std
    normalised_test = (test - mean)/std

    return normalised_train, normalised_vali, normalised_test


def denormalise_data(data, mean, std):
    denormalised_data = data*std + mean

    return denormalised_data


def create_midi_pattern(sample, tones_per_cell, num_features_per_tone):
    # Create and save midi pattern
    output_ticks_per_quarter_note = 384.0
    midi_pattern = midi.Pattern([], resolution=int(output_ticks_per_quarter_note))
    cur_track = midi.Track([])
    cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=100))
    future_events = {}
    last_event_tick = 0
    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    abs_tick_note_beginning = 0.0
    for frame in sample:
        abs_tick_note_beginning += frame[0]
        for subframe in range(tones_per_cell):
            offset = subframe * num_features_per_tone
            tick_len = int(round(frame[offset + 1]))
            freq = frame[offset + 2]
            velocity = min(int(round(frame[offset + 3])), 127)
            d = freq_to_tone(freq)
            if d is not None and velocity > 0 and tick_len > 0:
                # range-check with preserved tone, changed one octave:
                tone = d['tone']
                while tone < 0:
                    tone += 12
                while tone > 127:
                    tone -= 12
                pitch_wheel = cents_to_pitchwheel_units(d['cents'])
                song_events_absolute_ticks.append((abs_tick_note_beginning,
                                                   midi.events.NoteOnEvent(
                                                       tick=0,
                                                       velocity=velocity,
                                                       pitch=tone)))
                song_events_absolute_ticks.append((abs_tick_note_beginning + tick_len,
                                                   midi.events.NoteOffEvent(
                                                       tick=0,
                                                       velocity=0,
                                                       pitch=tone)))
    song_events_absolute_ticks.sort(key=lambda e: e[0])
    abs_tick_note_beginning = 0.0
    for abs_tick, event in song_events_absolute_ticks:
        rel_tick = abs_tick - abs_tick_note_beginning
        event.tick = int(round(rel_tick))
        cur_track.append(event)
        abs_tick_note_beginning = abs_tick

    cur_track.append(midi.EndOfTrackEvent(tick=int(output_ticks_per_quarter_note)))
    midi_pattern.append(cur_track)

    return midi_pattern


def discretize(sample):

    dist = np.inf
    authorized_values_pitch = range(127)
    authorized_values_duration = [0.25,  0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 16., 32.]
    authorized_values_rest = [0., 1., 2., 4., 8., 16., 32.]
    discretized_sample = np.zeros(shape=np.shape(sample))
    discretized_sample_arrays = []
    for i in range(len(sample)):
        for j in range(0, len(authorized_values_pitch)):
            if (sample[i][0] - authorized_values_pitch[j]) ** 2 < dist:
                dist = (sample[i][0] - authorized_values_pitch[j]) ** 2
                discretized_sample[i][0] = authorized_values_pitch[j]
        dist = np.inf
        for j in range(0, len(authorized_values_duration)):
            if (sample[i][1] - authorized_values_duration[j]) ** 2 < dist:
                dist = (sample[i][1] - authorized_values_duration[j]) ** 2
                discretized_sample[i][1] = authorized_values_duration[j]
        dist = np.inf
        for j in range(0, len(authorized_values_rest)):
            if (sample[i][2] - authorized_values_rest[j]) ** 2 < dist:
                dist = (sample[i][2] - authorized_values_rest[j]) ** 2
                discretized_sample[i][2] = authorized_values_rest[j]
        dist = np.inf
        discretized_sample_arrays.append(np.asarray(discretized_sample[i][:]))

    return discretized_sample_arrays


def create_midi_pattern_from_discretized_data(discretized_sample):
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
    tempo = 120
    ActualTime = 0  # Time since the beginning of the song, in seconds
    for i in range(0,len(discretized_sample)):
        length = discretized_sample[i][1] * 60 / tempo  # Conversion Duration to Time
        if i < len(discretized_sample) - 1:
            gap = discretized_sample[i + 1][2] * 60 / tempo
        else:
            gap = 0  # The Last element doesn't have a gap
        note = pretty_midi.Note(velocity=100, pitch=int(discretized_sample[i][0]), start=ActualTime,
                                end=ActualTime + length)
        voice.notes.append(note)
        ActualTime += length + gap  # Update of the time

    new_midi.instruments.append(voice)

    return new_midi


def mmd2(x, y, sigma=1):

    print(len(x))
    print(len(y))
    var_x = var_u(x, sigma)
    var_y = var_u(y, sigma)
    covar = covar_u(x, y, sigma)
    print("MMD2_x", var_x)
    print("MMD2_y", var_y)
    print("MMD2_xy", covar)

    return var_x - covar + var_y


def var_u(x, sigma):
    var_u = 0
    n = len(x)
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                var_u += rbf(x[i], x[j], sigma)

    n = np.double(len(x))
    print(1./(n*(n-1)) * var_u)
    return 1./(n*(n-1)) * var_u


def covar_u(x, y, sigma):
    n = len(x)
    m = len(y)
    covar_u = 0

    for i in range(0, n):
        for j in range(0, m):
            covar_u += rbf(x[i], y[j], sigma)

    n = np.double(n)
    m = np.double(m)
    return 2./(m * n) * covar_u


def rbf(x, y, sigma):
    frob = (sum((x - y) ** 2))

    return np.exp(-frob/((2*sigma)**2))


def print_model_stats(model_stats, num_songs):
    model_stats['stats_scale_tot'] = model_stats['stats_scale_tot'] / num_songs
    model_stats['stats_repetitions_2_tot'] = model_stats['stats_repetitions_2_tot'] / num_songs
    model_stats['stats_repetitions_3_tot'] = model_stats['stats_repetitions_3_tot'] / num_songs
    model_stats['stats_span_tot'] = model_stats['stats_span_tot'] / num_songs
    model_stats['stats_unique_tones_tot'] = model_stats['stats_unique_tones_tot'] / num_songs
    model_stats['stats_avg_rest_tot'] = model_stats['stats_avg_rest_tot'] / num_songs
    model_stats['num_of_null_rest_tot'] = model_stats['num_of_null_rest_tot'] / num_songs
    model_stats['songlength_tot'] = model_stats['songlength_tot'] / num_songs
    print('Average scale score :', model_stats['stats_scale_tot'])
    print('Average repetitions of len 2 :', model_stats['stats_repetitions_2_tot'])
    print('Average repetitions of len 3 :', model_stats['stats_repetitions_3_tot'])
    print('Average span (in tones) :', model_stats['stats_span_tot'])
    print('Average unique tones :', model_stats['stats_unique_tones_tot'])
    print('Average rest over all notes :', model_stats['stats_avg_rest_tot'])
    print('Average number of null rests :', model_stats['num_of_null_rest_tot'])
    print('Average songlength :', model_stats['songlength_tot'])
    print("*************************************************************************************************\n")
    print("BEST STATS OVER {} GENERATED SONGS============================================================\n"
          .format(num_songs))
    print('Best scale score :', model_stats['best_scale_score'])
    print('Highest num of repetition(s) of len 2 :', model_stats['best_repetitions_2'])
    print('Highest num of repetition(s) of len 3 :', model_stats['best_repetitions_3'])
    print('Number of perfect scale scores :', model_stats['num_perfect_scale'])
    print('Number of "good" songs :', model_stats['num_good_songs'])
    print("*************************************************************************************************\n")


def get_model_stats(model_stats, num_songs):

    average_model_stats = {}

    average_model_stats['stats_scale'] = model_stats['stats_scale_tot'] / num_songs
    average_model_stats['stats_repetitions_2'] = model_stats['stats_repetitions_2_tot'] / num_songs
    average_model_stats['stats_repetitions_3'] = model_stats['stats_repetitions_3_tot'] / num_songs
    average_model_stats['stats_span'] = model_stats['stats_span_tot'] / num_songs
    average_model_stats['stats_unique_tones'] = model_stats['stats_unique_tones_tot'] / num_songs
    average_model_stats['stats_avg_rest'] = model_stats['stats_avg_rest_tot'] / num_songs
    average_model_stats['num_of_null_rest'] = model_stats['num_of_null_rest_tot'] / num_songs
    average_model_stats['songlength'] = model_stats['songlength_tot'] / num_songs

    return average_model_stats


def discretize_length(sample):

    dist = np.inf
    authorized_values_duration = [0.25,  0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 16., 32.]
    discretized_sample = np.zeros(shape=np.shape(sample))
    discretized_sample_arrays = []
    for i in range(len(sample)):
        for j in range(0, len(authorized_values_duration)):
            if (sample[i][0] - authorized_values_duration[j]) ** 2 < dist:
                dist = (sample[i][0] - authorized_values_duration[j]) ** 2
                discretized_sample[i][0] = authorized_values_duration[j]
        dist = np.inf
        discretized_sample_arrays.append(np.asarray(discretized_sample[i][:]))

    return discretized_sample_arrays


def discretize_rest(sample):

    dist = np.inf
    authorized_values_rest = [0, 1, 2, 4, 8, 16, 32]
    discretized_sample = np.zeros(shape=np.shape(sample))
    discretized_sample_arrays = []
    for i in range(len(sample)):
        for j in range(0, len(authorized_values_rest)):
            if (sample[i][0] - authorized_values_rest[j]) ** 2 < dist:
                dist = (sample[i][0] - authorized_values_rest[j]) ** 2
                discretized_sample[i][0] = authorized_values_rest[j]
        dist = np.inf
        discretized_sample_arrays.append(np.asarray(discretized_sample[i][:]))

    return discretized_sample_arrays


def discretize_pitch(sample):

    dist = np.inf
    authorized_values_pitch = range(127)
    discretized_sample = np.zeros(shape=np.shape(sample))
    discretized_sample_arrays = []
    for i in range(len(sample)):
        for j in range(0, len(authorized_values_pitch)):
            if (sample[i][0] - authorized_values_pitch[j]) ** 2 < dist:
                dist = (sample[i][0] - authorized_values_pitch[j]) ** 2
                discretized_sample[i][0] = authorized_values_pitch[j]
        dist = np.inf
        discretized_sample_arrays.append(np.asarray(discretized_sample[i][:]))

    return discretized_sample_arrays
