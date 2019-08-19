
If you use our lyrics-melody dataset and lyrics embedding, please kindly cite our paper
"Conditional LSTM-GAN for Melody Generation from Lyrics" available at https://arxiv.org/pdf/1908.05551.pdf (more details about lyrics-melody dataset and melody generation from lyrics)

Our 12 melodies used in  subjective evaluation of this paper are provided. If you have any questions, please let us know.
Contact Yi Yu by yiyu@nii.ac.jp.

# Lyrics-Conditioned Neural Melody Generation


This is the dataset parsed and used for the project Lyrics Conditioned Neural Melody Generation.


The data comes from the LAKH Midi Dataset lmd-full (downloadable at this url : https://colinraffel.com/projects/lmd/). Only english songs were used from the dataset.
To download the MIDI files corresponding the .npy files from the dataset, you can search the names of the files in both dataset, that are unchanged and serve as ID.

The parsing is as follow :

— The syllable parsing :
  This format is the lowest level that pair together every notesand the corresponding syllable and it’ss attributes.

— The word parsing :
  This format regroups every notes of a word and gives the attributesof every syllables that makes the word.


— The Sentence parsing :
  Similarly, this format put together every notes that forms asyllable (or in most case, a lyric line) and it’s corresponding attributes.


— The Sentence and word parsing :
  Using the two last mentionned format, this one consist ofparsing the lyrics and notes in sentences and, whithin these sentences, to separateeverything in words.1
  
One note always containing one and only one syllable.
We parsed every songs in continous attributes and discrete attributes.

The discrete attributes are, in order:

— The Pitch of the note :
  In music, the pitch is what decide of how the note should beplayed. We used the Midi note number as an unit of pitch, it can take any integervalue between 0 and 127.

— The duration of the note :
  The duration of the note in number of staves. It can be a quarter note, a half note, a whole note or more. The exhaustive set of values itcan take in our parsing is : [0.25 ;0.5 ;1 ;2 ;4 ;8 ;16 ;32].


— The duration of the rest before the note :
  This value can take the same numerical values as the Duration but it can also be null (so zero).


The continuous attributes are, in order:

— The start of the note : In seconds since the beginning of the sung song.

— The length of the note : In seconds.

— The frequency of the note : In Hertz.

— The velocity of the note : Mesured as an integer by the pretty_midi python package.



An example on the song Listen to the Rhythm of the falling rain in the syllable parsing is :

List    [74.0, 0.5, 0.0]		[0.0, 0.18595050000000057, 587.3295358348151, 110.0]

en  		[72.0, 1.0, 0.0]		[0.247933999999999, 0.24380176666666742, 523.2511306011972, 110.0]

to	  	[72.0, 0.5, 0.0]		[0.24793400000000076, 0.18595050000000057, 523.2511306011972, 110.0]

the	  	[69.0, 1.0, 0.0]		[0.24793400000000076, 0.24380176666666564, 440.0, 110.0]

rhy	  	[69.0, 0.5, 0.0]		[0.247933999999999, 0.18595050000000057, 440.0, 110.0]

thm	  	[67.0, 1.0, 0.0]		[0.24793400000000076, 0.24380176666666564, 391.99543598174927, 110.0]

of		  [67.0, 1.0, 0.0]		[0.247933999999999, 0.24380176666666742, 391.99543598174927, 110.0]

the		  [65.0, 0.5, 0.0]		[0.24793400000000076, 0.18595050000000057, 349.2282314330039, 110.0]

fall		[67.0, 2.5, 0.0]		[0.24793400000000076, 0.41322333333333283, 391.99543598174927, 110.0]

ing		  [65.0, 1.0, 0.0]		[0.49586799999999975, 0.24380176666666564, 349.2282314330039, 110.0]

rain		[65.0, 4.0, 0.0]		[0.247933999999999, 0.9917360000000013, 349.2282314330039, 110.0]

Tel		  [74.0, 1.0, 1.0]		[1.2396700000000003, 0.24380176666666742, 587.3295358348151, 110.0]

ling		[72.0, 0.5, 0.0]		[0.24793400000000076, 0.18595050000000057, 523.2511306011972, 110.0]

me		  [72.0, 1.0, 0.0]		[0.247933999999999, 0.24380176666666742, 523.2511306011972, 110.0]

just		[69.0, 0.5, 0.0]		[0.24793400000000076, 0.18595050000000057, 440.0, 110.0]

what		[69.0, 0.5, 0.0]		[0.24793400000000076, 0.1239669999999986, 440.0, 110.0]

a		    [67.0, 0.5, 0.0]		[0.1239669999999986, 0.12396700000000038, 391.99543598174927, 110.0]

fool		[69.0, 1.5, 0.0]		[0.12396700000000038, 0.37190100000000115, 440.0, 110.0]

I've		[72.0, 0.5, 0.0]		[0.49586799999999975, 0.18595050000000057, 523.2511306011972, 110.0]

been		[72.0, 2.0, 0.0]		[0.24793400000000076, 0.49586799999999975, 523.2511306011972, 110.0]
