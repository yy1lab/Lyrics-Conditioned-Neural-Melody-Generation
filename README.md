# Lyrics-Conditioned-Neural-Melody-Generation

This is the dataset parsed and used for the project Lyrics Conditioned Neural Melody.


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
The discrete attributes are :

— The Pitch of the note :
  In music, the pitch is what decide of how the note should beplayed. We used the Midi note number as an unit of pitch, it can take any integervalue between 0 and 127.

— The duration of the note :
  The duration of the note in number of staves. It can be a quarter note, a half note, a whole note or more. The exhaustive set of values itcan take in our parsing is : [0.25 ;0.5 ;1 ;2 ;4 ;8 ;16 ;32].


— The duration of the rest before the note :
  This value can take the same numerical values as the Duration but it can also be null (so zero).


The continuous attributes are :

— The start of the note : In seconds since the beginning of the sung song.

— The length of the note : In seconds.

— The frequency of the note : In Hertz.

— The velocity of the note : Mesured as an integer by the pretty_midi python package.
