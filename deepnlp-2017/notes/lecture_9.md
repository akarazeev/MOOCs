### Lecture 9 - Speech Recognition (ASR)

Deep Learning Speech Systems

- Acoustic Representation
- Phonetic Representation
- Probabilistic speech recognition

Neural Networks:
- Hybrid neural networks
- Training losses
- Sequence discriminative training

Automatic speech recognition (ASR)

We don't pronounce punctuation. Recognize words, nicely formatted sentences.

Text-to-speech synthesis (TTS)

Noise become more salient topic.

Speaker adaptive models. IBM via voice dictation system. Speaker identification. Speech enhancement. "Cocktail party effect".

Fast Fourier transform (FFT) is still too high-dimensional.

Mel frequency representation.

In ASR term "utterance" is used instead of "sentence". Minimal unit is "phoneme". Set of 40-60 distinct sounds (X-SAMPA - like ASCII).

**Probabilistic speech recognition**

- Signal is a sequence of observations
- We want to find the most likely word sequence `w`
- Hidden Markov Models

Phonemes: "cat" -> /K/, /AE/, /T/

Most systems use a phonetic approach. Context dependent states. With LSTM using context dependent phones they were good. But nowadays most systems use `context dependent states` still.

Context dependent phonetic clustering

**Fundamental equation of speech recognition**

Decoder output most likely sequence w^ from all possible sequences for an observation sequence o: w^ = argmax P(w|o) = argmax P(o|w)P(w)

A product of _Acoustic model_ and _Language model_ scores: P(o|w) = sum P(o|c)P(c|p)P(p|w) where p - phone sequence, c - state sequence.

**Speech recognition as transduction**

Gaussian Mixture Models
- it's a dominant paradigm for ASR from 1990 to 2010
- model probability distribution of the acoustic features for each state
- train by EM algorithm alternating:
    - M: forced alignment computing the max-likelihood state sequence for each utterance
    - E: parameter (mu, sigma) estimation
- complex training procedures to incrementally fit increasing numbers of components per mixture:
    - more components, better fit
- given an alignment mapping audio frames to states, this is parallelizable by state
- hard to share parameters/data across states

**Forced alignment**

- hard or soft segmentation

**Decoding**

Main paradigms for NNs for speech:
- use NNs to compute nonlinear feature representation
- use NNs to estimate phonetic unit probabilities

Train the network as a classifier with a Softmax across the phonetic units.

**Hybrid NN decoding**

**CNNs**

- Time delay NNs - convolution in time
- CNNs in time or frequency domain - pooling in a frequency domain ... (male/female voice)
- Wavenet

**RNNs**

**Human parity in speech recognition**

- ensemble of BLSTMs
- i-vector for speaker normalization
    - it's an embedding of audio trained to discriminate between speakers (Speaker ID)

**Connectionist Temporal Classification (Graves et al.)**

- CTC introduces an optional blank symbol between the "real" labels
- always use soft targets
- don't scale by posterior

**Sequence discriminative training**

**Sequence2Sequence**

**Watch, Listen, Attend and Spell (Chung et al.)**

**Neural transducer (Jaitly et al.)**
