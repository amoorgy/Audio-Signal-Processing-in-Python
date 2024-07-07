### README.md

```markdown
# Audio Signal Processing in Python

## Overview

This project focuses on recording, processing, and playing back audio signals using Python. Various signal processing techniques, including Fourier transforms, filtering, and signal manipulation, are applied to analyze and modify the recorded audio.

## Features

- **Audio Recording:** Capture audio using the microphone and save it as a WAV file.
- **Signal Plotting:** Visualize the audio signal in the time domain.
- **Signal Manipulation:** Scale and shift the audio signal.
- **Signal Addition:** Combine two audio signals.
- **Fourier Transform:** Compute and visualize the Fourier transform of the audio signal.
- **Filtering:** Apply low-pass and high-pass filters to the audio signal.
- **Triangular Filter:** Apply a custom triangular filter to the audio signal.
- **Audio Playback:** Play the original and processed audio signals.

## Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- SoundDevice

You can install the required libraries using pip:

```bash
pip install numpy matplotlib scipy sounddevice
```

## Usage

### Recording Audio

Record audio using the microphone and save it as a WAV file.

```python
record_audio('recording.wav', duration=5, fs=44100)
```

### Plotting Time Signal

Plot the original time-domain signal.

```python
plot_signal(signal, fs, 'Original Time Signal')
```

### Signal Manipulation

Scale and shift the audio signal.

```python
scaled_shifted_signal = scale_and_shift(signal, a=2, t0=0.5, fs=44100)
plot_signal(scaled_shifted_signal, fs, 'Scaled and Shifted Time Signal')
```

### Adding Signals

Combine the original and manipulated signals.

```python
added_signal = add_signals(signal, scaled_shifted_signal)
plot_signal(added_signal, fs, 'Added Signal')
```

### Fourier Transform

Compute and plot the Fourier transform of the audio signal.

```python
signal_freq = compute_fourier(signal)
plt.plot(np.fft.fftfreq(signal.size, 1/fs), np.abs(signal_freq))
plt.title('Fourier Transform')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()
```

### Filtering

Apply low-pass and high-pass filters to the audio signal.

```python
low_passed_freq = low_pass_filter(signal_freq, cutoff=1000, fs=44100)
low_passed_signal = inverse_fourier(low_passed_freq)
plot_signal(low_passed_signal, fs, 'Low Passed Signal')

high_passed_freq = high_pass_filter(signal_freq, cutoff=1000, fs=44100)
high_passed_signal = inverse_fourier(high_passed_freq)
plot_signal(high_passed_signal, fs, 'High Passed Signal')
```

### Triangular Filter

Apply a custom triangular filter to the audio signal.

```python
triangular_filter('recording.wav', wc=4000)
```

### Audio Playback

Play the original and processed audio signals.

```python
play_wav(signal, fs)
play_wav(scaled_shifted_signal, fs)
play_wav(added_signal, fs)
play_wav(low_passed_signal, fs)
play_wav(high_passed_signal, fs)
```

## File Structure

- **audio_signal_processing.py:** Main script containing all the functions and the main script logic.
- **recording.wav:** Example WAV file recorded by the script.
- **scaled_shifted_signal.wav:** WAV file containing the scaled and shifted signal.
- **added_signal.wav:** WAV file containing the added signals.
- **low_passed_signal.wav:** WAV file containing the low-pass filtered signal.
- **high_passed_signal.wav:** WAV file containing the high-pass filtered signal.

## Acknowledgments

Special thanks to the open-source community for providing the libraries and resources that made this project possible.
```

Feel free to customize the README file and other details according to your specific needs and preferences.
