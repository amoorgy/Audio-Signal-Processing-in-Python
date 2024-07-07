import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft, fftshift, fftfreq


# Function to record audio
def record_audio(filename, duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    wav.write(filename, fs, recording)
    print(f"Recording saved to {filename}")

# Function to plot time signal
def plot_signal(signal, fs, title):
    t = np.arange(signal.shape[0]) / fs
    plt.figure()
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

# Function to scale and shift time signal
def scale_and_shift(signal, a, t0, fs):
    t = np.arange(signal.shape[0]) / fs
    y_t = np.interp(t, (t - t0) / a, signal, left=0, right=0)
    return y_t

# Function to add two signals
def add_signals(signal1, signal2):
    return signal1 + signal2

# Function to compute Fourier transform
def compute_fourier(signal):
    return fft(signal)

# Function to inverse Fourier transform
def inverse_fourier(signal_freq):
    return ifft(signal_freq).real

# Function to apply low pass filter
def low_pass_filter(signal_freq, cutoff, fs):
    freqs = np.fft.fftfreq(signal_freq.size, 1/fs)
    filter_mask = np.abs(freqs) <= cutoff
    return signal_freq * filter_mask

# Function to apply high pass filter
def high_pass_filter(signal_freq, cutoff, fs):
    freqs = np.fft.fftfreq(signal_freq.size, 1/fs)
    filter_mask = np.abs(freqs) > cutoff
    return signal_freq * filter_mask

# Function to play a WAV file
def play_wav(data,sample_rate):
    sd.play(data, sample_rate)
    sd.wait()

def triangular_filter(filepath, wc):
    samplerate, data = wav.read(filepath)

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/samplerate)

    response = np.zeros_like(xf)

    for i in range(len(xf)):
        if -wc <= xf[i] <= -wc/2:
            response[i] = (xf[i] + wc) / (wc/2)
        elif -wc/2 < xf[i] <= wc/2:
            response[i] = 1-2 * abs(xf[i])/wc
        elif wc/2 < xf[i] < wc:
            response[i] = (wc-xf[i])/(wc/2)

    yf_filtered = yf * response
    filtered_signal = ifft(yf_filtered)
    play_wav(filtered_signal.real,samplerate)
    plt.figure(figsize=(12,6))
    plt.plot(xf, response)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Triangular filter response')
    plt.show()


# Main script
fs = 44100  # Sampling frequency
duration = 5  # Duration in seconds
filename = 'recording.wav'

# Record audio
record_audio(filename, duration, fs)

# Load recorded audio
fs, signal = wav.read(filename)
signal = signal.flatten()

# Plot original time signal
plot_signal(signal, fs, 'Original Time Signal')

# Scale and shift the signal
a = 2  # Scaling factor
t0 = 0.5  # Time shift
scaled_shifted_signal = scale_and_shift(signal, a, t0, fs)

# Plot scaled and shifted signal
plot_signal(scaled_shifted_signal, fs, 'Scaled and Shifted Time Signal')

# Add scaled and shifted signal to original
added_signal = add_signals(signal, scaled_shifted_signal)

# Plot added signal
plot_signal(added_signal, fs, 'Added Signal')

# Compute Fourier transform
signal_freq = compute_fourier(signal)

# Plot Fourier transform
plt.figure()
plt.plot(np.fft.fftfreq(signal.size, 1/fs), np.abs(signal_freq))
plt.title('Fourier Transform')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()

# Shift Fourier transform by a frequency ws
ws = 1000  # Shift frequency
shifted_freq = fftshift(signal_freq)
shifted_signal = inverse_fourier(shifted_freq)

# Plot shifted signal
plot_signal(shifted_signal, fs, 'Shifted Time Signal')

# Apply low pass filter
cutoff = 1000  # Cutoff frequency
low_passed_freq = low_pass_filter(signal_freq, cutoff, fs)
low_passed_signal = inverse_fourier(low_passed_freq)

# Plot low passed signal
plot_signal(low_passed_signal, fs, 'Low Passed Signal')

# Apply high pass filter
high_passed_freq = high_pass_filter(signal_freq, cutoff, fs)
high_passed_signal = inverse_fourier(high_passed_freq)

# Plot high passed signal
plot_signal(high_passed_signal, fs, 'High Passed Signal')

# Save processed signals as WAV files
wav.write('scaled_shifted_signal.wav', fs, scaled_shifted_signal.astype(np.float32))
wav.write('added_signal.wav', fs, added_signal.astype(np.float32))
wav.write('low_passed_signal.wav', fs, low_passed_signal.astype(np.float32))
wav.write('high_passed_signal.wav', fs, high_passed_signal.astype(np.float32))

# Play all WAV files
sample_rate1, signal1 = wav.read("recording.wav")
sample_rate2, signal2 = wav.read("scaled_shifted_signal.wav")
sample_rate3, signal3 = wav.read("added_signal.wav")
sample_rate4, signal4 = wav.read("low_passed_signal.wav")
sample_rate5, signal5 = wav.read("high_passed_signal.wav")
triangular_filter("recording.wav", 4000)

play_wav(signal1, sample_rate1)
play_wav(signal2, sample_rate2)
play_wav(signal3, sample_rate3)
play_wav(signal4, sample_rate4)
play_wav(signal5, sample_rate5)

print("All tasks completed and files saved.")