import numpy as np
import librosa

def wav2mel(wav_filename,
            sample_rate = 20510,
            fft_size = 512,
            num_channels = 80,
            num_frames = 100):

    """
    for converting a wav audio file to mel spectrogram

    :param wav_filename: directory of the wav file
    :param sample_rate: sample rate of audio file TODO: automate this
    :param fft_size: fast fourier transform bins
    :param num_channels: number of channels (y axis)
    :param num_frames: number of frames (x axis)
    :return: np.array of mel spectrogram
    """ 

    audio, sr = librosa.load(wav_filename, sr = sample_rate, mono=True)
    
    # Apply pre-emphasis filter
    pre_emphasis = 0.95
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Define frame length and stride in samples
    frame_length = int(sr * 0.025)  # 25ms
    hop_length = int(sr * 0.01)  # 10ms

    # Compute the power spectrum using a 512-point FFT
    power_spectrum = np.abs(librosa.stft(emphasized_audio, 
                                         n_fft=fft_size, 
                                         hop_length=hop_length, 
                                         win_length=frame_length))**2

    # Compute the filter banks with 40 triangular filters
    filter_banks = librosa.filters.mel(n_fft = fft_size, 
                                       sr = sr, 
                                       n_mels=num_channels)

    # Apply the filter banks to the power spectrum
    mel_spec = np.dot(filter_banks, power_spectrum)

    # Crop or pad to passed frames by repeating the last frame
    current_steps = mel_spec.shape[1]
    if current_steps < num_frames:
        padding = np.tile(mel_spec[:, -1:], (1, num_frames - current_steps))
        mel_spec = np.hstack((mel_spec, padding))
    elif current_steps > num_frames:
        mel_spec = mel_spec[:, :num_frames]

    # Convert power spectrogram to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db