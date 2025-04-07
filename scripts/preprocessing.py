import os 
import numpy as np 
import librosa.display
import matplotlib.pyplot as plt 

def get_mel_spectrogram_with_mask(file_path, sampling_rate=2250, duration=9, number_of_mel_bins=128):
    # Load the audio file to an array
    y, _ = librosa.load(file_path, sr=sampling_rate, duration=duration)
    
    # Define standard length of sound array to handle different video durations
    standard_length = duration*sampling_rate 

    # If too short: pad with zeros, if too long: cut to length
    if len(y) < standard_length:
        signal_length = len(y)             
        y = np.pad(y, pad_width=(0, standard_length-len(y)), mode='constant', constant_values=0)
    else: 
        signal_length = standard_length
        y = y[:standard_length]

    # Actually get the mel spectrogram. Log it and normalize it for nicer handling in CNN
    mel_spec_matrix = librosa.feature.melspectrogram(y=y, sr=sampling_rate, n_mels=number_of_mel_bins)
    mel_spec_matrix_dB = librosa.power_to_db(mel_spec_matrix, ref=np.max)

    # Add a mask channel, indicating what is signal (1) and what is padding (0)
    total_time_frames = mel_spec_matrix_dB.shape[1]
    signal_time_frames = int((signal_length/standard_length)*total_time_frames) #Copy original proportion
    mask = np.zeros_like(mel_spec_matrix_dB)
    mask[:, :signal_time_frames] = 1
    mel_stacked_with_mask = np.stack([mel_spec_matrix_dB, mask], axis=0)

    return mel_stacked_with_mask

# Plot setup to inspect mel spectrogram, uncomment and test to see a cool spectrogram!
'''
mel = get_mel_spectrogram_with_mask('TarotX6_B2_Hover_1.wav')[0]
plt.figure(figsize=(8,6))
librosa.display.specshow(mel, sr=22050, x_axis='time', y_axis='mel')
plt.title('Mel Spectrogram')
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()
'''

WAV_ROOT = r'C:\Users\isakg\Desktop\drone_data_root\wavs' # Where all gathered data is located
MEL_ROOT = r'C:\Users\isakg\Desktop\drone_data_root\mels' # Where it gets sent after melification
CATEGORIES = ['drone', 'non_drone'] # Defines the class labels (names of folders) inside mel and wav  

# A loop to convert wav files to mel spectrograms, then sort by label into two folders
count = 0
for category in CATEGORIES: 
    wav_dir = os.path.join(WAV_ROOT, category)
    mel_dir = os.path.join(MEL_ROOT, category)

    for file_name in os.listdir(wav_dir):
        if file_name.endswith('.wav'): # Make sure only target file type gets handled
            wav_full_path = os.path.join(wav_dir, file_name) # Full path of .wav file
            mel_no_extension = os.path.splitext(file_name)[0] # Strip away .wav extension (outputs a tuple)
            mel_npy = os.path.join(mel_dir, mel_no_extension + '.npy') # Add .npy extension

            if os.path.exists(mel_npy): # So we can run this script whenever we augment data
                count += 1 
                print(f'File nr {count}: {file_name} already exists - skipping.')
                continue

            mel_with_mask = get_mel_spectrogram_with_mask(wav_full_path) # Convert into mel spectrogram
            np.save(mel_npy, mel_with_mask) # Save to disk 
            count += 1 
            print(f'Saved mel spectrogram nr {count}: {mel_npy}')















