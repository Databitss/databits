import os
import numpy as np
import librosa
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32  # Default batch size

class LoaderData:
    def __init__(self, data, sound_folder, batch_size=32, seq_length=512, n_mfcc=20, num_feature=8):
        self.data = data
        self.sound_folder = sound_folder
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_mfcc = n_mfcc
        self.num_feature = num_feature
        self.num_classes = 3  # Assuming 3 classes for the disease labels

    def segment_cough_sound(self, signal, sr, cough_threshold=0.05, min_cough_duration=0.1, padding=0.05):

        hop_length = int(min_cough_duration*sr)
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        energy = librosa.feature.rms(y=signal, hop_length=hop_length)[0]

        # Normalize the energy values
        normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

        # Set the energy threshold for event detection
        cough_threshold = np.max(normalized_energy) * cough_threshold
        min_cough_samples = round(sr * min_cough_duration)


        # Find the cough segments
        cough_segments = []
        event_start = None

        for i, value in enumerate(normalized_energy):
            if value >= cough_threshold:
                if event_start is None:
                    event_start = i*hop_length
            else:
                if event_start is not None:
                    cough_duration = i*hop_length - event_start
                    if cough_duration >= min_cough_samples:
                        event_end = i*hop_length + int(padding * sr)
                        event_start -= int(padding * sr)
                        event_start = max(event_start, 0)
                        cough_segments.append(signal[event_start: event_end+1])
                    event_start = None

        # Convert cough segments to time in seconds
        # cough_segments = [(start / sr, end / sr) for start, end in cough_segments]

        return cough_segments

    def extract_mfcc(self, file_path, n_mfcc=None, target_length=None):
        n_mfcc = n_mfcc if n_mfcc is not None else self.n_mfcc
        target_length = target_length if target_length is not None else self.seq_length
        try:
            audio, sr = librosa.load(file_path)
            cough_segments = self.segment_cough_sound(audio, sr)

            # If no cough segments found, use full audio
            if not cough_segments:
                print(f"No cough segments detected for {file_path}, using full signal.")
                segmented_audio = audio
            else:
                segmented_audio = np.concatenate(cough_segments)

            # Extract MFCC from the segmented audio
            mfcc = librosa.feature.mfcc(y=segmented_audio, sr=sr, n_mfcc=n_mfcc)
            mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

            # Pad or trim to target length
            if len(mfcc) > target_length:
                mfcc = mfcc[:target_length]
            elif len(mfcc) < target_length:
                pad_width = target_length - len(mfcc)
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

            return tf.convert_to_tensor(mfcc, dtype=tf.float32)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return tf.zeros((target_length, n_mfcc), dtype=tf.float32)


    def preprocess_features(self, row):
        age = row["age"] / 100.0
        pack_years = row["packYears"] / 100.0
        gender = row["gender"]
        tb_contact_history = row["tbContactHistory"]
        wheezing_history = row["wheezingHistory"]
        phlegm_cough = row["phlegmCough"]
        family_asthma_history = row["familyAsthmaHistory"]

        fever_history = row["feverHistory"]

        features = [
            age, gender, pack_years, tb_contact_history, wheezing_history,
            phlegm_cough, family_asthma_history, fever_history
        ]
        features = np.array(features).reshape(-1, 1)
        return features

    def process_row(self, row):
        candidate_id = row["candidateID"]
        audio_path = os.path.join(self.sound_folder, str(candidate_id), "cough.wav")
        mfcc = self.extract_mfcc(audio_path)
        features = self.preprocess_features(row)
        label = row["disease"]
        label = tf.one_hot(label, depth=3)
        return mfcc, features, label

    def data_generator(self, data):
        for _, row in data.iterrows():
            mfcc, features, label = self.process_row(row)
            yield mfcc, features, label

    def create_dataset(self, data, batch_size=BATCH_SIZE):
        output_signature = (
            tf.TensorSpec(shape=(self.seq_length, self.n_mfcc), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_feature, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(data),
            output_signature=output_signature
        )
        return dataset.batch(batch_size).shuffle(256).prefetch(tf.data.AUTOTUNE)

    def split_data(self, data, test_size=0.2, random_state=42):
        train_data, valid_data = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data['disease']
        )
        return train_data, valid_data

    def train_valid_split(self, data):
        train_data_df, valid_data_df = self.split_data(data)
        train_dataset = self.create_dataset(train_data_df, batch_size=BATCH_SIZE)
        valid_dataset = self.create_dataset(valid_data_df, batch_size=BATCH_SIZE)
        return train_dataset, valid_dataset
