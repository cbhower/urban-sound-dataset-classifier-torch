import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class MidiReplayDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal, sample_rate)
        signal = self._mix_to_mono(signal)
        signal = self._trim(signal)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)

        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)

        return signal

    @staticmethod
    def _mix_to_mono(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _trim(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\Users\Christian\Documents\cbhower\cbhower\urban-sound-dataset-classifier-torch\UrbanSound8K\metadata\UrbanSound8K.csv"
    AUDIO_DIR = r"C:\Users\Christian\Documents\cbhower\cbhower\urban-sound-dataset-classifier-torch\UrbanSound8K\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 8820  # 4/10 of a second

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    usd = MidiReplayDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
