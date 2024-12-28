import os
import librosa
import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from scipy.signal import resample
from scipy.io.wavfile import write
from Model import CustomResNet


class AudioPipeline:
    def __init__(self, model, segment_duration=30, audio_frame_rate=44100):
        self.model = model
        self.segment_duration = segment_duration
        self.audio_frame_rate = audio_frame_rate

        self.audio_array = None
        self.audio_name = None
        self.cropped_audio = None

    def extract_audio(self, video_path):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f'File {video_path} does not exist.')
        self.audio_name = os.path.basename(video_path).split('.')[0] + '.wav'
        audio = AudioSegment.from_file(video_path)
        self.audio_array = np.array(audio.get_array_of_samples())
        self.audio_array = self.audio_array.reshape((audio.channels, -1))
        self.resample_audio(audio.frame_rate)

    def resample_audio(self, orig_sr):
        number_of_samples = round(self.audio_array.shape[1] * self.audio_frame_rate / orig_sr)
        self.audio_array = resample(self.audio_array, number_of_samples, axis=1)

    @staticmethod
    def get_spectrogram(audio):
        return librosa.amplitude_to_db(abs(librosa.stft(audio)))

    def load_crops(self):
        dir_name = self.audio_name.split('.')[0]
        os.mkdir(dir_name)
        for i in range(self.cropped_audio.shape[1]):
            crop = []
            for j in range(self.cropped_audio.shape[0]):
                crop.append(self.cropped_audio[j, i])
            crop = np.array(crop).T
            crop = crop.astype(np.int16)
            file_name = self.audio_name.replace('.wav', f'_{i}.wav')
            file_path = os.path.join(dir_name, file_name)
            write(file_path, self.audio_frame_rate, crop)

    def get_predicts(self):
        self.load_crops()
        names = []
        neg_class = []
        pos_class = []
        with (torch.no_grad()):
            for idx, crop in enumerate(self.cropped_audio[0]):
                self.model.eval()
                spec = self.get_spectrogram(crop)
                input = torch.from_numpy(spec.reshape((1, 1, spec.shape[0], spec.shape[1]))).float()
                output = self.model(input)
                neg_class.append(output[0][0].item())
                pos_class.append(output[0][1].item())
                names.append(self.audio_name.replace('.wav', f'_{idx}.wav'))
        data = pd.DataFrame()
        data['names'] = names
        data['neg_class'] = neg_class
        data['pos_class'] = pos_class
        dir_name = self.audio_name.split('.')[0]
        file_name = self.audio_name.replace('wav', 'csv')
        data.to_csv(os.path.join(dir_name, file_name), index=False)

    def crop_audio(self):
        self.cropped_audio = []
        num_segments = self.audio_array.shape[1] // (self.segment_duration * self.audio_frame_rate)
        for channel in self.audio_array:
            cropped_channel = []
            end = 0
            for i in range(num_segments):
                start = i * self.segment_duration * self.audio_frame_rate
                end = start + self.segment_duration * self.audio_frame_rate
                segment = channel[start: end]
                cropped_channel.append(segment)
            padding_size = end + self.segment_duration * self.audio_frame_rate - len(channel)
            padding = np.zeros(padding_size)
            cropped_channel.append(np.hstack((channel[end:], padding)))
            self.cropped_audio.append(np.array(cropped_channel))
        self.cropped_audio = np.array(self.cropped_audio)


if __name__ == '__main__':
    pip = AudioPipeline(model=CustomResNet(2), audio_frame_rate=8000, segment_duration=3)
    path = 'resources/vid.MP4'
    pip.extract_audio(path)
    pip.crop_audio()
    pip.get_predicts()
