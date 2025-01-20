import argparse
import os
import librosa
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from scipy.io.wavfile import write
from Model import CustomResNet


class AudioProcessor:
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

    def read_audio(self, audio_path):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f'File {audio_path} does not exist.')
        self.audio_name = os.path.basename(audio_path).split('.')[0] + '.wav'
        self.audio_array, sample_rate = sf.read(audio_path)
        self.resample_audio(sample_rate)

    def resample_audio(self, orig_sr):
        number_of_samples = round(self.audio_array.shape[1] * self.audio_frame_rate / orig_sr)
        self.audio_array = resample(self.audio_array, number_of_samples, axis=1)

    @staticmethod
    def get_spectrogram(audio):
        return librosa.amplitude_to_db(abs(librosa.stft(audio)))

    def load_crops(self, output_dir):
        dir_name = os.path.join(output_dir, self.audio_name.split('.')[0])
        os.makedirs(dir_name)
        for i in range(self.cropped_audio.shape[1]):
            crop = []
            for j in range(self.cropped_audio.shape[0]):
                crop.append(self.cropped_audio[j, i])
            crop = np.array(crop).T
            crop = crop.astype(np.int16)
            file_name = self.audio_name.replace('.wav', f'_{i}.wav')
            file_path = os.path.join(dir_name, file_name)
            write(file_path, self.audio_frame_rate, crop)

    def get_predicts(self, output_dir):
        self.load_crops(output_dir)
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
        data.to_csv(os.path.join(output_dir, os.path.join(dir_name, file_name)), index=False)

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


class AudioPipeline:
    @staticmethod
    def process_file(filename, output_path, audio_frame_rate, segment_duration, video):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} does not exist.')
        processor = AudioProcessor(model=CustomResNet(2), audio_frame_rate=audio_frame_rate,
                                   segment_duration=segment_duration)
        if video:
            processor.extract_audio(filename)
        else:
            processor.read_audio(filename)
        processor.crop_audio()
        processor.get_predicts(output_path)

    @staticmethod
    def process_csv(filename, output_path, audio_frame_rate, segment_duration, video):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} does not exist.')
        paths = pd.read_csv(filename)
        paths = paths.iloc[:, 0].tolist()
        for path in paths:
            AudioPipeline.process_file(path, output_path, audio_frame_rate, segment_duration, video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default='', type=str)
    parser.add_argument("--video", default=False, type=bool)
    parser.add_argument("--file_path", default='', type=str)
    parser.add_argument("--output_path", default='', type=str)
    parser.add_argument("--segment_duration", default=3, type=int)
    parser.add_argument("--audio_frame_rate", default=8000, type=int)
    args = parser.parse_args()

    segment_duration = args.segment_duration
    audio_frame_rate = args.audio_frame_rate
    if not args.output_path:
        raise ValueError("The --output_path argument is required to specify the directory for saving results.")
    if not args.csv_path and not args.file_path:
        raise ValueError("Either --csv_path or --file_path must be provided.")
    if args.csv_path and args.file_path:
        raise ValueError(
            "You cannot specify both --csv_path and --file_path at the same time. Please provide only one.")
    if args.file_path:
        AudioPipeline.process_file(args.file_path, args.output_path, audio_frame_rate, segment_duration, args.video)
    else:
        AudioPipeline.process_csv(args.csv_path, args.output_path, audio_frame_rate, segment_duration, args.video)


if __name__ == '__main__':
    main()
