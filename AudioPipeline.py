import argparse
import os
import librosa
import numpy as np
import pandas as pd
import random
import torch
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from scipy.io.wavfile import write
from Model import CustomResNet

class AudioProcessor:
    def __init__(self, model, weights_path, GPU, segment_duration=30, audio_frame_rate=44100):
        self.model = model
        if GPU:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        else:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True), map_location='cpu', )
        self.model.eval()

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
        self.audio_array = self.audio_array.T
        self.resample_audio(sample_rate)

    def resample_audio(self, orig_sr):
        number_of_samples = round(self.audio_array.shape[1] * self.audio_frame_rate / orig_sr)
        self.audio_array = resample(self.audio_array, number_of_samples, axis=1)

    @staticmethod
    def get_spectrogram(audio):
        return librosa.amplitude_to_db(abs(librosa.stft(audio)))

    def load_crops(self, output_dir, csv=None):
        dir_name = os.path.join(output_dir, self.audio_name.split('.')[0])
        os.makedirs(dir_name, exist_ok=True)
        for i in range(self.cropped_audio.shape[1]):
            crop = []
            if csv:
                for j in range(self.cropped_audio.shape[0]):
                    if csv[j]['label'] == 1:
                        crop.append(self.cropped_audio[j, i])
            else:
                for j in range(self.cropped_audio.shape[0]):
                    crop.append(self.cropped_audio[j, i])
            crop = np.array(crop).T
            crop = crop.astype(np.int16)
            file_name = self.audio_name.replace('.wav', f'_{i}.wav')
            file_path = os.path.join(dir_name, file_name)
            write(file_path, self.audio_frame_rate, crop)

    def my_softmax(self, x):
        x = np.array(x)
        return (np.exp(x)/np.sum(np.exp(x)))

    def get_predicts(self, output_dir, save_status):
        dir_name = os.path.join(output_dir, self.audio_name.split('.')[0])
        os.makedirs(dir_name, exist_ok=True)
        names = []
        neg_class = []
        pos_class = []
        labels = []
        with (torch.no_grad()):
            for idx, crop in enumerate(self.cropped_audio[0]):
                self.model.eval()
                spec = self.get_spectrogram(crop)
                input = torch.from_numpy(spec.reshape((1, 1, spec.shape[0], spec.shape[1]))).float()
                output = self.model(input)
                output = self.my_softmax(output)
                neg_class.append(output[0][0].item())
                pos_class.append(output[0][1].item())
                labels.append(output[0].argmax().item())
                names.append(self.audio_name.replace('.wav', f'_{idx}.wav'))
        data = pd.DataFrame()
        data['names'] = names
        data['neg_class'] = neg_class
        data['pos_class'] = pos_class
        data['label'] = labels
        dir_name = self.audio_name.split('.')[0]
        file_name = self.audio_name.replace('wav', 'csv')
        data.to_csv(os.path.join(output_dir, os.path.join(dir_name, file_name)), index=False)
        if save_status == 2:
            self.load_crops(output_dir, None)
        elif save_status == 1:
            self.load_crops(output_dir, data)

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
    def process_file(filename, output_path, weights_path, audio_frame_rate, segment_duration, video, save_status, GPU):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} does not exist.')
        processor = AudioProcessor(model=CustomResNet(2), weights_path=weights_path, GPU=GPU, audio_frame_rate=audio_frame_rate,
                                   segment_duration=segment_duration)
        if video:
            processor.extract_audio(filename)
        else:
            processor.read_audio(filename)
        processor.crop_audio()
        processor.get_predicts(output_path, save_status)

    @staticmethod
    def process_csv(filename, output_path, weights_path, audio_frame_rate, segment_duration, video, save_status, GPU):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'File {filename} does not exist.')
        paths = pd.read_csv(filename)
        paths = paths.iloc[:, 0].tolist()
        for path in paths:
            AudioPipeline.process_file(path, output_path, weights_path, audio_frame_rate, segment_duration, video, save_status, GPU)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default='',
                        help="Path to .csv file with pathes to audios/videos. If empty, audio is to be extracted from file_path", type=str)
    parser.add_argument("--file_path", default='',
                        help="Path to audio/video file.", type=str)
    parser.add_argument("--weights_path", default='',
                        help="Path to weights of model.", type=str)
    parser.add_argument("--video", default=False,
                        help="If True - audio is to be extracted from video, if False - from .wav file.", type=bool)
    parser.add_argument("--output_path", default = './',
                        help="Place where output csv will be saved", type=str)
    parser.add_argument("--segment_duration", default = 31,
                        help="Duration of segmentes", type=int)
    parser.add_argument("--audio_frame_rate", default=8000, 
                        help="Audio frame rate", type=int)
    parser.add_argument("--save_all_audio", default = False,
                        help="If True, ALL fragments will be saved", type=bool)
    parser.add_argument("--save_audio", default = False,
                        help="If True, fragments with positive class will be saved. If --save_all_audio is True, ALL fragments will be saved", type=bool)
    parser.add_argument("--GPU", default = True,
                        help="If True - GPU. If False - CPU", type=bool)
    args = parser.parse_args()

    segment_duration = args.segment_duration
    audio_frame_rate = args.audio_frame_rate
    save_status = 0
    if args.save_audio:
        save_status = 1
    if args.save_all_audio:
        save_status = 2

    if not args.output_path:
        raise ValueError("The --output_path argument is required to specify the directory for saving results.")
    if not args.csv_path and not args.file_path:
        raise ValueError("Either --csv_path or --file_path must be provided.")
    if args.csv_path and args.file_path:
        raise ValueError(
            "You cannot specify both --csv_path and --file_path at the same time. Please provide only one.")
    if args.file_path:
        AudioPipeline.process_file(args.file_path, args.output_path, args.weights_path, audio_frame_rate, segment_duration, args.video, save_status, args.GPU)
    else:
        AudioPipeline.process_csv(args.csv_path, args.output_path, args.weights_path, audio_frame_rate, segment_duration, args.video, save_status, args.GPU)

if __name__ == '__main__':
    main()
