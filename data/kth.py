import random
import os
import numpy as np
import torch
from scipy import misc
from torch.utils.serialization import load_lua


class KTH(object):
    def __init__(self, train, data_root, seq_len=20, image_size=64):
        self.data_root = '../data/kth'
        self.seq_len = seq_len
        self.image_size = image_size 
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.data_root)

        self.seed_set = False

    def get_sequence(self):
        path = '../data/kth'
        list_f = [x for x in os.listdir(path)]

        for x in range(batch_size):
            rand_folder = random.choice(list_f)
            path_to_file = path + '/' + rand_folder
            file_name = random.choice(os.listdir(path_to_file))
            path_to_video = path_to_file + '/' + file_name
            vidcap = cv2.VideoCapture(path_to_video)
            n_frames = vidcap.get(7)
            frame_rate = vidcap.get(5)
            ret, frame = vidcap.read()
            # print(n_frames, rand_folder, file_name, frame_rate, frame.shape)
            stacked_frames = []
            while vidcap.isOpened():
                frame_id = vidcap.get(1)  # current frame number
                ret, frame = vidcap.read()
                if not ret or len(stacked_frames) > (time_step - 1):
                    break
                frame = frame / 255.0
                if rand_folder == 'running' or rand_folder == 'walking' or rand_folder == 'jogging':
                    if frame_id % 1 == 0 and frame_id > 5:
                        resized_frame = cv2.resize(frame, size=[height, width], interpolation='INTER_NEAREST')
                elif n_frames < 350:
                    if frame_id % 1 == 0 and frame_id > 5:
                        resized_frame = cv2.resize(frame, size=[height, width], interpolation='INTER_NEAREST')
                else:
                    if frame_id % 1 == 0 and frame_id > 10:
                        resized_frame = cv2.resize(frame, size=[height, width], interpolation='INTER_NEAREST')

                stacked_frames.append(resized_frame)

                if len(stacked_frames) < time_step:
                    continue

        stacked_frames = np.reshape(stacked_frames, newshape=(time_step, height, width, 3))
        stacked_frames = np.transpose(stacked_frames, (1, 0, 2, 3))
        stacked_frames = np.reshape(stacked_frames, newshape=(height, time_step * width, 3))
        return stacked_frames

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            #torch.manual_seed(index)
        return torch.from_numpy(self.get_sequence())

    def __len__(self):
        return len(self.dirs)*36*5 # arbitrary
