import os
from PIL import Image
import time
import argparse
import torch
import pickle
import numpy as np
from torchvision.ops import roi_align
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoAnomalyDataset(Dataset):
    """Video Anomaly Dataset."""
    def __init__(self,
                 data_dir=None, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 filter_ratio=0.9,
                 frame_num=7):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        assert dataset in ['shanghaitech', 'ped2', 'avenue'], 'dataset missing'
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.filter_ratio = filter_ratio
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'odd number is preferred'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

         # speed up image loading in the case of multiple objects within the same frame
        self.cache_clip = None 
        self.cache_video = None
        self.cache_frame = None

        if 'train' in data_dir:
            self.test_stage = False
        elif 'test' in data_dir:
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'

        # only use 20% samples for shanghaitech
        self.sample_step = 1 if self.test_stage  else 5
        if self.dataset != 'shanghaitech':
            self.sample_step = 1
        
        with open(detect_dir, 'rb') as f:
            self.detect = pickle.load(f)

        self.objects_list = []
        self._load_data(file_list)
        self.save_objects()
    
    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(l)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                detect_result = self.detect[video_file][frame]
                detect_result = detect_result[detect_result[:, 4] > self.filter_ratio, :]
                object_num = detect_result.shape[0]
                for i in range(object_num):
                    self.objects_list.append({"video_name":video_file, "frame":frame, "object": i})
        print("Load {} videos {} frames, {} objects, in {} s.".format(self.videos, total_frames, len(self.objects_list), time.time() - t0))

    def save_objects(self):
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)
        static_obj = []
        for i in tqdm(range(len(self.objects_list))):
            record = self.objects_list[i]
            obj = self.get_object(record["video_name"], record["frame"], record["object"])
            video_dir = os.path.join(self.dataset, self.phase, record["video_name"])
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            obj = obj.numpy()
            np.save(os.path.join(video_dir, str(record['frame']) + '_' + str(record['object']) + '.npy'), obj)
        

    def __len__(self):
        return len(self.objects_list)

    def __video_list__(self):
        return self.videos_list

    def get_object(self, video_name, frame, obj_id):
        detect_result = self.detect[video_name][frame]
        detect_result = detect_result[detect_result[:, 4] > self.filter_ratio, :]
        frame = frame - self.half_frame_num
        if video_name == self.cache_video and self.cache_frame == frame:
            img = self.cache_clip
        else:
            img = self.get_frame(video_name, frame)
            self.cache_frame = frame
            self.cache_video = video_name
            self.cache_clip = img
        obj = self.crop_object(img, detect_result, obj_id)
        return obj

    def get_frame(self, video_name, frame):
        video_dir = self.data_dir + '/' + video_name + '/'
        frame_list = os.listdir(video_dir)
        img = self.read_frame_data(video_dir, frame, frame_list)
        return img

    def read_single_frame(self, video_dir, frame, frame_list):
        transform = transforms.ToTensor()

        img = None
        if self.dataset == 'ped2':
            frame_ = "{:03d}.jpg".format(frame)
        elif self.dataset == 'avenue':
            frame_ = "{:04d}.jpg".format(frame)
        else:
            if(self.test_stage):
                frame_ = "{:03d}.jpg".format(frame)
            else:
                frame_ = "{:06d}.jpg".format(frame)
        assert (frame_ in frame_list),\
            "The frame {} is out of the range:{}.".format(int(frame_), len(frame_list))

        jpg_dir = '{}/{}'.format(video_dir, frame_)
        assert os.path.exists(jpg_dir), "{} isn\'t exists.".format(jpg_dir)

        img = Image.open(jpg_dir)
        img = transform(img).unsqueeze(dim=0) 
        img = img.permute([1, 0, 2, 3])
        return img


    def read_frame_data(self, video_dir, frame, frame_list):
        img = None
        for f in range(self.frame_num):
            _img = self.read_single_frame(video_dir, frame + f, frame_list)
            if f == 0:
                img = _img
            else:
                img = torch.cat((img, _img), dim=1)
        return img

    
    # TO-DO: compare the RoiAlign with Crop+Resize
    def crop_object(self, frame_img, bbox, i, size=(64, 64)):
        # frame_img : C * D * H * W
        shape = frame_img.shape
        bbox = torch.from_numpy(bbox[i, :4]).float()
        frame_img = frame_img.reshape(1, -1, shape[2], shape[3])
        frame_img = roi_align(frame_img, [bbox.unsqueeze(dim=0)], output_size=size)
        frame_img = frame_img.reshape(-1, shape[1], size[0], size[1])
        return frame_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="patch generation")
    parser.add_argument("--dataset", type=str, default='shanghaitech')
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--filter_ratio", type=float, default=0.8)
    parser.add_argument("--sample_num", type=int, default=9)

    args = parser.parse_args()
    data_dir = "/irip/wangguodong_2020/projects/datasets/vad/"  # directory for raw frames
    shanghai_dataset = VideoAnomalyDataset(data_dir=data_dir + args.dataset + '/' + args.phase + 'ing/', 
                                           detect_dir='detect/' + args.dataset + '_' + args.phase + '_detect_result_yolov3.pkl',
                                           dataset=args.dataset,
                                           filter_ratio=args.filter_ratio, 
                                           frame_num=args.sample_num)
