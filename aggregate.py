import pickle
import os
import numpy as np
from numpy.core.fromnumeric import nonzero
from scipy.ndimage.measurements import label
from tool import evaluate
import argparse
from scipy.ndimage import convolve
import torch.nn.functional as F
import torch
import math


def video_label_length(dataset='shanghaitech'):
    if dataset == 'shanghaitech':
        label_path = "/irip/wangguodong_2020/projects/datasets/vad/shanghaitech/frame_masks/"
        video_length = {}
        files = sorted(os.listdir(label_path))
        length = 0
        for f in files:
            label = np.load("{}/{}".format(label_path, f))
            video_length[f.split(".")[0]] = label.shape[0]
            length += label.shape[0]
    elif dataset in ['ped1', 'ped2', 'avenue']:
        test_frame_path = '/irip/wangguodong_2020/projects/datasets/vad/' + dataset + '/testing/'
        files = sorted(os.listdir(test_frame_path))
        video_length = {}
        for f in files:
            video_length[f] = len(os.listdir(os.path.join(test_frame_path, f)))
    return video_length


def score_smoothing(score, ws=43, function='mean', sigma=10):
    assert ws % 2 == 1, 'window size must be odd'
    assert function in ['mean', 'gaussian'], 'wrong type of window function'

    r = ws // 2
    weight = np.ones(ws)
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    return new_score


def load_objects(dataset, frame_num=7):
    root = '/irip/wangguodong_2020/projects/datasets/vad'
    data_dir = os.path.join(root, dataset, 'testing') 

    file_list = sorted(os.listdir(data_dir))

    detect_dir = f'detect/{dataset}_test_detect_result_yolov3.pkl'
    with open(detect_dir, 'rb') as f:
        detect = pickle.load(f)

    if dataset == 'ped2':
        filter_ratio = 0.5
    elif dataset == 'avenue':
        filter_ratio = 0.8

    objects_list = []
    videos_list = []

    total_frames = 0
    contain = 0
    total_small_ = 0
    videos = 0
    start_ind = frame_num // 2

    for video_file in file_list:
        if video_file not in videos_list:
            videos_list.append(video_file)
        l = os.listdir(data_dir + '/' + video_file)
        videos += 1
        length = len(l)
        total_frames += length
        for frame in range(start_ind, length - start_ind):
            if detect is not None:
                detect_result = detect[video_file][frame]
                detect_result = detect_result[detect_result[:, 4] > filter_ratio, :]
                object_num = detect_result.shape[0]
            else:
                object_num = 1

            flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(object_num, 0)
            is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
            is_contain = is_contain.any(-1)
            is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10
            for i in range(object_num):
                if not is_contain[i]:
                    if not is_small[i]:
                        objects_list.append({"video_name":video_file, "frame":frame, "object": i, "loc": detect_result[i, :4]})
                    else:
                        total_small_ += 1
                else:
                    contain += 1

    print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects."\
        .format(videos, total_frames, len(objects_list), contain, total_small_))
    return objects_list


def remake_video_3d_output(video_output, dataset='ped2', frame_num=7):
    object_list = load_objects(dataset, frame_num=frame_num)

    video_length = video_label_length(dataset=dataset)
    

    return_output_spatial = []
    return_output_temporal = []
    return_output_complete = []

    if dataset == 'ped2':
        video_height = 240
        video_width = 360
        block_scale = 1
        dim = 5
    elif dataset == 'avenue':
        video_height = 360
        video_width = 640
        block_scale = 20
        dim = 5


    video_l = sorted(list(video_output.keys()))
    cnt = 0
    for i in range(len(video_l)):
        video = video_l[i]
        frame_record = video_output[video]
        frame_l = sorted(list(frame_record.keys()))

        block_h = int(round(video_height / block_scale))
        block_w = int(round(video_width / block_scale))
        video_ = np.ones((block_h, block_w, video_length[video]))
        video2_ = np.ones((block_h, block_w, video_length[video]))

        local_max_ = 0
        local_max2_ = 0
        local_min_ = 1
        local_min2_ = 1
        for fno in frame_l:
            object_record = frame_record[fno]
            for score_, score2_ in object_record:
                loc_V3 = object_list[cnt]['loc']
                loc_V3 = (np.round(loc_V3 / block_scale)).astype(np.int32)

                video_[loc_V3[1]: loc_V3[3] + 1, loc_V3[0]: loc_V3[2] + 1, fno] = \
                    np.minimum(
                        video_[loc_V3[1]: loc_V3[3] + 1, loc_V3[0]: loc_V3[2] + 1,fno], 
                        score_)
                video2_[loc_V3[1]: loc_V3[3] + 1, loc_V3[0]: loc_V3[2] + 1, fno] = \
                    np.minimum(
                        video2_[loc_V3[1]: loc_V3[3] + 1, loc_V3[0]: loc_V3[2] + 1,fno], 
                        score2_)

                local_max_ = max(score_, local_max_)
                local_min_ = min(score_, local_min_)
                local_max2_ = max(score2_, local_max2_)
                local_min2_ = min(score2_, local_min2_)

                cnt += 1

        # spatial
        video_ = (video_ - local_min_) / (local_max_ - local_min_)

        # temporal
        video2_ = (video2_ - local_min2_) / (local_max2_ - local_min2_)

        score = np.stack((video_, video2_))
        score = torch.from_numpy(score).unsqueeze(1)
        score = score.permute((0, 1, 4, 2, 3)).float().cuda()
        # padding 
        p3d = (dim // 2, dim // 2, dim // 2, dim // 2, dim // 2, dim // 2)
        score_padding = F.pad(score, p3d, mode='constant', value=1)
        # 3d mean filter
        score_3d = F.avg_pool3d(score_padding, kernel_size=dim, stride=1, padding=0).cpu().numpy()
        score_3d = score_3d.transpose(0, 1, 3, 4, 2).squeeze()

        video_ = score_3d[0]
        video2_ = score_3d[1]
        

        frame_scores = np.ones(video_length[video])
        frame_scores2 = np.ones(video_length[video])
        frame_scores3 = np.ones(video_length[video])
        for i in range(video_length[video]):
            frame_scores[i] =  0.5 * video_[:, :, i].min() + 0.5 * video2_[:, :, i].min()
            frame_scores2[i] = video_[:, :, i].min()
            frame_scores3[i] = video2_[:, :, i].min() 
        
        frame_scores -= frame_scores.min()
        frame_scores /= frame_scores.max()

        frame_scores2 -= frame_scores2.min()
        frame_scores2 /= frame_scores2.max()

        frame_scores3 -= frame_scores3.min()
        frame_scores3 /= frame_scores3.max()
        
        return_output_complete.append(frame_scores)
        return_output_spatial.append(frame_scores2)
        return_output_temporal.append(frame_scores3)

    return return_output_spatial, return_output_temporal, return_output_complete

def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter  


def remake_video_output(video_output, dataset='ped2'):
    video_length = video_label_length(dataset=dataset)
    return_output_spatial = []
    return_output_temporal = []
    return_output_complete = []
    video_l = sorted(list(video_output.keys()))
    for i in range(len(video_l)):
        video = video_l[i]
        frame_record = video_output[video]
        frame_l = sorted(list(frame_record.keys()))
        video_ = np.ones(video_length[video])
        video2_ = np.ones(video_length[video])

        local_max_ = 0
        local_max2_ = 0
        for fno in frame_l:
            object_record = frame_record[fno]
            object_record = np.array(object_record)
            video_[fno], video2_[fno] = object_record.min(0)

            local_max_ = max(object_record[:, 0].max(), local_max_)
            local_max2_ = max(object_record[:, 1].max(), local_max2_)
        
        # spatial
        non_ones = (video_ != 1).nonzero()[0]
        local_max_ = max(video_[non_ones])
        video_[non_ones] = (video_[non_ones] - min(video_)) / (local_max_ - min(video_))

        # temporal
        non_ones = (video2_ != 1).nonzero()[0]
        local_max2_ = max(video2_[non_ones])
        video2_[non_ones] = (video2_[non_ones] - min(video2_)) / (local_max2_ - min(video2_))
            
        video_ = score_smoothing(video_)
        video2_ = score_smoothing(video2_)

        return_output_spatial.append(video_)
        return_output_temporal.append(video2_)

        combined_video = (video2_ + video_) / 2.0
        return_output_complete.append(combined_video)

    return return_output_spatial, return_output_temporal, return_output_complete


def evaluate_auc(video_output, dataset='shanghaitech'):
    result_dict = {'dataset': dataset, 'psnr': video_output}
    smoothed_results, aver_smoothed_result = evaluate.evaluate_all(result_dict, reverse=True, smoothing=True)
    print("(smoothing: True): {}  aver_result: {}".format(smoothed_results, aver_smoothed_result))
    return smoothed_results, aver_smoothed_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Prediction')
    parser.add_argument('--file', default=None, type=str, help='pkl file')
    parser.add_argument('--dataset', default='ped2', type=str)
    parser.add_argument('--frame_num', required=True, type=int)

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        output = pickle.load(f)
    if args.dataset == 'shanghaitech':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(output, dataset=args.dataset)
    else:
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(output, dataset=args.dataset, frame_num=args.frame_num)
    
    evaluate_auc(video_output_spatial, dataset=args.dataset)
    evaluate_auc(video_output_temporal, dataset=args.dataset)
    evaluate_auc(video_output_complete,  dataset=args.dataset)
