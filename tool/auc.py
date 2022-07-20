import os
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

def SVM_score(loss, normalize=False):

    score = np.array(loss)
    return score

def Combine(loss, normalize=False):

    rgb_score = np.array(loss[0])
    opt_score = np.array(loss[1])
    if(normalize):
        rgb_score -= rgb_score.min()
        rgb_score /= rgb_score.max() if rgb_score.max() != 0 else 1
        opt_score -= opt_score.min()
        opt_score /= opt_score.max() if opt_score.max() != 0 else 1
    else:
        rgb_score /= rgb_score.max()
        opt_score /= opt_score.max()
    
    score = rgb_score + opt_score 
    score /= 2
    return score

def RGB(loss, normalize=False):

    rgb_score = np.array(loss[0])
    if(normalize):
        rgb_score -= rgb_score.min()
        rgb_score /= rgb_score.max() if rgb_score.max() != 0 else 1
    else:
        rgb_score /= rgb_score.max()
    
    score = rgb_score
    return score

def Optical(loss, normalize=False):

    opt_score = np.array(loss[1])
    if(normalize):
        opt_score -= opt_score.min()
        opt_score /= opt_score.max() if opt_score.max() != 0 else 1
    else:
        opt_score /= opt_score.max()

    score = opt_score
    return score

def ORIGIN(loss, normalize=False):
    origin = np.array(loss)
    if(normalize):
        origin -= origin.min()
        origin /= origin.max() if origin.max() != 0 else 1

    score = origin
    return score

get_score_type = \
    {
        "svm":      SVM_score,
        "combine":  Combine,
        "rgb":      RGB,
        "optical":  Optical,
        "ORIGIN":   ORIGIN
    }

def ComputeShanghaitechAUC(loss, combine_type):
    assert (type(loss) == dict),\
        "loss_file type is error : {}.".format(type(loss))
    label_dir = '/home/wangguodong_2020/datasets/vad/shanghaitech/frame_masks'
    assert (type(combine_type) == list), \
        "combine_type is error: {}.".format(type(combine_type))

    auc = {}
    for video in loss:
        gt_dir = '{}/{}.npy'.format(label_dir, video)
        assert (os.path.exists(gt_dir)),\
            "the video file {} doesn't exits".format(video)
        gt = np.load(gt_dir)

        for t in combine_type:
            try:
                assert (t in get_score_type),\
                    "the score type {} is not defined.".format(t)
                score = get_score_type[t](loss[video])
            
                fpr, tpr, thresholds = metrics.roc_curve(gt, score, pos_label=0)
                auc[t] = metrics.auc(fpr, tpr)
                if(auc[t] == np.nan):
                    auc[t] = 0

                print('the auc of  {} type video {} : {}'.format(t, video, auc[t]))
            except Exception as e:
                print(e)

    return auc
    
def ComputeShanghaitechLossFileAUC(loss_dir, combine_type):
    assert (os.path.exists),\
        "the loss file doesn't exists."
    
    with open(loss_dir, 'rb') as f:
        loss = pickle.load(f)

    label_dir = '/home/wangguodong_2020/datasets/vad/shanghaitech/frame_masks'
    assert (type(combine_type) == list), \
        "combine_type is error: {}.".format(type(combine_type))

    auc = {}
    for t in combine_type:
        try:
            assert (t in get_score_type),\
                "the score type {} is not defined.".format(t)
            labels = np.array([], dtype = np.float32)
            scores = np.array([], dtype = np.float32)
            for video in loss:
                gt_dir = '{}/{}.npy'.format(label_dir, video)
                assert (os.path.exists(gt_dir)),\
                    "the video file {} doesn't exits".format(video)
                gt = np.load(gt_dir)

                score = get_score_type[t](loss[video])
                scores = np.concatenate((scores, score), axis=0)
                labels = np.concatenate((labels, gt), axis=0)
            
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
            auc[t] = metrics.auc(fpr, tpr)
            if(auc[t] == np.nan):
                auc[t] = 0

            print('the auc of the {} type whole video : {}'.format(t, auc[t]))
        except Exception as e:
            print(e)
    
    return auc

if __name__ == '__main__':
    #ComputeShanghaitechLossFileAUC('result.pkl')
    None
