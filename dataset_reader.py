import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import  Dataset
class feat_dataset(Dataset):
    def __init__(self,partition):
        self.img_feat = pd.read_pickle('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/features/{}/img_features.pkl'.format(partition))
        self.txt_feat = pd.read_pickle('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/features/{}/text_features.pkl'.format(partition))
        self.audio_feat = pd.read_pickle('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/features/{}/audio_features.pkl'.format(partition))
        self.video_name = list(self.img_feat.keys())
        self.annotations= pd.read_pickle('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/meta_data/annotation_{}.pkl'.format(partition))

        self.index=list(range(0,len(self.video_name)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        index=self.index[item]
        video_name = self.video_name[index]
        text_feat=self.txt_feat[video_name+".mp4"]
        img_feat = self.img_feat[video_name]
        labels = [self.annotations['extraversion'][video_name+".mp4"], self.annotations['neuroticism'][video_name+".mp4"],
                  self.annotations['agreeableness'][video_name+".mp4"], self.annotations['conscientiousness'][video_name+".mp4"],
                  self.annotations['openness'][video_name+".mp4"], self.annotations['interview'][video_name+".mp4"]]
        audio_feat=self.audio_feat[video_name+".mp4"]

        text_feat = torch.from_numpy((np.array(text_feat,dtype=float).T).reshape(1024,))
        audio_feat = torch.from_numpy((np.array(audio_feat,dtype=float).T).reshape(512,))
        img_feat = torch.from_numpy((np.array(img_feat,dtype=float).T).reshape(10,1024))


        return (video_name,torch.from_numpy(np.array(text_feat)),torch.from_numpy(np.array(audio_feat)),torch.from_numpy(np.array(img_feat)),torch.from_numpy(np.array(labels)))










