import string

import numpy as np
from bert_serving.client import BertClient
from pickle import dump
import pandas as pd
import os

# extract transcriptions for images
def load_transcriptions(datapath, transcriptionpath):
    mapping = dict()
    # process lines
    transcriptions = pd.read_pickle(transcriptionpath)
    for video in os.listdir(datapath):  # listing videos
        # store description
        if video.endswith(".mp4"):
            if video not in mapping:
                mapping[video] = list()
            mapping[video].append(transcriptions[video])
    return mapping


def clean_transcriptions(transcriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in transcriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)



####################Not required for BERT###############################
# # convert the loaded transcriptions into a vocabulary of words
# def to_vocabulary(transcriptions):
#     # build a list of all description strings
#     all_desc = set()
#     for key in transcriptions.keys():
#         [all_desc.update(d.split()) for d in transcriptions[key]]
#     return all_desc
#############################################################################

# save transcriptions to file, one per line
def save_features(text_features, filename):
    dump(text_features, open(filename, 'wb'))
    print("Text Features Saved Successfully")




def extract_text_featuers(transcriptions):
    empty = []
    bc = BertClient()

    video = transcriptions.keys()
    textfeatures = dict.fromkeys(video)
    for key, trans in transcriptions.items():
        for i in range(len(trans)):
            desc = trans[i]
            if len(desc) == 0:
                feat = [np.zeros(1024)]
                empty.append(key)
            else:
                text = [desc]
                feat = bc.encode(text)  # Use BERT server to get the text feature

            # store as string
            textfeatures[key] = feat

    return textfeatures


def preprocess_text(partition):
    video_folder = '/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/data/video_data/{}'.format(partition)
    transcription_path = '/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/meta_data/transcription_{}.pkl'.format(partition)
    # load transcriptions and parse transcriptions
    transcriptions = load_transcriptions(video_folder,transcription_path)
    print('Loaded: %d ' % len(transcriptions))
    # clean transcriptions
    clean_transcriptions(transcriptions)

    text_features = extract_text_featuers(transcriptions)
    print('Extracted Features: %d' % len(text_features))
    # save to file
    save_features(text_features,'/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/features/{}/text_features.pkl'.format(partition))


for partition in ['validate','train','test']:
    preprocess_text(partition)
