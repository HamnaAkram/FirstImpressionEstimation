from os import listdir
from pickle import dump
from keras.applications import DenseNet121 as densenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input
from keras.models import Model
import keras

def build_feature_extractor(IMG_SIZE=224):
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    outputs = feature_extractor(inputs)
    model = keras.Model(inputs, outputs, name="feature_extractor")
    print(model.summary())
    return model

# extract img_features from each photo in the directory
def extract_img_features(directory):
    # load the model
    model = densenet()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract img_features from each photo
    img_features = dict()
    for vid_name in listdir(directory):
        # load an image from file
        print('Working on video:', vid_name)
        frame_paths = directory + '/' + vid_name
        count = 0
        for img in listdir(frame_paths):
            count= count +1
            print('Frame: ', count)
            img_path = frame_paths + '/' + img
            image = load_img(img_path, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get img_features
            feature = model.predict(image, verbose=0)
            # # get image id
            # image_id = name.split('.')[0]
            # store feature
            if vid_name not in img_features:
                img_features[vid_name] = list()

            img_features[vid_name].append(feature)

    return img_features


def save_img_features(img_features, filename):
    dump(img_features, open(filename, 'wb'))
    print("Image features Saved Successfully")

def preprocess_frames(partition):
    # extract img_features from all images
    directory = '/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/data/image_data/{}_data'.format(partition)
    img_features = extract_img_features(directory)
    print('Extracted img_features: %d' % len(img_features))
    # save to file
    save_img_features(img_features,'/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/features/{}/img_features.pkl'.format(partition))



for partition in ['validate','train','test']:
    preprocess_frames(partition)