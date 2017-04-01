
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib.request
#from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import pickle

url = 'http://yaroslavvb.com/upload/notMNIST/'

# def maybe_download(filename, expected_bytes):
#  """Download a file if not present, and make sure it's the right size."""
#  if not os.path.exists(filename):
#    filename, _ = urllib.request.urlretrieve(url + filename, filename)
#  statinfo = os.stat(filename)
#  if statinfo.st_size == expected_bytes:
#    print('Found and verified', filename)
#  else:
#    raise Exception(
#      'Failed to verify' + filename + '. Can you get to it with a browser?')
#  return filename
#
# train_filename = maybe_download('data/notMNIST_large.tar.gz', 247336696)
# test_filename = maybe_download('data/notMNIST_small.tar.gz', 8458043)
#
# def extract(filename):
#  tar = tarfile.open(filename)
#  tar.extractall()
#  tar.close()
#  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
#  data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
#  if len(data_folders) != num_classes:
#    raise Exception(
#      'Expected %d folders, one per class. Found %d instead.' % (
#        num_classes, len(data_folders)))
#  print(data_folders)
#  return data_folders

num_classes = 10

train_folders = ['data/notMNIST_large/A', 'data/notMNIST_large/B', 'data/notMNIST_large/C', 'data/notMNIST_large/D', 'data/notMNIST_large/E', 'data/notMNIST_large/F', 'data/notMNIST_large/G', 'data/notMNIST_large/H', 'data/notMNIST_large/I', 'data/notMNIST_large/J']#extract(train_filename)
test_folders = ['data/notMNIST_small/A', 'data/notMNIST_small/B', 'data/notMNIST_small/C', 'data/notMNIST_small/D', 'data/notMNIST_small/E', 'data/notMNIST_small/F', 'data/notMNIST_small/G', 'data/notMNIST_small/H', 'data/notMNIST_small/I', 'data/notMNIST_small/J']#extract(test_filename)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load(data_folders, min_num_images, max_num_images):
    dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0
    for folder in data_folders:
        print(folder)
        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' % (num_images, max_num_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        label_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    print('Labels:', labels.shape)
    return dataset, labels

def pickle_load(filename):
    with open(filename, 'rb') as f:
        b = pickle.load(f)
    return b

def pickle_save(data,filename):
    with open(filename, "wb+") as f:
        pickle.dump(data,f)
    return True

try:
    train_dataset = pickle_load('data/train_dataset.pkl')
    test_dataset = pickle_load('data/test_dataset.pkl')
    train_labels = pickle_load('data/train_labels.pkl')
    test_labels = pickle_load('data/test_labels.pkl')
    print("Successfully loaded data")
except Exception as e:
    print(e)# coding=utf-8
    train_dataset, train_labels = load(train_folders, 450000, 550000)
    test_dataset, test_labels = load(test_folders, 18000, 20000)
    pickle_save(train_dataset,'data/train_dataset.pkl')
    pickle_save(test_dataset,'data/test_dataset.pkl')
    pickle_save(train_labels,'data/train_labels.pkl')
    pickle_save(test_labels,'data/test_labels.pkl')

np.random.seed(133)
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


train_size = 450000
valid_size = 18000
valid_dataset = train_dataset[:valid_size,:,:]
valid_labels = train_labels[:valid_size]
train_dataset = train_dataset[valid_size:valid_size+train_size,:,:]
train_labels = train_labels[valid_size:valid_size+train_size]
print('Training', train_dataset.shape, train_labels.shape)
print('Validation', valid_dataset.shape, valid_labels.shape)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
pickle_file = os.path.join(dir_path,'data','notMNIST.pkl')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)



def trainModel(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,num):
    logreg = LogisticRegression()
    train_dataset = [list(x.flatten()) for x in train_dataset[:num]]
    train_labels = train_labels[:num]
    test_dataset = [list(x.flatten()) for x in test_dataset]
    valid_dataset = [list(x.flatten()) for x in valid_dataset]

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_dataset,train_labels)
    acc = logreg.score(valid_dataset,valid_labels)
    print("Validation accuracy:",acc)
    acc = logreg.score(test_dataset,test_labels)
    print("Test accuracy:", acc)

trainModel(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,10000)
