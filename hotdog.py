import sys, os
import matplotlib.pyplot as plt
import cv2
import numpy as np

import mxnet as mx

# Download model if it doesn't already exist
filelist = ['resnet-152-symbol.json', 'resnet-152-0000.params', 'synset.txt']
if not all([os.path.isfile(f) for f in filelist]):
    path='http://data.mxnet.io/models/imagenet-11k/'
    [mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json'),
     mx.test_utils.download(path+'resnet-152/resnet-152-0000.params'),
     mx.test_utils.download(path+'synset.txt')]

sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0) 
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        plt.imshow(img)
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    os.remove(fname) #Optional: remove the downloaded file
    return img

def read_image(fname, show=False):
    # only show the image
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        plt.imshow(img)
        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    # os.remove(fname) #Optional: remove the downloaded file
    return img

def predict(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

#test files:        
#get_hotdog('http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
#https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Hot_dog_with_mustard.png/1200px-Hot_dog_with_mustard.png
def get_hotdog(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    if (a[0] == 7460 or a[0] == 7424):
        print('Hotdog!')
    else:
        print('Not hotdog!')
    
if (len(sys.argv) == 2 and sys.argv[1].startswith('http')):
    get_hotdog(sys.argv[1])

def read_hotdog(filename):
    img = read_image(filename, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    if (a[0] == 7460 or a[0] == 7424):
        return True
    else:
        return False