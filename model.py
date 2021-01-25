import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection
import glob
import os
import tqdm as tqdm
import datetime
import logging
import json

import tensorflow.keras as keras
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import os
import os.path as pth
import shutil
import time
from tqdm import tqdm

import itertools
from itertools import product, combinations

import numpy as np
from PIL import Image


from multiprocessing import Process, Queue
import datetime

import tensorflow.keras as keras

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, \
                                    Flatten, Conv3D, AveragePooling3D, MaxPooling3D, Dropout, \
                                    Concatenate, GlobalMaxPool3D, GlobalAvgPool3D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler, \
                                        EarlyStopping
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm


tf.get_logger().setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")

BASE_MODEL_NAME = 'MobileNetV2-for-upload'
my_model_base = keras.applications.mobilenet_v2
my_model = my_model_base.MobileNetV2

config = {
    'is_zscore':True,
    
    # 'input_shape': (540, 960, 3),
    'aug': {
        'resize': (270, 480),
    },
    # 'input_shape': (224, 360, 3),
    #'input_shape': (270, 480, 3),
    
    'input_shape': (224, 224, 3),
    'input_size': (224, 224, 3),
    'momentum': 0.9,
    'n_classes': 1049,

    'output_activation': 'softmax',
    'num_class': 1049,
    'output_size': 1049,
    
    'conv':{
        'conv_num': (0), # (3,5,3),
        'base_channel': 0, # 4,
        'kernel_size': 0, # 3,
        'padding':'same',
        'stride':'X'
    },
    'pool':{
        'type':'X',
        'size':'X',
        'stride':'X',
        'padding':'same'
    },
    'fc':{
        'fc_num': 0,
     },
    
    'activation':'relu',
    
    'between_type': 'avg',
    
    'is_batchnorm': True,
    'is_dropout': False,
    'dropout_rate': 0.5,
    
    'batch_size': 64,
    'buffer_size': 256,
    'loss': 'CategoricalCrossentropy',
    
    'num_epoch': 10000,
    'learning_rate': 1e-3,
    
    'random_state': 7777
}

image_feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'randmark_id': tf.io.FixedLenFeature([], tf.int64),
    # 'id': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def map_func(target_record):
    img = target_record['image_raw']
    label = target_record['randmark_id']
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.dtypes.cast(img, tf.float32)
    return img, label

def resize_and_crop_func(image, label):
    result_image = tf.image.resize(image, config['aug']['resize'])
    # result_image = tf.image.random_crop(image, size=config['input_shape'], seed=7777)
    return result_image, label

def image_aug_func(image, label):
    pass
    return image, label

def post_process_func(image, label):
    # result_image = result_image / 255
    result_image = my_model_base.preprocess_input(image)
    onehot_label = tf.one_hot(label, depth=config['num_class'])
    return result_image, onehot_label

conv_comb_list = []
conv_comb_list += [(0,)]

base_channel_list = [0]

fc_list = [0] # 128, 0

# between_type_list = [None, 'avg', 'max']
between_type_list = ['avg']

batch_size_list = [80]

activation_list = ['relu']

# len(conv_comb_list), conv_comb_list

def create_model(config):
    input_layer = Input(shape=config['input_shape'], name='input_layer')
    pret_model = my_model(
        input_tensor=input_layer, include_top=False, weights='imagenet', 
        input_shape=config['input_shape'], pooling=config['between_type'], 
        classes=config['output_size']
    )

    pret_model.trainable = False
    
    x = pret_model.output
    
    if config['between_type'] == None:
        x = Flatten(name='flatten_layer')(x)
        
    if config['is_dropout']:
        x = Dropout(config['dropout_rate'], name='output_dropout')(x)    
            
    x = Dense(config['output_size'], activation=config['output_activation'], 
          name='output_fc')(x)
#     x = Activation(activation=config['output_activation'], name='output_activation')(x)
    model = Model(inputs=input_layer, outputs=x, name='{}'.format(BASE_MODEL_NAME))

    return model



def read_image(image_path):
    image = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image, channels=3)


def preprocess_input(image, target_size, augment=False):
    image = tf.image.resize(
        image, target_size, method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image


def pred_image(image_path):
    model = create_model(config)
    model.compile(loss=config['loss'], optimizer=Adam(lr=config['learning_rate']),
                  metrics=['acc', 'Precision', 'Recall', 'AUC'])

    checkpoint_path = "./landmark-app/training_1/cp.ckpt/<hdf5 file>"
    print(model.load_weights(checkpoint_path))

    image = read_image(image_path)
    image = preprocess_input(image, config['input_size'][:2])
    image = np.reshape(image, (1, 224, 224, 3))

    print('start pred')
    pred = model.predict(image)
    print(pred)
    probs_argsort = tf.argsort(pred, direction='DESCENDING', stable=True)
    probs = pred[0][probs_argsort][:5]
    print(probs)

    category = pd.read_csv('landmark-app\category.csv')
    mapping = 'landmark-app\mapping.json'
    with open(mapping) as f:
        json_data = json.load(f)

    probs = []
    classes = []
    for i in range(5):
        probs.append(pred[0][[probs_argsort[0][i]]])
        idx = probs_argsort[0][i]
        t = int(idx)
        classes.append(category['landmark_name'][t])
    print('c',classes, 'p', probs)

    ncreds = {
    "client_id": "<naver client id 입력>",      
    "client_secret" : "<naver client secret 입력>"
    }
    nheaders = {
    "X-Naver-Client-Id" : ncreds.get('client_id'),
    "X-Naver-Client-Secret" : ncreds.get('client_secret')
    }

    import urllib
    # urllib.parse.quote(query) URL에서 검색어를 인코딩하기 위한 라이브러리

    # 네이버 지역 검색 주소
    naver_local_url = "https://openapi.naver.com/v1/search/local.json?"

    # 검색에 사용될 파라미터
    # 정렬 sort : 리뷰순(comment)
    # 검색어 query : 인코딩된 문자열
    params_format = "sort=comment&query="

    # 위치는 사용자가 사용할 지역으로 변경가능
    location = classes[0]

    # 추천된 맛집을 담을 리스트
    recommands = []
    # 검색어 지정
    query = location + " 맛집"
    # 지역검색 요청 파라메터 설정
    params = "sort=comment" \
            + "&query=" + query \
            + "&display=" + '5'
    import requests

    # 검색
    # headers : 네이버 인증 정보
    res = requests.get(naver_local_url + params, headers=nheaders)
    
    # 맛집 검색 결과
    result_list = res.json().get('items')
    
    # 경우 1,2 처리
    # 해당 음식 검색 결과에서 가장 상위를 가져옴
    for i in range(0,3):
            recommands.append(result_list[i])

    kcreds = {
    "access_token" : "<access_token입력해주세요>"
    }
    kheaders = {
    "Authorization": "Bearer " + kcreds.get('access_token')
    }

    url = "https://kauth.kakao.com/oauth/token"

    data = {
    "grant_type" : "authorization_code",
    "client_id" : "<client id입력해주세요>",
    "redirect_uri" : "https://localhost.com",
    "code"         : "<code입력>"
    
    }
    response = requests.post(url, data=data)

    tokens = response.json()

    print('t',tokens)


    kakaotalk_template_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    info_url = "https://namu.wiki/w/"+classes[0]

    # 리스트 템플릿 형식 만들기
    contents = []
    template = {
        "object_type" : "list",
        "header_title" : classes[0] + " 주변 맛집 추천",
        "header_link" : {
        "web_url": info_url,
        "mobile_web_url" : info_url
        },
        "contents" : contents,
        "buttons" : [
            {
                "title" : "주변 정보 상세보기",
                "link" : {
                    "web_url": info_url,
                    "mobile_web_url" : info_url
                }
            }
        ],
    }
    print(recommands)
    # contents 만들기
    for place in recommands:
        title = place.get('title')  # 장소 이름
    
        # html 태그 제거
        title = title.replace('<b>','').replace('</b>','')
    
        category = place.get('category')  # 장소 카테고리
        telephone = place.get('telephone')  # 장소 전화번호
        address = place.get('address')  # 장소 지번 주소

        # 각 장소를 클릭할 때 네이버 검색으로 연결해주기 위해 작성된 코드
        enc_address = urllib.parse.quote(address + ' ' + title)
        query = "query=" + enc_address

        # 장소 카테고리가 카페이면 카페 이미지
        # 이외에는 음식 이미지
        if '카페' in category:
            image_url = "https://freesvg.org/img/pitr_Coffee_cup_icon.png"
        else:
            image_url = "https://freesvg.org/img/bentolunch.png?w=150&h=150&fit=fill"

        # 전화번호가 있다면 제목과 함께 넣어줍니다.
        if telephone:
            title = title + "\ntel) " + telephone

        # 카카오톡 리스트 템플릿 형식에 맞춰줍니다.
        content = {
            "title": "[" + category + "] " + title,
            "description": ' '.join(address.split()[1:]),
            "image_url": image_url,
            "image_width": 50, "image_height": 50,
            "link": {
                "web_url": "https://search.naver.com/search.naver?" + query,
                "mobile_web_url": "https://search.naver.com/search.naver?" + query
            }
        }
    
        contents.append(content)

    # JSON 형식 -> 문자열 변환
    payload = {
        "template_object" : json.dumps(template)
    }

    # 카카오톡 보내기
    res = requests.post(kakaotalk_template_url, data=payload, headers=kheaders)

    if res.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(res.json()))



    return classes, probs