# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

# Modified to train Iris model with early stopping - CL

import os
import numpy as np
import json
import pandas as pd
# Set random seed
np.random.seed(0)

import tensorflow as tf
import argparse


def model(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential()    
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, input_dim=x_train.shape[1]))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))    
    
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    model.fit(x_train, y_train, 
              epochs=300,
              batch_size=32, 
              validation_data=(x_test,y_test),
              callbacks=[early_stopping])
    
    return model


def _load_data(file_path, channel):
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(file_path, file) for file in os.listdir(file_path) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(file_path, channel))
        
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    df = pd.concat(raw_data)  
    
    features = df.iloc[:,1:].values
    label = df.iloc[:,0].values
    return features, label


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_data(args.train,'train')
    eval_data, eval_labels = _load_data(args.test,'test')

    classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
