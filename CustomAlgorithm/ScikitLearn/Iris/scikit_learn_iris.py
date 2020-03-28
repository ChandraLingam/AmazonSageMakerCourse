#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os
import pandas as pd
import json

from sklearn import ensemble
from sklearn.externals import joblib


def model(args, x_train, y_train, x_test, y_test):   
    model = ensemble.RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth)
    model.fit(x_train, y_train)
    
    print("Training Accuracy: {:.3f}".format(model.score(x_train,y_train)))
    print("Testing Accuracy: {:.3f}".format(model.score(x_test,y_test)))
    
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

    # Hyperparameters are described here.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == '__main__':
    
    args, unknown = _parse_args()
    
    train_data, train_labels = _load_data(args.train,'train')
    eval_data, eval_labels = _load_data(args.test,'test')

    classifier = model(args, train_data, train_labels, eval_data, eval_labels)
    
    if args.current_host == args.hosts[0]:
        # Print the coefficients of the trained classifier, and save the coefficients
        joblib.dump(classifier, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    classifier = joblib.load(os.path.join(model_dir, "model.joblib"))
    return classifier
