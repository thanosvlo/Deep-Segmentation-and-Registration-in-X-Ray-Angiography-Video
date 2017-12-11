# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

from tensorflow.contrib import predictor

from dltk.core import metrics as metrics

from dltk.utils import sliding_window_segmentation_inference
import cv2

from deploy_reader import read_fn

READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = 1
dices=[]

def predict(args):
    # Read in the csv with the file names you would want to predict on
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    # file_names = file_names[-N_VALIDATION_SUBJECTS:]

    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    export_dir = [os.path.join(args.model_path, o) for o in os.listdir(args.model_path)
                  if os.path.isdir(os.path.join(args.model_path, o)) and
                  o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Fetch the output probability op of the trained network
    y_prob = my_predictor._fetch_tensors['y_prob']
    num_classes = y_prob.get_shape().as_list()[-1]

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    k=0
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)
        lbl = np.expand_dims(output['labels']['y'], axis=0)

        # Do a sliding window inference with our DLTK wrapper
        pred = sliding_window_segmentation_inference(
            session=my_predictor.session,
            ops_list=[y_prob],
            sample_dict={my_predictor._feed_tensors['x']: img},
            batch_size=32)[0]

        # Calculate the prediction from the probabilities
        pred = np.argmax(pred, -1)

        # Calculate the Dice coefficient
        dsc = metrics.dice(pred, lbl, num_classes)[1:].mean()

        # Save the file as .nii.gz using the header information from the
        # original sitk image
        output_fn = os.path.join('./Inference','{}_mask.nii.gz'.format(k))
        k+=1

        new_sitk = sitk.GetImageFromArray(pred[0].astype(np.int32))
        # temp=pred.astype(np.int32)
        # temp=temp.T
        
        temp=sitk.GetArrayFromImage(output['sitk']).astype(np.float32)
        temp=temp.T
        temp=temp[0,:,:]
        temp=temp[np.newaxis,...]
        print(temp.shape)
        temp = sitk.GetImageFromArray(temp.astype(np.int32))
        new_sitk.CopyInformation(temp)
        sitk.WriteImage(new_sitk, output_fn)
        # cv2.imwrite(output_fn,temp)

        # Print outputs
        dices.append(dsc)
        print('Id={}; Dice={:0.4f}; time={:0.2} secs; output_path={};'.format(
            k, dsc, time.time() - t0, output_fn))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MRBrainS13 example segmentation deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default='./outputs')
    parser.add_argument('--csv', default='/home/av2514/DLTK-master/examples/applications/MRBrainS13_tissue_segmentation/test_2.csv')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Call training
    predict(args)
    dices=np.mean(dices)
    print('Average Dice Distance: ',dices)
