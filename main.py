#!/usr/bin/env python3
#import DataProcessingPipeline as dpp
from DataProcessingPipeline.DataExtraction import DataExtraction
from DataProcessingPipeline.GaitProcessing import GaitProcessing
from DataProcessingPipeline.FeatureExtraction_old import FeatureExtraction
import pandas as pd
from tabulate import tabulate
import os, sys
#sys.path.append('/project/yuece/SourceLoc/bin/python/')
sys.path.append('/home/isratjahantulin/Downloads/SPINE_Code')

if __name__ == '__main__':
    #enode_data_dir = "/data/project/spine/enode_data/"
    enode_data_dir = "/home/isratjahantulin/Downloads/all_data/all_data/Enode/"  ## ISRAT's Location
    #sp_data_dir = "/data/project/spine/smartphone_data/"
    sp_data_dir = "/home/isratjahantulin/Downloads/all_data/all_data/Smartphone/"   ## ISRAT's Location
    data_dir = {"enode": enode_data_dir, "phone": sp_data_dir}

    # get test and task info from app
    #test = "gait"
    # task = "2min"

    #test = "sts"
    #task = "5rep"

    #test = "tug"
    #task = "single"
    #task = "dual"

    test = "gait"
    task = "2min"

    completed = 0
    valid = 0
    fs = 100

    # all Features
    all_feat = []
    all_feat_list = []

    while not completed:
        # Read, preprocess data
        data = DataExtraction(data_dir).run()

        # Event Detection for each test/task
        gp = GaitProcessing(data)
        acc, gyr, events, unproc_acc, valid = gp.run(test)

        # validity: server -> app
        # Feature Extraction for each test/task
        fe = FeatureExtraction(acc, gyr, events, fs, unproc_acc)
        feat, feat_list = fe.run(test, task)

        # Store extracted features
        all_feat.append(feat)
        all_feat_list.append(feat_list)

        # Set completed = 1 to stop the loop
        completed = 1

    data_df = pd.DataFrame(data=all_feat, columns=all_feat_list)

    # Set display options for better table visibility
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Increase the display width
    pd.set_option('display.colheader_justify', 'center')  # Center column headers
    pd.set_option('display.float_format', '{:.2f}'.format)  # Format float values

    print("Data Frame:")
    print(tabulate(data_df, headers='keys', tablefmt='psql'))

    # combine entire dataset
    # fed the model with entire dataset
    # compute feature importance
    # reasoning based on important features
    # send list of exercises to app

