import pandas as pd
import tez
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import time 
from model import BERTBaseUncased
from dataset import BERTDataset
from engine import TrainModel
import argparse 
from datetime import datetime as dt 

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to configuration file")
    ap.add_argument("-t", "--task", default="train", type=str, help="Task to perform. Choose between ['train', 'test']")

    args = vars(ap.parse_args())
    
    if args["task"] == "train":
        train = TrainModel(args)
        train.begin("action")
        print("model for action is trained, Now proceeding for others.....")
        train.begin("object")
        print("model for object is trained, Now proceeding for last one.....")
        train.begin("location")
        print("Training Completed!!!")
    
    elif args["task"] == "test":
        assert args["output_dir"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        test = TrainModel(args)
        test.get_test_performance(args)


    elif args["task"] == "single_test":
        #Due to lack of time this was not completed
        print("In Progress Don't worry it is easy.......:)")

    else:
        raise ValueError(f"Unrecognized argument passed to --task: {args['task']}")
