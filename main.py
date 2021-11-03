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
    ap.add_argument("-d", "--device", default="cpu", type=str, help="Whether to perform task on CPU ('cpu') or GPU ('cuda')")
    ap.add_argument("-t", "--task", default="train", type=str, help="Task to perform. Choose between ['train', 'test']")
    ap.add_argument("-p", "--path", default=None, type=str, help=  "path to test file in csv")
    ap.add_argument("-o", "--output_dir", required = True, type=str, help="Output directory path")
    ap.add_argument("-l", "--load", default=None, type=str, help="Path to directory containing models and meta_data")
    ap.add_argument("-f", "--file", default="", type=str, help="Path to single testing file")
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
        test.get_test_performance(args["path"])


    elif args["task"] == "single_test":
        assert os.path.exists(args["file"]), "No text file provided."
        assert args["load"] is not None, "Please provide a checkpoint to load using --load to check test performance"
        trainer.predict_for_file(args["file"])

    else:
        raise ValueError(f"Unrecognized argument passed to --task: {args['task']}")
