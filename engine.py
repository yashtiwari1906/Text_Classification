import pandas as pd
import tez
import yaml 
import datetime
import os
import joblib
import torch
import torch.nn as nn
import transformers
from model import BERTBaseUncased
from dataset import BERTDataset
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random

def open_config(file):
    ''' Opens a configuration file '''

    config = yaml.safe_load(open(file, 'r'))
    return config

class TrainModel():
    def __init__(self, args):
        #path = "/content/drive/MyDrive/task_data/train_data.csv"
        config = open_config(args["config"])
        self.output_dir = config["output_dir"]
        self.batch_size, self.epochs, self.path = config["batch_size"], config["epochs"], config["path"]
        self.dfx = pd.read_csv(self.path)
        self.device = config["device"] if torch.cuda.is_available() else "cpu"
        print("Running on {}........".format(config["device"]))


    def begin(self, col):
        
        self.dfx[col] = self.dfx[col].apply(lambda x: x.replace(" ", "_"))
        #self.dfx = self.dfx.dropna().reset_index(drop=True)
        lbl_enc = preprocessing.LabelEncoder()
        self.dfx[col] = lbl_enc.fit_transform(self.dfx[col].values)

        self.df_train, self.df_valid = model_selection.train_test_split(
            self.dfx, test_size=0.1, random_state=42,stratify=self.dfx[col].values
        )

        self.df_train = self.df_train.reset_index(drop=True)
        self.df_valid = self.df_valid.reset_index(drop=True)

        meta_data = {
            "lbl_enc_"+str(col): lbl_enc
        }

        joblib.dump(meta_data, os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +str(col)+ ".bin"))

        train_dataset = BERTDataset(
            text=self.df_train.transcription.values, target=self.df_train[col].values
        )
        
        valid_dataset = BERTDataset(
            text=self.df_valid.transcription.values, target=self.df_valid[col].values
        )

        n_train_steps = int(len(self.df_train) / self.batch_size* 10)
        model = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx[col].nunique()
        )

        # model.load("model.bin")
        #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        self.tb_logger = tez.callbacks.TensorBoardLogger(log_dir="/content/logs/")
        self.es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path=os.path.join(self.output_dir, "models", "ModelFor"+str(col)+".bin"))
        
        model.fit(
            train_dataset,
            valid_dataset=valid_dataset,
            train_bs=self.batch_size,
            device=self.device,
            epochs=self.epochs,
            callbacks=[self.tb_logger, self.es],
            fp16=True,
        )
        model.save(os.path.join(self.output_dir, "models", "ModelFor"+str(col)+".bin"))
        
    def calculate_f1_score(self, action_ref, action_pred, object_ref, object_pred, location_ref, location_pred):
        score_action = f1_score(action_ref, action_pred, average = "micro")
        score_object = f1_score(object_ref, object_pred, average = "macro")
        score_location = f1_score(location_ref, location_pred, average = "weighted")
        print("=========================================================")
        print("F1 Score for the Action is {}".format(score_action))
        print("F1 Score for the Object is {}".format(score_object))
        print("F1 Score for the Location is {}".format(score_location))
        print("=========================================================")
    
    def get_test_performance(self):
      
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"action"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_action"]
        #if test_file not None:
        self.dfx = pd.read_csv(self.path)
        self.dfx["action"] = lbl_enc.fit_transform(self.dfx["action"].values)
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"object"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_object"]
        self.dfx["object"] = lbl_enc.fit_transform(self.dfx["object"].values)
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"location"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_location"]
        self.dfx["location"] = lbl_enc.fit_transform(self.dfx["location"].values)
        self.df_valid = self.dfx 
        self.df_valid = self.df_valid.reset_index(drop=True)
        n_train_steps = int(len(self.df_valid) / self.batch_size* 10)
        modelAction = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["action"].nunique()
        )
        modelObject = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["object"].nunique()
        )
        modelLocation = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["location"].nunique()
        )
        modelAction.load(os.path.join(self.output_dir, "models", "ModelForaction.bin"))
        modelObject.load(os.path.join(self.output_dir, "models", "ModelForobject.bin"))
        modelLocation.load(os.path.join(self.output_dir, "models", "ModelForlocation.bin"))
        actions, objects, locations = [], [], []
        f1_action, f1_object, f1_location = [], [], []

        action_dataset = BERTDataset(
            text=self.df_valid.transcription.values, target=self.df_valid["action"].values
        )
        object_dataset = BERTDataset(
            text=self.df_valid.transcription.values, target=self.df_valid["object"].values
        )
        location_dataset = BERTDataset(
            text=self.df_valid.transcription.values, target=self.df_valid["location"].values
        )
        
        preds1 = modelAction.predict(action_dataset, batch_size=self.batch_size, n_jobs=-1)
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"action"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_action"]
        for p in preds1:
            tensor = torch.tensor(p)
            output = torch.argmax(tensor, dim=1).cpu().detach().numpy()
            f1_action.extend(output)
            actions.extend(lbl_enc.inverse_transform(output))
        
        preds2 = modelObject.predict(object_dataset, batch_size=self.batch_size, n_jobs=-1)
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"object"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_object"]
        for p in preds2:
            tensor = torch.tensor(p)
            output = torch.argmax(tensor, dim=1).cpu().detach().numpy()
            f1_object.extend(output)
            objects.extend(lbl_enc.inverse_transform(output))
        preds3 = modelLocation.predict(location_dataset, batch_size=16, n_jobs=-1)
        meta_data = joblib.load(os.path.join(self.output_dir, "meta_data" , "lbl_enc_" +"location"+ ".bin"))
        lbl_enc = meta_data["lbl_enc_location"]
        for p in preds3:
            tensor = torch.tensor(p)
            output = torch.argmax(tensor, dim=1).cpu().detach().numpy()
            f1_location.extend(output)
            locations.extend(lbl_enc.inverse_transform(output))

        texts = list(self.df_valid.transcription.values)
        
        print()
        print("="*150)
        for text, action, object, location in zip(random.sample(texts, 20), random.sample(actions, 20), random.sample(objects, 20), random.sample(locations, 20)):
            print("{}".format(text)+" "*(60-len(text))+"||"+" "*5 + "{}".format(action)+" "*(20-len(action))+"||"+" "*5 + "{}".format(object)+" "*(10-len(object))+"||"+" "*5, "{}".format(location)+" "*(10-len(location))+"||")
            print("-"*150)
        print("="*150)
        #print(len(actions), len(objects), len(locations), len(texts))
        self.calculate_f1_score(list(self.df_valid.action.values), f1_action, list(self.df_valid.object.values), f1_object, list(self.df_valid.location.values), f1_location)


    def inference_on_single_text(self, text):
        #tried some of the approaches but they didn't worked :(
        modelAction = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["action"].nunique()
        )
        modelObject = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["object"].nunique()
        )
        modelLocation = BERTBaseUncased(
            num_train_steps=n_train_steps, num_classes=self.dfx["location"].nunique()
        )
        modelAction.load(os.path.join(self.output_dir, "models", "ModelForaction.bin"))
        modelObject.load(os.path.join(self.output_dir, "models", "ModelForobject.bin"))
        modelLocation.load(os.path.join(self.output_dir, "models", "ModelForlocation.bin"))
