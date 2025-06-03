from config import *
from model import YOLONetwork
from dataset import VOCDataset
from utils import (inference, val, train_one_epoch, sum_square_error_loss, clean_prediction,
                   extract_boxes, non_max_suppression, fun_iou, read_content
)
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
from PIL import Image
import sys
import argparse



def main(args):
  os.makedirs(args.output_dir, exist_ok =True)
  # set the model path
  model_path = args.output_dir + "/" + args.model_name
  # get the device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Auto-detect GPU
  print("Device:", device)
  cuda_count = torch. cuda. device_count()
  print(f"We have {cuda_count} cuda devices")

  # setting the seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(args.seed)
  print("Seed set succesfully")

  # set the transform
  transform = transforms.Compose([transforms.Resize((INPUT_SHAPE[2], INPUT_SHAPE[3])), transforms.ToTensor()])

  # download the data
  if not os.path.exists(args.annotation_path):
    print("Downloading the VOC dataset....")
    torchvision.datasets.VOCDetection(root=args.dataset_path, download=True)
  print("There is ", len(os.listdir(args.annotation_path)), " Annotations.")
  print("There is ", len(os.listdir(args.images_path)), " Images.")

  torch.cuda.empty_cache()

  # get all the classes
  classes = []
  for annotation_name in os.listdir(args.annotation_path):
    # read the xml content
    bbox, classes_ = read_content(os.path.join(args.annotation_path, annotation_name))
    classes.extend(classes_)
  classes = list(set(classes))
  n_classes =  len(classes)
  print("There is ", n_classes, " classes.")

  # loading all the file paths
  input_files = [os.path.join(args.images_path, f) for f in os.listdir(args.images_path)]
  output_files = []
  for i, f in enumerate(os.listdir(args.images_path)):
    annotation_file = f.replace('.jpg', '.xml')
    output_files.append(os.path.join(args.annotation_path, annotation_file))


  train_inputs, test_inputs, train_output, test_outputs = train_test_split(input_files, output_files, test_size=1-args.train_ratio, 
                                                                           random_state=args.seed)
  print(f"Length of training: {len(train_inputs)}, Length of validation : {len(test_inputs)}")

  train_dataset = VOCDataset(train_inputs, train_output, transform, classes, factor=448/7, new_shape=(INPUT_SHAPE[2], INPUT_SHAPE[3]))
  test_dataset = VOCDataset(test_inputs, test_outputs, transform, classes, factor=448/7, new_shape=(INPUT_SHAPE[2], INPUT_SHAPE[3]))


  train_random_sampler = RandomSampler(train_dataset)
  val_random_sampler = RandomSampler(test_dataset)

  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=2, sampler=train_random_sampler)
  val_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=2, sampler=val_random_sampler)
  # Loading the model
  torch.cuda.empty_cache()
  gpu_devices = ','.join([str(id) for id in range(0, cuda_count)])
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

  # define the model
  network =  YOLONetwork()

  # load the model into all cuda available (in our case 2)
  netowrk = nn.DataParallel(network)
  network.to(device)
  # define the Adam optmizer
  optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum ,weight_decay=args.weight_decay)
  # scheduler
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  # define the loss
  loss_fn = sum_square_error_loss

  torch.cuda.empty_cache()
  losses = {
      "val_loss":[],
      "train_loss":[]
  }
  best_loss = 10000
  epoch_waited = 0
  print(f"Starting training for {args.epochs}...")
  for epoch in range(0, args.epochs):
      torch.cuda.empty_cache()
      print(f"Training for {epoch} epoch")
      avg_loss = train_one_epoch(network, train_dataloader, loss_fn, optimizer, device)
      avg_loss_val = val(network,loss_fn, val_dataloader , device)
      scheduler.step(avg_loss_val)
      # Printing the training and validation loss
      print(f"==Training loss:{avg_loss} Validation loss:{avg_loss_val}===")
      # Append the loss for later ploting
      losses["train_loss"].append([avg_loss, i])
      losses["val_loss"].append([avg_loss_val, i])
      if best_loss > avg_loss_val:
          best_loss = avg_loss_val
          print(f"Saving the best model to {model_path}")
          torch.save(network.state_dict().copy(), model_path)
      else:
          epoch_waited += 1
      # break if no improvement happen
      if epoch_waited >= args.early_stopping:
          print("Breaking... No improvement")
          break

  print("Train finished successfully")
  print("Training weights saved to ", model_path)
  print("Saving the losses metrics...")
  pd.DataFrame(losses).to_csv(f"{args.output_dir}/losses.csv")

  # # load the model
  # network.load_state_dict(torch.load(model_path, weights_only=True))
  # network.eval()


if __name__=="__main__":
  parser = argparse.ArgumentParser(prog="Training YOLOv1 from scratch")
  parser.add_argument("--epochs", help="How many iteration to train the model for",default=3, type=int)
  parser.add_argument("--output_dir", help="Path to save the model", default="models")
  parser.add_argument("--model_name", help="Path to save the model", default="models.pth")
  parser.add_argument("--batch_size", help="Batch size", default=16, type=int)
  parser.add_argument("--annotation_path", help="Batch size", default="/content/dataset/VOCdevkit/VOC2012/Annotations")
  parser.add_argument("--images_path", help="Batch size", default="/content/dataset/VOCdevkit/VOC2012/JPEGImages")
  parser.add_argument("--seed", help="Seed for repreducibility", default=42)
  parser.add_argument("--lr", help="Learining rate", default=3e-4)
  parser.add_argument("--early_stopping", help="Number of epoch to try for if no improvement during the training", default=3, type=int)
  parser.add_argument("--dataset_path", help="Where to save the downloaded dataset", default="/content/dataset")
  parser.add_argument("--train_ratio", help="Training size", default=0.8, type=float)
  parser.add_argument("--momentum", help="Momentum", default=0.9, type=float)
  parser.add_argument("--weight_decay", help="Weight decay during training", default=0.0005, type=float)

  args = parser.parse_args()
  main(args)

