from torch.utils.data import  Dataset
from utils import read_content, read_image
import torch
import numpy as np
class VOCDataset(Dataset):
    def __init__(self, input_files, output_files,transform, classes, factor, new_shape=(448, 448)):
      self.input_files = input_files
      self.output_files = output_files
      self.transform = transform
      self.classes = classes
      self.factor = factor
      self.new_shape = new_shape

    def __getitem__(self, idx):
      # check if it is exists
      if idx >= len(self.input_files):
          raise IndexError("File doasn't exists")
      # load the annotation and the image path
      bboxs, _ = read_content(self.output_files[idx])
      # load the image
      img = read_image(self.input_files[idx])

      # transform it
      transformed_image = self.transform(img)
      # rescale the bounding boxes to the new shape
      w_original, h_original = img.size
      # (3, 448, 448) -> (7, 7, 30)
      output = torch.zeros((7, 7, 30))
      for bbox in bboxs:
        # load the bbox
        x_min = bbox[0] * self.new_shape[0] / w_original # top-left coorner
        y_min = bbox[1] * self.new_shape[1] / h_original # top-left coorner
        x_max = bbox[2] * self.new_shape[0] / w_original # bottom-right
        y_max = bbox[3] * self.new_shape[1] / h_original # bottom-right

        # transform to yolo format
        x_center = (x_min + x_max )/2
        y_center = (y_min + y_max )/2
        w = x_max  - x_min
        h = y_max  - y_min
        # labeling based on the coordinate
        for cell_i in range(7):
          mini, maxi = cell_i * self.factor, (cell_i + 1) * self.factor
          for cell_j in range(7):
            minj, maxj = cell_j * self.factor, (cell_j + 1) * self.factor
            if minj <= x_center < maxj and mini <= y_center < maxi:
              # fil with normalized values
              x_center = x_center / self.new_shape[0]
              y_center = y_center / self.new_shape[1]
              w = w / self.new_shape[0]
              h = h / self.new_shape[1]
              output[cell_i, cell_j, :5] = torch.tensor([x_center, y_center, w, h, 1]) # fill the correct bounding box
              output[cell_i, cell_j, 10+self.classes.index(bbox[-1])] = torch.tensor(1) # fill the correct class

      return transformed_image, output

    def __len__(self):
        return len(self.input_files)
