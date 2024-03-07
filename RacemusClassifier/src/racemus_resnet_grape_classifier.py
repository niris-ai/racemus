import os
import torch
import random
import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from PIL import Image



# Detects if we have a GPU available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Name of the directory of the dataset
DATASET = "dataset_training"
# Name of the directory in which to save/load models.
MODELS_DIR = "models"
# Name of the directory in which to save/load statistics files.
STATS_DIR = "stats"
# Name of the directory in which are stored the images to predict.
PRED_DIR = "dataset_prediction"
# Name of the directory in which to save the model predictions.
OUTPUT_DIR = "outputs"
# Name for the model
MODEL_NAME = "resnet50_prova"



# Number of classes in the dataset
NUM_CLASSES = 2
# Percentage of examples for training
TRAIN_PERC = 0.6
# Percentage of examples for testing
TEST_PERC = 0.5
# Batch size for training (depends on available memory)
BATCH_SIZE = 32
# Number of workers for the dataloaders
NUM_WORKERS = 2
# If True, use a mechanism to handle unbalanced dataset
WEIGHTED = True
# If True, use augmentation on train dataset
AUGMENTED = True
# If True, show some examples of the dataset before the start of training/prediction
SHOW_EXAMPLES = False

# Number of epochs to train for
NUM_EPOCHS = 25
# Indicates the consecutive number of epochs with no improvements over the validation set that are waited before early stopping the training.
# (The saved model is the one before these 3 epochs)
EARLY_STOP = 3
# Random seed for dataset splits
RAND_SEED = 99999

# Momentum
MOMENTUM = 0.9
# Weight decay
WEIGHT_DECAY = 0.00001
# Learning rate
LEARNING_RATE = 0.001



class ClassificationMetrics:
  def __init__(self, num_classes, device):
    self.C = torch.zeros(num_classes, num_classes)
    self.C = self.C.to(device)

  def add(self, yp, yt):
    with torch.no_grad():
      self.C+=(yt*self.C.shape[1]+yp).bincount(minlength=self.C.numel()).view(self.C.shape).float()

  def clear(self):
    self.C.zero_()

  def acc_sw(self):
    return self.C.diag().sum().item()/self.C.sum()

  def acc_cw(self):
    return (self.C.diag()/(self.C.sum(1))).mean()

  def confusion_matrix(self):
    return self.C



class ModelResults:
  def __init__(self, model_tag):
    self.model_tag = model_tag
    self.train_acc_sw = []
    self.train_acc_cw = []
    self.val_acc_sw = []
    self.val_acc_cw = []
    self.test_acc_sw = None
    self.test_acc_cw = None
    self.test_precision_score = None
    self.test_recall_score = None
    self.test_f1_score = None
    self.conf_matrix_train = None
    self.conf_matrix_val = None

  def add_acc_sw(self, phase, data):
    if phase == 'train':
      self.train_acc_sw.append(data)
    elif phase == 'val':
      self.val_acc_sw.append(data)
    elif phase == 'test':
      self.test_acc_sw = data

  def add_acc_cw(self, phase, data):
    if phase == 'train':
      self.train_acc_cw.append(data)
    elif phase == 'val':
      self.val_acc_cw.append(data)
    elif phase == 'test':
      self.test_acc_cw = data

  def add_test_scores(self, precision, recall, f1):
    self.test_precision_score = precision
    self.test_recall_score = recall
    self.test_f1_score = f1

  def sava_data_on_csv(self):
    script_dir = os.path.dirname(__file__)
    stats_rel_path = "../data/" + STATS_DIR
    stats_abs_path = os.path.join(script_dir, stats_rel_path)

    dataframe = pd.DataFrame(columns=['EPOCH','TRAIN_ACC_SW','TRAIN_ACC_CW','VAL_ACC_SW','VAL_ACC_CW'])
    for epoch in range(len(self.train_acc_sw)):
      df = pd.DataFrame({'EPOCH': epoch+1, 'TRAIN_ACC_SW': self.train_acc_sw[epoch], 'TRAIN_ACC_CW': self.train_acc_cw[epoch], 'VAL_ACC_SW': self.val_acc_sw[epoch], 'VAL_ACC_CW': self.val_acc_cw[epoch]}, index=[0])
      dataframe = pd.concat([dataframe,df], ignore_index=True, axis=0)
    dataframe.to_csv(stats_abs_path + "/" + self.model_tag + "_train_val_results.csv", index=False)

    dataframe = pd.DataFrame({"TEST_ACC_SW": self.test_acc_sw, "TEST_ACC_CW": self.test_acc_cw, "PRECISION": self.test_precision_score, "RECALL": self.test_recall_score, "F1": self.test_f1_score}, index=[0])
    dataframe.to_csv(stats_abs_path + "/" + self.model_tag + "_test_results.csv", index=False)
    


class CustomDataSet(torch.utils.data.Dataset):
  def __init__(self, main_dir, transform):
    self.main_dir = main_dir
    self.transform = transform
    all_imgs = os.listdir(main_dir)
    self.total_imgs = natsort.natsorted(all_imgs)

  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    tensor_image = self.transform(image)
    return tensor_image



def imshow(inp, title=None):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.figure(figsize = (10,10))
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)



def create_dataset_splits(dataset, train_size=0.8, test_size=0.5, weighted=False, augmented=False, batch_size=32, num_workers=2):
  sampler = None

  torch.random.manual_seed(RAND_SEED)
  random.seed(RAND_SEED)

  print('CREATING DATASET SPLITS...')

  data_transforms = {
    'base': 
      transforms.Compose([
        transforms.Resize(256),      
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    'base+aug':
      transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-15,15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  }

  script_dir = os.path.dirname(__file__)
  dataset_rel_path = "../data/" + dataset
  dataset_abs_path = os.path.join(script_dir, dataset_rel_path)

  if augmented:
    dataset_train = datasets.ImageFolder(dataset_abs_path, data_transforms['base+aug'])
  else:
    dataset_train = datasets.ImageFolder(dataset_abs_path, data_transforms['base'])
  dataset_val = datasets.ImageFolder(dataset_abs_path, data_transforms['base'])
  dataset_test = datasets.ImageFolder(dataset_abs_path, data_transforms['base'])

  class_names = dataset_train.classes
  num_classes = len(class_names)
  val_size = (1 - train_size)*test_size

  print('AVAILABLE CLASSES : ' + str(class_names))
  print('SPLITS : train=' + str(train_size) + ' , val=' + str(round(val_size, 3)) + ' , test=' + str(round(1-train_size-val_size, 3)))

  targets = dataset_train.targets

  X_train, X_rem, y_train, y_rem = train_test_split(np.arange(len(targets)),targets, train_size=train_size, stratify=targets, shuffle=True, random_state=RAND_SEED)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=test_size, stratify=y_rem, shuffle=True, random_state=RAND_SEED)

  dataset_subsets = {
    'train': torch.utils.data.Subset(dataset_train, X_train),
    'val': torch.utils.data.Subset(dataset_val, X_valid),
    'test': torch.utils.data.Subset(dataset_test, X_test),
  }

  if(weighted):
    print('WEIGHTS: COUNTING LABELS...')

    train_classes = [label for _, label in dataset_subsets['train']]
    class_count = Counter(train_classes)

    print('WEIGHTS: WEIGHTED RANDOM SAMPLER WEIGHTS CALCULATION...')

    weight = [1. / c for c in pd.Series(class_count).sort_index().values]
    samples_weight = np.array([weight[t] for t in train_classes])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

  print('GENERATING DATALOADERS...')

  dataloaders = {
    'train': torch.utils.data.DataLoader(dataset_subsets['train'],
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         sampler=sampler),
    'val': torch.utils.data.DataLoader(dataset_subsets['val'],
                                       batch_size=batch_size,
                                       num_workers=num_workers),
    'test': torch.utils.data.DataLoader(dataset_subsets['test'],
                                        batch_size=batch_size,
                                        num_workers=num_workers)
  }

  if SHOW_EXAMPLES:
    inputs, classes = next(iter(dataloaders['train']))
    out = make_grid(inputs)
    imshow(out)
    plt.show()

  print('DATASET SPLITS READY!')

  return dataloaders["train"], dataloaders["val"], dataloaders["test"], class_names, num_classes



def create_pred_dataloader(img_dir):
  data_transforms = {
    'base': 
      transforms.Compose([
        transforms.Resize((224,224)),      
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  }

  script_dir = os.path.dirname(__file__)
  dataset_rel_path = "../data/" + img_dir
  dataset_abs_path = os.path.join(script_dir, dataset_rel_path)

  dataset_test = CustomDataSet(dataset_abs_path, data_transforms['base'])

  dataloader = torch.utils.data.DataLoader(dataset_test,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

  if SHOW_EXAMPLES:
    inputs = next(iter(dataloader))
    out = make_grid(inputs)
    imshow(out)
    plt.show()

  return dataloader, dataset_test.total_imgs



def train_one_epoch(model, loss_func, metric_tracker, dataloader, optimizer, epoch, scaler, device):
  model.train()

  metric_tracker.clear()

  for i,(X,yt) in enumerate(dataloader):

    X,yt = X.to(device), yt.to(device)

    optimizer.zero_grad()

    if (DEVICE == 'cuda:0'):
      with torch.cuda.amp.autocast():
        Y = model(X)
        loss = loss_func(Y, yt)
    else:
      Y = model(X)
      loss = loss_func(Y, yt)

    y = Y.argmax(-1).to(device)

    metric_tracker.add(y, yt)

    if (DEVICE == 'cuda:0'):
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      optimizer.step()

  return loss



def validate(model, metric_tracker, dataloader, device):
  model.eval()

  metric_tracker.clear()

  with torch.no_grad(): 
    for i,(X,yt) in enumerate(dataloader):

      X,yt = X.to(device), yt.to(device)

      Y = model(X)
      y = Y.argmax(-1).to(device)

      metric_tracker.add(y,yt)



def save_model(model_name, model, optimizer, loss, num_epochs, class_names):
  script_dir = os.path.dirname(__file__)
  models_rel_path = "../data/" + MODELS_DIR
  models_abs_path = os.path.join(script_dir, models_rel_path)

  path = models_abs_path + "/model_" + model_name + ".pt"
  torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

  class_names_df = pd.DataFrame(class_names, columns=["NAME"])
  class_names_df.to_csv(models_abs_path + "/model_" + model_name + "_classes.csv",header=False, index = False)



def load_model(model_name):
  script_dir = os.path.dirname(__file__)
  models_rel_path = "../data/" + MODELS_DIR
  models_abs_path = os.path.join(script_dir, models_rel_path)

  class_names_df = pd.read_csv(models_abs_path + "/model_" + model_name + "_classes.csv", names=["NAME"])
  class_names = class_names_df.NAME.to_list()
  num_classes = len(class_names)

  checkpoint = torch.load(models_abs_path + "/model_" + model_name + ".pt")
  model = models.resnet50().to(DEVICE)

  for param in model.parameters():
    param.requires_grad = False
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, num_classes).to(DEVICE)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  class_names_df = pd.read_csv(models_abs_path + "/model_" + model_name + "_classes.csv", names=["NAME"])
  class_names = class_names_df.NAME.to_list()

  return model, class_names



def train(model, model_name, loss_function, trDataLoader, vlDataLoader, optimizer, lr_scheduler, num_epochs, results, num_classes, class_names, device):

  epochs_no_improve = 0
  epochs_stop = EARLY_STOP
  prev_val = 0

  loss_func = loss_function

  metric_tracker = ClassificationMetrics(num_classes, device)

  scaler = torch.cuda.amp.GradScaler()

  for epoch in range(1,num_epochs+1):

    if epochs_no_improve == epochs_stop:
      print("-- EARLY STOP -------------------------\n")
      break

    print("-- EPOCH {}/{} -------------------------\n".format(epoch, num_epochs))

    epoch_loss = train_one_epoch(model, loss_func, metric_tracker, trDataLoader, optimizer, epoch, scaler, device)

    print("\tTRAIN | acc_sw: {:.4f} | acc_cw: {:.4f}".format(
        metric_tracker.acc_sw(), metric_tracker.acc_cw()
    ))

    results.add_acc_sw("train", metric_tracker.acc_sw().item())
    results.add_acc_cw("train", metric_tracker.acc_cw().item())

    validate(model, metric_tracker, vlDataLoader, device)

    print("\tEVAL  | acc_sw: {:.4f} | acc_cw: {:.4f}\n".format(
        metric_tracker.acc_sw(), metric_tracker.acc_cw()
    ))

    results.add_acc_sw("val", metric_tracker.acc_sw().item())
    results.add_acc_cw("val", metric_tracker.acc_cw().item())

    if metric_tracker.acc_sw().item() <= prev_val:
      epochs_no_improve += 1
    else:
      epochs_no_improve = 0
      prev_val = metric_tracker.acc_sw().item()
      save_model(model_name, model, optimizer, epoch_loss, epoch, class_names)

    lr_scheduler.step()

  return epoch_loss



def test(model, dataloader, results, num_classes, device):
  metric_tracker = ClassificationMetrics(num_classes, device)

  validate(model,metric_tracker, dataloader, device)

  y_true = []
  y_pred = []

  with torch.no_grad(): 
    for i,(X,yt) in enumerate(dataloader):
      X,yt = X.to(device), yt.to(device)
      Y = model(X)
      y_true.extend(yt.cpu().data.numpy())
      y_pred.extend(Y.argmax(-1).cpu().data.numpy())
  
  precision = metrics.precision_score(y_true,y_pred)
  recall = metrics.recall_score(y_true,y_pred)
  f1 = metrics.f1_score(y_true,y_pred)

  print("\tTEST  | acc_sw: {:.4f} | acc_cw: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f}\n".format(
      metric_tracker.acc_sw(), metric_tracker.acc_cw(), precision, recall, f1
  ))

  results.add_acc_sw("test", metric_tracker.acc_sw().item())
  results.add_acc_cw("test", metric_tracker.acc_cw().item())
  results.add_test_scores(precision, recall, f1)

  return metric_tracker



def predict(model, model_name, class_names, pred_dir, out_dir):
  dataloader, imgs = create_pred_dataloader(pred_dir)

  y_pred = []

  with torch.no_grad(): 
    for i,X in enumerate(dataloader):
      X = X.to(DEVICE)
      Y = model(X)
      y_pred.extend(Y.argmax(-1).cpu().data.numpy())

  data = []
  counter = 0
  for x in y_pred:
    label = class_names[x]
    data.append([counter, imgs[counter] , label])
    counter += 1

  script_dir = os.path.dirname(__file__)
  out_rel_path = "../data/" + out_dir
  out_abs_path = os.path.join(script_dir, out_rel_path)

  labels_df = pd.DataFrame(data, columns=["#","IMAGE","PREDICTION"])
  labels_df.to_csv(out_abs_path + "/" + model_name + "_output.csv",header=True, index = False)



def train_model(model_name):

  print("DEVICE : " + str(DEVICE))

  train_dataloader, val_dataloader, test_dataloader, class_names, num_classes = create_dataset_splits(DATASET,
    train_size=TRAIN_PERC, test_size=TEST_PERC, weighted=WEIGHTED, augmented=AUGMENTED, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

  results = ModelResults(model_name)

  model = models.resnet50(pretrained=True).to(DEVICE)
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, num_classes).to(DEVICE)

  print("MODEL NAME : " + model_name)
  print()
  print(model)
  print()
  
  for param in model.parameters():
    param.requires_grad = True

  optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
  
  lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LEARNING_RATE/4, max_lr=LEARNING_RATE, mode="exp_range")

  loss_function = torch.nn.CrossEntropyLoss()

  loss = train(model, model_name, loss_function, train_dataloader, val_dataloader, optimizer, lr_scheduler, NUM_EPOCHS, results, num_classes, class_names, DEVICE)
  
  test(model, test_dataloader, results, num_classes, DEVICE)

  results.sava_data_on_csv()



def get_prediction(model_name):

  model, class_names = load_model(model_name)

  predict(model, model_name, class_names, PRED_DIR, OUTPUT_DIR)



if __name__ == '__main__':
  train_model(MODEL_NAME)
  get_prediction(MODEL_NAME)
