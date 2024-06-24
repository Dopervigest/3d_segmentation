import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from scripts.models import Segmentation_Network_full
from scripts.models import resnet18 as resnet
from scripts.models import UNet
from scripts.unet import Unet_Classification_head as Classification_head
from scripts.resnet import ResnetDecoder


from scripts.dataset import H5Dataset
from scripts.utils import plot_loss, plot_accuracy

from scripts.trainer import Trainer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using ", device, "for training")
print()


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='For how many epochs to train', required=True)
parser.add_argument('--batch', type=int, help='The size of mini-batch', required=True)
parser.add_argument('--model', type=str, help='What model to train', required=True)
parser.add_argument('--dataset', type=str, help='What dataset to use', required=True)
parser.add_argument('--tl', type=bool, help='Use transfer learning?')

args = vars(parser.parse_args())

batch_size = args['batch'] # 128
epochs = args['epochs'] # 100


out_dir = './models/'


print(f'''Got the following parameters: \n 
            model: {args['model']}; \n
            dataset: {args['dataset']}; \n
            epochs: {args['epochs']}; \n
            batch: {args["batch"]}; \n
            transfer learning: {args["tl"]}. \n''')


if args['dataset'] == 'clinical':
    out_dir += 'clinical'
    n_classes = 4
    train_dataset = H5Dataset('./TL_prepared_data/train/data.hdf5')
    test_dataset = H5Dataset('./TL_prepared_data/test/data.hdf5')

elif args['dataset'] == 'acdc':
    out_dir += 'ACDC'
    n_classes = 3

    train_dataset = H5Dataset('./prepared_data/train/data.hdf5')
    test_dataset = H5Dataset('./prepared_data/test/data.hdf5')

else:
    raise ValueError('Dataset must be "clinical" or "acdc"')


if args['model'] == "cnn":

    if args['tl']:
        out_dir += '_TL'
        m = torch.load('./models/ACDC/best.pt').cpu()
        m.eval()
        model = Segmentation_Network_full(n_classes=n_classes).float()

        for i in range(len(m.network)):
            layer = m.network[i]
            if type(layer) == torch.nn.modules.conv.Conv3d and layer.in_channels != 256:
                model.network[i].load_state_dict(layer.state_dict())

                for param in model.network[i].parameters():
                    param.requires_grad = False

    elif args['dataset'] == 'acdc':
        model = Segmentation_Network_full(n_classes=n_classes).float()

    else:
        out_dir += '_no_TL'
        model = Segmentation_Network_full(n_classes=n_classes).float()


elif args['model'] == "unet":

    if args['tl']:
        out_dir += '_TL_Unet'
        model = torch.load('./models/ACDC_Unet/best.pt').cpu()
        for i in model.parameters():
            i.requires_grad = False
        model.head = Classification_head(n_classes = n_classes)
    
    elif args['dataset'] == 'acdc':
        out_dir += '_Unet'
        model = UNet(n_classes=n_classes).float()

    else:
        out_dir += '_no_TL_Unet'
        model = UNet(n_classes=n_classes).float()

elif args['model'] == "resnet":

    if args['tl']:
        out_dir += '_TL_resnet'
        model = torch.load('./models/ACDC_resnet/best.pt').cpu()
        for i in model.parameters():
            i.requires_grad = False

        model.decoder = ResnetDecoder(model.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    elif args['dataset'] == 'acdc':
        out_dir += '_resnet'
        model = resnet(1, n_classes=n_classes).float()
    
    else:
        out_dir += '_no_TL_resnet'
        model = resnet(1, n_classes=n_classes).float()

else:
    raise ValueError('Model must be "cnn", "unet" or "resnet"')

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
criterion = F.binary_cross_entropy  
model.to(device).float() 


print('Path for model: ', out_dir)
if os.path.exists(out_dir):
    print("path exists")
else:
    os.makedirs(out_dir)

print()
print()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Loaded dataset : ", len(train_dataset), len(test_dataset))


print('Training started')
trainer = Trainer(model, opt, criterion, train_dataloader, test_dataloader, out_dir, epochs, device)
trainer.fit()
print('Training finished')


print('Plotting results')
# Plot results
plot_loss(out_dir, '/logs.csv')
plot_accuracy(out_dir, '/logs.csv')


print('Making classification report')
# Classification report
preds = []
labels = []

model = torch.load(f'{out_dir}/best.pt') 
model.eval() 
for x,y in iter(test_dataloader): 
    x = x.to(device)

    pred = model(x) 
    preds.append(pred.detach().cpu())

    labels.append(y)

preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)

preds = np.argmax(preds, axis=1)
labels = np.argmax(labels, axis=1)

report = classification_report(y_true=labels, y_pred=preds)
print(report)

# Saving report to a file
report_for_csv = classification_report(y_true=labels, y_pred=preds, output_dict=True)
df_out = pd.DataFrame(report_for_csv).transpose()
df_out.to_csv(out_dir + '/classification_report.csv') 
print(df_out)


print('Making confusion matrix')
# Confusion matrix
cm = confusion_matrix(y_true=labels,
                      y_pred=preds)

plt.Figure()
disp = ConfusionMatrixDisplay(cm,)

disp.plot(values_format='')
plt.savefig(out_dir + '/cm.png')


print('SUCCESS!')
