import torch
from unet import UNet
from medical_dataset import MedicalDataset
import torch.optim as optim
from loss import Active_Contour_Loss
from tqdm import tqdm
# import torchvision.transforms as transforms


import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    description='train model')
parser.add_argument('-label', default=1, type=int,
                    help='int: label (1, 2 or 3)')
parser.add_argument('-loss', default="CE", help='str: loss (CE or AC or MSE)')
parser.add_argument('-epochs', default=9, type=int,
                    help='int: number of epochs')
parser.add_argument('-lr', default=10**-4, type=float,
                    help='float: learning rate')

args = parser.parse_args()


def CE_loss(y_pred, y_true):
    eps = 10**-6
    loss = y_true*torch.log(y_pred*(1-eps)+eps) + \
        (1-y_true)*torch.log((1-y_pred)*(1-eps)+eps)
    return -loss.mean()


def MSE_loss(y_pred, y_true):
    return ((y_pred-y_true)**2).mean()


if not(args.loss in ["CE", "MSE", "AC"]):
    raise argparse.ArgumentTypeError('Invalid loss')

label = int(args.label)
loss = {"CE": CE_loss,  "MSE": MSE_loss, "AC": Active_Contour_Loss}[args.loss]
sigmoid = True if args.loss == "CE" else False


# cust_transforms = [
#     transforms.Resize((256, 256)),
#     transforms.ColorJitter(0.2, 0.2),
#     transforms.GaussianBlur(3, sigma=(1.0, 2.0)),
# ]


D_train = MedicalDataset(label, "data_train")
D_val = MedicalDataset(label, "data_validation")
D_test = MedicalDataset(label, "data_test")

train_loader = torch.utils.data.DataLoader(
    D_train, batch_size=5, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    D_val, batch_size=5, shuffle=False
)

model = UNet(1, 1)
model.to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)


def train(model, optimizer, epoch, loader, loss_func, sigmoid=False):
    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(loader):
        if device == "cuda":
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if sigmoid:
            output = torch.sigmoid(output)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 20 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader.dataset),
                    100.0 * batch_idx / len(loader),
                    loss.data.item(),
                ),
                flush=True,
            )
    return losses


def dice(a, b):
    return 2*(a*b).sum()/(a+b).sum()


def evaluate(model, loader, sigmoid=False):
    model.eval()
    dice_sum = 0
    dice_ind = 0
    for _, (data, target) in enumerate(tqdm(loader)):
        if device == "cuda":
            data, target = data.cuda(), target.cuda()

        output = (torch.sigmoid(model(data)) > 0.5) * \
            1.0 if sigmoid else (model(data) > 0.5)*1.0
        dice_sum += dice(output, target)
        dice_ind += 1

    return dice_sum/dice_ind


evalu = []

for i in range(args.epochs):
    train(model, optimizer, i, train_loader, loss, sigmoid)
    evalu.append(evaluate(model, val_loader, sigmoid))
    print(evalu)


torch.save(model.state_dict(),
           'models/unet{}_{}.pt'.format(args.label, args.loss))
