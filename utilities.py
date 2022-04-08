import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def dice(a, b):
    return 2*(a*b).sum()/(a+b).sum()


def evaluate(model, loader, sigmoid=False):
    model.eval()
    dice_sum = 0
    dice_ind = 0
    for _, (data, target) in enumerate(loader):
        if device == "cuda":
            data, target = data.cuda(), target.cuda()

        output = (torch.sigmoid(model(data)) > 0.5) * \
            1.0 if sigmoid else (model(data) > 0.5)*1.0
        dice_sum += dice(output, target)
        dice_ind += 1

    return dice_sum/dice_ind
