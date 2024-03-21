import numpy as np
import torch


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    del xx
    del yy
    del zz
    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def MMD_with_sample(x, y, split_size, iterations, kernal):
    '''Big brain time'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_mmd = []
    for split in range(iterations):
        b1 = np.random.choice(x, split_size, replace=True)
        b2 = np.random.choice(y, split_size, replace=True)
        tensor_a = torch.from_numpy(np.reshape(b1, (len(b1), 1))).to(device)
        tensor_b = torch.from_numpy(np.reshape(b2, (len(b2), 1))).to(device)
        all_mmd.append(MMD(tensor_a, tensor_b, kernel=kernal).item())

    all_mmd = np.array(all_mmd)
    mean_mmd = np.mean(all_mmd)
    std_dev_mmd = np.std(all_mmd)

    return mean_mmd, std_dev_mmd
