import numpy as np
import torch


def MMD(x, y, kernel, bandwidth_range):
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

        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2. * XY)


def MMD_with_sample(x, y, split_size, iterations, kernal, bandwidth_range):
    # print the shape of x and y
    print(x.shape)
    print(y.shape)
    '''Big brain time'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_mmd = []
    for split in range(iterations):
        # get a random number between 0 and len(x) - split_size
        rand_ind = np.random.randint(0, len(x) - split_size)
        b1 = x[rand_ind:rand_ind + split_size]
        # get a random number between 0 and len(y) - split_size
        rand_ind = np.random.randint(0, len(y) - split_size)
        b2 = y[rand_ind:rand_ind + split_size]
        tensor_a = torch.from_numpy(np.reshape(b1, (len(b1), 1))).to(device)
        tensor_b = torch.from_numpy(np.reshape(b2, (len(b2), 1))).to(device)
        all_mmd.append(MMD(tensor_a, tensor_b, kernal, bandwidth_range).item())

    all_mmd = np.array(all_mmd)
    mean_mmd = np.mean(all_mmd)
    std_dev_mmd = np.std(all_mmd)

    return mean_mmd, std_dev_mmd
