import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from needle.data import MNISTDataset, DataLoader

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(main), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    main = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                norm=norm,
                drop_prob=drop_prob,
            )
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    return main
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_fn = nn.SoftmaxLoss()
    total_loss = []
    err_num = 0.0
    for x, y in dataloader:
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss.append(loss.numpy())
        err_num += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
        if opt is not None:  # training
            opt.reset_grad()
            loss.backward()
            opt.step()
    return err_num / len(dataloader.dataset), np.mean(total_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_set = MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
