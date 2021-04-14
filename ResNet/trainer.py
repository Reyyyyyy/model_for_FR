import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from ReynetV3 import ReyNet

def load_data(normalization=True):
    train_xs = np.load(r'G:\lab\1\datasets\ECM\dataset\enhanced_ecm\enhanced_train_batches.npy', allow_pickle=True)
    train_ys = np.load(r'G:\lab\1\datasets\ECM\dataset\enhanced_ecm\enhanced_train_labels.npy', allow_pickle=True)
    test_xs = np.load(r'G:\lab\1\datasets\ECM\dataset\new_arange_dataset\test_batches.npy', allow_pickle=True)
    test_ys = np.load(r'G:\lab\1\datasets\ECM\dataset\new_arange_dataset\test_labels.npy', allow_pickle=True)
    train_xs = np.transpose(train_xs, [0, 3, 1, 2])
    test_xs = np.transpose(test_xs, [0, 3, 1, 2])
    train_ys = np.argmax(train_ys, axis=1)
    test_ys = np.argmax(test_ys, axis=1)
    if normalization:
        train_xs = train_xs / 255.
        test_xs = test_xs / 255.
    return (train_xs, train_ys), (test_xs, test_ys)

def adjust_lr(optimizer, lr):
    """手动调整学习率"""
    for params in optimizer.param_groups:
        params['lr'] = lr

if __name__ == '__main__':
    # Hyper parmas
    lr = 1e-3
    batch_size = 8
    epochs = 80
    weight_decay_rate = 1e-2
    momentum = 0.9

    # load data
    (x_train, y_train), (x_test, y_test) = load_data(normalization=True)
    x_vali = x_test[:500]
    y_vali = y_test[:500]

    # arange data
    with torch.no_grad():
        x_train = torch.autograd.Variable(torch.Tensor(x_train)).cuda()
        y_train = torch.autograd.Variable(torch.Tensor(y_train)).cuda()
        x_test = torch.autograd.Variable(torch.Tensor(x_test)).cuda()
        y_test = torch.autograd.Variable(torch.Tensor(y_test)).cuda()
        x_vali = torch.autograd.Variable(torch.Tensor(x_vali)).cuda()
        y_vali = torch.autograd.Variable(torch.Tensor(y_vali)).cuda()
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    vali_data = torch.utils.data.TensorDataset(x_vali, y_vali)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(vali_data, batch_size=batch_size, shuffle=True)

    # load model
    model = ReyNet((224, 224, 3), 2).cuda()

    # define cost and optim
    cost = nn.CrossEntropyLoss()
    losses = []
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-3, weight_decay=weight_decay_rate)

    # define accuracy
    total_train_steps = x_train.shape[0] / batch_size
    total_valid_steps = x_vali.shape[0] / batch_size
    accs_train = 0
    accs_valid = 0
    accs_test = 0
    train_accs_list = []
    valid_accs_list = []

    # train
    for epoch in range(epochs):
        # decreasing learning rate
        """
        if epoch > 45 and lr >= 1e-3:
            if (valid_accs_list[-1] - valid_accs_list[-2]) < 1e-3:
                lr /= 10
                adjust_lr(optimizer,lr)
            elif valid_accs_list[-1] < valid_accs_list[-2]:
                lr /= 10
                adjust_lr(optimizer,lr)
        if epoch == 70:
            lr /= 10
            adjust_lr(optimizer, lr)
            """
        if epoch == 30:
            lr /= 10
            adjust_lr(optimizer, lr)

        elif epoch == 50:
            lr /= 10
            adjust_lr(optimizer, lr)

        for step, (batch_x, batch_y) in enumerate(train_loader):
            # forward  propagation
            output = model(batch_x)
            loss = cost(output, batch_y.long())
            accs_train += ((output.argmax(axis=1) == batch_y).sum()).tolist() / batch_size
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for step, (batch_x, batch_y) in enumerate(vali_loader):
            output = model(batch_x)
            accs_valid += ((output.argmax(axis=1) == batch_y).sum()).tolist() / batch_size
        # store loss and acc
        acc_train = accs_train / (total_train_steps)
        acc_valid = accs_valid / (total_valid_steps)
        print(
            'Epoch[{}/{}]，Loss = {:.4f}，Accuracy = {:.4f} , Validation Accuracy = {:.4f} , Learning Rate = {}\n'.format(
                epoch + 1, epochs, loss.data.tolist(), acc_train, acc_valid, lr))
        train_accs_list.append(acc_train)
        valid_accs_list.append(acc_valid)
        losses.append(loss)
        accs_valid = 0
        accs_train = 0

    # save
    torch.save(model, r'my_model')

    # evaluate
    print('Testing\n')
    model.eval()
    for step, (batch_x, batch_y) in enumerate(test_loader):
        output = model(batch_x)
        accs_test += ((output.argmax(axis=1) == batch_y).sum()).tolist() / batch_size
    print('Test Accuracy：{:.4f}'.format(accs_test / (x_test.shape[0] / batch_size)))

    # plot accuracy and loss
    plt.subplot(1, 2, 1)
    plt.plot(valid_accs_list, label='valid')
    plt.plot(train_accs_list, label='train')
    plt.xlabel('epoch')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()










