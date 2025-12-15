import argparse
import time
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ssim_loss import SSIMLossForSequence
from CCL import SENN
from CM import parametriser_regulariser
from tqdm import tqdm  # 引入 tqdm

def train(args, writer, model, device, trainloader, concept_opt, relevance_opt, cls_opt, cls_loss, rec_loss, epoch):

    # log metrics
    correct = 0
    train_loss = 0.0

    model.train()
    start_time = time.time()
    img_size = 28
    ssim_loss = SSIMLossForSequence(window_size=11, size_average=True)
    for batch_idx, (data, label) in enumerate(tqdm(trainloader, desc='Training')):
        data, label = data.to(device), label.to(device)
        data.requires_grad = True

        # reset grad
        concept_opt.zero_grad()
        relevance_opt.zero_grad()
        cls_opt.zero_grad()

        # senn output
        h, h_hat, theta, g = model(data)

        # loss + CM
        classification_loss = cls_loss(g, label)
        CM = parametriser_regulariser(data, h, theta, g, num_concepts=args.num_concepts,img_size=img_size)
        reconstruction_loss = rec_loss(h_hat, data.view(data.size(0), -1))
        total_loss = classification_loss + 2e-4*CM + 2e-5*reconstruction_loss+2e-5*ssim_loss(h_hat,  data.view(data.size(0), -1),device)

        # update grad
        total_loss.backward()
        concept_opt.step()
        relevance_opt.step()
        cls_opt.step()

        # update log metrics
        train_loss += total_loss.sum().item()
        pred = g.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

    train_loss /= len(trainloader.dataset)
    train_acc = 100. * correct / len(trainloader.dataset)

    # post log metrics
    writer.add_scalar('loss/train', train_loss, epoch)
    writer.add_scalar('accuracy/train', train_acc, epoch)
    print('Epoch {:<2} {:5}  AvgLoss:{:.4f}  Accuracy:{:.2f}%  Time:{:.2f}s'.format(epoch,
                                                                                    '[train]',
                                                                                    train_loss,
                                                                                    train_acc,
                                                                                    time.time() - start_time))

def val(args, writer, model, device, valloader, cls_loss, rec_loss, epoch):

    # log metrics
    val_loss = 0.0
    correct = 0

    model.eval()
    start_time = time.time()
    img_size = 28
    for batch_idx, (data, label) in enumerate(tqdm(valloader, desc='Testing')):
        data, label = data.to(device), label.to(device)
        data.requires_grad = True

        # senn output
        h, h_hat, theta, g = model(data)

        # loss + CM
        classification_loss = cls_loss(g, label)
        CM = parametriser_regulariser(data, h, theta, g, num_concepts=args.num_concepts,img_size=img_size)
        reconstruction_loss = rec_loss(h_hat, data.view(data.size(0), -1))
        total_loss = classification_loss + 2e-4 * CM + 2e-5 * reconstruction_loss

        # update log metrics
        val_loss += total_loss.sum().item()  # sum up batch loss
        pred = g.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(valloader.dataset)
    val_acc = 100. * correct / len(valloader.dataset)

    # post log metrics
    writer.add_scalar('loss/val', val_loss, epoch)
    writer.add_scalar('accuracy/val', val_acc, epoch)
    print('Epoch {:<2} {:7}  AvgLoss:{:.4f}  Accuracy:{:.2f}%  Time:{:.2f}s'.format(epoch,
                                                                                    '[val]',
                                                                                    val_loss,
                                                                                    val_acc,
                                                                                    time.time() - start_time))
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch SENN')
    parser.add_argument('--num-concepts', type=int, default=5, metavar='N', help='number of concepts (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers for dataloader (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for Adam optimiser')
    parser.add_argument('--seed', type=int, default=1337, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset(minist,ASL_Alphabet)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter()

    if args.dataset=='minist':
        channel=1
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        valset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        class_num=10
    elif args.dataset=='cifar10':
        channel = 3
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR-10的均值和标准差
        ])


        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_num = len(trainset.classes)
    elif args.dataset == 'FashionMNIST':
        channel = 1  # fashion-mnist 是灰度图像
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Fashion-MNIST的均值和标准差（假设）
        ])

        
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        class_num = len(trainset.classes)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

    model = SENN(args.num_concepts,channel,class_num).to(device)

    concept_parameters = list(model.concept_autoencoder.parameters())
    concept_optimizer = optim.Adam(concept_parameters, lr=args.lr)

    relevance_parameters = list(model.relevance_parametrizer.parameters())
    relevance_optimizer = optim.Adam(relevance_parameters, lr=args.lr)

    classification_paramteres = list(model.aggregator.parameters())
    classification_optimizer = optim.Adam(classification_paramteres, lr=args.lr)

    cls_loss = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()

    best_val_loss = 100
    for epoch in range(1, args.epochs + 1):
        train(args, writer, model, device, trainloader, concept_optimizer, relevance_optimizer, classification_optimizer, cls_loss, reconstruction_loss, epoch)
        val_loss = val(args, writer, model, device, valloader, cls_loss, reconstruction_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "senn_clip_best_model-cifar4"+args.dataset+'.pt')


if __name__ == '__main__':
    main()
