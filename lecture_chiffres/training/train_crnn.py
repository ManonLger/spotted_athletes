from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from warpctc_pytorch import CTCLoss

import os
from utils import *
from dataset import *

from models.crnn import *

# Settings
trainroot="/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/lecture_chiffres/training/data/data_training"
valroot="/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/lecture_chiffres/training/data/data_test"
workers=1
batchSize=1
imgH=32
imgW=100
nh=256
niter=5
lr=0.01
beta1=0.5
ngpu=1
crnn_adress="/Users/eymardhoudeville/Documents/Centrale/Info/visord/spotted_athletes/lecture_chiffres/models/crnn.pth"
alphabet='0123456789'
experiment="save_model/"
displayInterval=500
n_test_disp=10
valInterval=500
saveInterval=500
adam='store_true'
adadelta='store_true'
keep_ratio='store_true'
random_sample='store_false'

if experiment is None:
    experiment = 'expr'
os.system('mkdir {0}'.format(experiment))

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

env = lmdb.open(trainroot)

with env.begin() as txn:
    length = txn.stat()['entries']
    print("Database number of entries is %i" %length)

train_dataset = lmdbDataset(root=trainroot)

print("Dataset is of length %i" %len(train_dataset))
assert train_dataset
if not random_sample:
    sampler = randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_dataset = lmdbDataset(
    root=valroot)


nclass = len(alphabet) + 1
nc = 1

converter = strLabelConverter(alphabet)
criterion = CTCLoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = CRNN(imgH, nc, nclass, nh)
crnn.apply(weights_init)

pre_trainmodel = torch.load(crnn_adress)

model_dict = crnn.state_dict()

# replace the classfidy layer parameters
for k,v in model_dict.items():

    if not (k == 'rnn.1.embedding.weight' or k == 'rnn.1.embedding.bias'):
        model_dict[k] = pre_trainmodel[k]


crnn.load_state_dict(model_dict)

# We only finetune the last layers (RNN)
crnn.cnn.requires_grad=False
crnn.rnn.requires_grad=True

image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)


image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = averager()

# setup optimizer
if adam:
    optimizer = optim.Adam(crnn.parameters(), lr=lr,
                           betas=(beta1, 0.999))
elif adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batchSize, num_workers=int(workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        loadData(text, t)
        loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)

    #loadData(text, t)
    #loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(niter):
    print("Epoch %i" % epoch)

    train_iter = iter(train_loader)
    i = 0

    while i < len(train_loader):
        print("Dealing with %i th batch" %i)
        crnn.train()
        i = i + 1
        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)

        if i % displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % saveInterval == 0:
            print("Saving model...")
            torch.save(
                crnn.state_dict(), '{0}/TrainingNetCRNN_{1}_{2}.pth'.format(experiment, epoch, i))
