import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_Mamba import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.UNetFormer import UNetFormer
from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'UNetformer':
    net = UNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'RS3Mamba':
    net = RS3Mamba(num_classes=N_CLASSES).cuda()
    net = load_pretrained_ckpt(net)

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

def coltime(str1, str2):
    seconds = str2 - str1
    minute = seconds // 60
    seconds = seconds - minute * 60
    hours = minute // 60
    minute = minute - hours * 60
    return 'hour:' + str(hours) + ", minute:" + str(minute) + ",  seconds:" + str(seconds);

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# Load the datasets
print("training : ", str(len(train_ids)) + ", testing : ", str(len(test_ids)) + ", Stride_Size : ", str(Stride_Size), ", BATCH_SIZE : ", str(BATCH_SIZE))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,pin_memory=True)

base_lr = 0.015
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 25, 35], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Urban':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False, mininterval=0):
            # time.sleep(0.1)
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
            
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.01
    starttime = time.time()
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = loss_calc(output, target, weights)

            loss.backward()
            optimizer.step()

            # losses[iter_] = loss.data
            # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

            ## !!! You can increase the frequency of testing to find better models.
            if iter_ % 999 == 0:
                endtime = time.time()
                test_starttime = time.time()
                print("train_time:",coltime(starttime, endtime))
                MIoU_best = 0.00
                # We validate with the largest possible stride for faster computing
                net.eval()
                MIoU = test(net, test_ids, all=False, stride=Stride_Size)
                net.train()
                if MIoU > MIoU_best:
                    if DATASET == 'Vaihingen':
                        torch.save(net.state_dict(), '/root/autodl-tmp/RS3Mamba/result_weight/Vaihingen/5/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    elif DATASET == 'Urban':
                        torch.save(net.state_dict(), '/root/autodl-tmp/RS3Mamba/result_weight/Urban/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                    MIoU_best = MIoU
                test_endtime = time.time()
                print("test_time:",coltime(test_starttime, test_endtime))

if MODE == 'Train':
    train(net, optimizer, 70, scheduler)
elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('/root/autodl-tmp/RS3Mamba/result_weight/Vaihingen/4/RS3Mamba_epoch28_0.8282291956445895'), strict=False) # seg
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsv/inference_oRS3_'+MODEL+'_tile_{}.png'.format(id_), img)

    elif DATASET == 'Urban':
        net.load_state_dict(torch.load('/root/autodl-tmp/RS3Mamba/result_weight/Urban/RS3Mamba_epoch50_0.1685224719961308'), strict=False)  # seg
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsu/inference5058_'+MODEL+'_tile_{}.png'.format(id_), img)
