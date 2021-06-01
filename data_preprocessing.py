import os
import sys
import logging
import mxnet as mx
import numpy as np
import numbers
import time
import torch
import torch.distributed as dist
import torch.utils.data.distributed

from dataset import DataLoaderX
from torch.utils.data import DataLoader, Dataset
from utils.utils_logging import AverageMeter, init_logging

class MXFaceDatasetV2(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDatasetV2, self).__init__()
        # self.transform = transforms.Compose(
        #     [transforms.ToPILImage(),
        #      transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor(),
        #      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #      ])
        self.transform = None
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


def write_io(img_path, label_path, data):
    pass


if __name__ == "__main__":
    # configuration
    start_time = time.time()
    # print("{:.2f}".format(start_time))

    root_dir = "/home/zzz/pytorch/zzw/data/ms1m-retinaface-t1"
    local_rank = 0
    part = 20
    batchsize = 50
    output_dir = os.path.join(root_dir, "divided_version")
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = 0
    torch.cuda.set_device(local_rank)

    log_path = "ms1mv3_arcface_r50"
    if not os.path.exists(log_path) and rank is 0:
        os.makedirs(log_path)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, log_path)

    #
    print("Initialization Time:{:.2f}".format(time.time() - start_time))
    start_time = time.time()
    print("=>Start building dataset...")
    train_set = MXFaceDatasetV2(root_dir, local_rank)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
    # train_loader = DataLoaderX(
    #     local_rank=local_rank, dataset=train_set, batch_size=batchsize,
    #     sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    # No sampler specified
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batchsize,
        num_workers=2, pin_memory=True, drop_last=True)
    trainset_size = len(train_set)
    print("DataLoader Time:{:.2f}".format(time.time() - start_time))
    start_time = time.time()
    print("The total number of dataset is {}".format(trainset_size))
    batch_num = trainset_size // batchsize + 1
    batch_num_per_file = batch_num // part
    cnt = 0
    savefile = 1
   
    for step, (img, label) in enumerate(train_loader):
        cnt += 1
        write_io(os.path.join(output_dir, "train_{}.rec".format(savefile)), 
                 os.path.join(output_dir, "train_{}.idx".format(savefile)),
                 (img, label)
                )

        if cnt == batch_num_per_file:
            savefile += 1
    