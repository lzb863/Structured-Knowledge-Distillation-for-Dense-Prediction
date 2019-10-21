import sys
from utils.train_options import TrainOptions
from networks.kd_model import NetModel
from networks.Seg_kd_model import Seg_NetModel
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from dataset.datasets import CSDataSet
import numpy as np
from tensorboardX import SummaryWriter

LOG_DIR = './logs/struct_dis_res50_to_res18_0.5_256_256'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
args = TrainOptions().initialize()
h, w = map(int, args.input_size.split(','))

trainloader = data.DataLoader(CSDataSet(args.data_dir, '/home/remo/Desktop/structure_knowledge_distillation-master/dataset/demo/tran.txt', max_iters=args.num_steps*args.batch_size, crop_size=(h, w),
                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN), 
                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader = data.DataLoader(CSDataSet(args.data_dir, '/home/remo/Desktop/structure_knowledge_distillation-master/dataset/demo/tran.txt', crop_size=(256, 256), mean=IMG_MEAN, scale=False, mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True)
# save_steps = int(2975/args.batch_size)
save_steps = 1
model = Seg_NetModel(args)
tblogger = SummaryWriter(LOG_DIR)
itr = 0
mean_IU, IU_array = 0,0
for epoch in range(args.start_epoch, args.epoch_nums):
    for step, data in enumerate(trainloader, args.last_step+1):
        '''
        data:[image,mask,img_shape,img_name]
        image:[B, N, W, H] torch.tensor
        mask:[B, W, H] torch.tensor
        img_shape:[[H, W, C],
                    [H, W, C]
                    ...
                    ] Batch_size个 [B,3]
        img_name:['img_name1',
                    'img_name2'] [B,1]
        '''
        model.adjust_learning_rate(args.lr_g, model.G_solver, step)
        model.adjust_learning_rate(args.lr_d, model.D_solver, step)
        model.set_input(data)
        model.optimize_parameters()
        model.print_info(epoch, step)
        # if (step > 1) and ((step % save_steps == 0) and (step > args.num_steps - 1000)) or (step == args.num_steps - 1):
        if (step > 1) and (step % save_steps == 0):
            mean_IU, IU_array = model.evalute_model(model.student, valloader, '0', '256,256', 9, True)
            model.save_ckpt(epoch, step, mean_IU, IU_array)
            logging.info('[val 512,512] mean_IU:{:.6f}  IU_array:{}'.format(mean_IU, IU_array))
            tblogger.add_scalar('mIoU', mean_IU, itr)
        itr += 1


