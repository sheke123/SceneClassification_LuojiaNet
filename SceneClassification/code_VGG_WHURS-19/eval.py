import os
import ast
import argparse
from unittest import result
import pandas as pd
from luojianet_ms import context, nn
from luojianet_ms.train.model import Model
from luojianet_ms.common import set_seed
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net

from config import *
from utils import create_dataset,CrossEntropySmooth
from vgg import *
# from Resnet import *
# from Resnet_se import *

set_seed(1)
CACHE = "/cache/data/"
CKPT_CACHE = "/cache/ckpt/"
import moxing as mox

if __name__ == '__main__':
    mox.file.copy_parallel(config.obs_checkpoint_path+'model/',CKPT_CACHE)
    mox.file.copy_parallel(src_url=config.dataset_path, dst_url=CACHE)
    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('--dataset_path', type=str, default=CACHE, help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')

    args_opt = parser.parse_args()
    context.set_context(device_target=args_opt.device_target)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                        do_train=False,
                        batch_size=config.batch_size)


    # define net
    net = vgg16_bn(num_classes=config.class_num)

    # eval_result_list
    eval_output=[]

    # load checkpoint
    for ckpt in os.listdir(CKPT_CACHE):
        print(ckpt)
        if(ckpt.endswith('ckpt')):

            step_size = dataset.get_dataset_size()
            param_dict = load_checkpoint(CKPT_CACHE+ckpt)
            load_param_into_net(net, param_dict)
            net.set_train(False)

            # define loss
            loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

            # define model
            eval_metrics = {'Loss': nn.Loss(),
                            'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                            'Top_5_Acc': nn.Top5CategoricalAccuracy(),
                            'acc':nn.Accuracy()}
            model = Model(net, loss_fn=loss, metrics=eval_metrics)

            # eval model
            res = model.eval(dataset,dataset_sink_mode=False)
            print("result:", res, "ckpt=", ckpt)
            eval_output.append([ckpt.split('-')[1].split('_')[0],res['Loss'],res['Top_1_Acc'],res['Top_5_Acc']])
            dataset.reset()
    name=['epoch','loss','Top_1_Acc','Top_5_Acc']
    pdfile=pd.DataFrame(columns=name,data=eval_output)
    pdfile.to_csv('/cache/output.csv')
    mox.file.copy_parallel('/cache/output.csv','obs://luojianet-benchmark/temp/output.csv')