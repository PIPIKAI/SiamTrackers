# Copyright (c) SenseTime. All Rights Reserved.
import argparse
import os
import torch
import sys 
from rknn.api import RKNN

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg

from nanotrack.utils.model_load import load_pretrain
from nanotrack.models.model_builder import ModelBuilder

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target_platform = 'rk3588'
parser = argparse.ArgumentParser(description='lighttrack')

parser.add_argument('--config', type=str, default='./models/config/configv3.yaml',help='config file')

parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str,  help='snapshot models to eval')

args = parser.parse_args()


def convert(model, input_size , output_name):
    rknn = RKNN(verbose=True)
    # Pre-process config
    print('--> Config model')
    # 注意要将target_platform指定清楚
    rknn.config( target_platform=target_platform)
    print('done')
    
    print(f'--> Exporting model {model} to RKNN')
    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(f'./output/{output_name}.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    
rk_backbone_T = './output/pt/v3_backbone_127.pt'
rk_backbone_X = './output/pt/v3_backbone_255.pt'
rk_head = './output/pt/v3_head.pt'
    
def main():

    cfg.merge_from_file(args.config)


    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

    model = ModelBuilder()

    model = load_pretrain(model, pretrained_path=args.snapshot)
    
    model.eval().to(device)

    backbone_net = model.backbone

    head_net = model.ban_head
    
    
    
    # backbone 模板特征提取模型
    trace_model = torch.jit.trace(backbone_net, torch.Tensor(1, 3, 127, 127))
    trace_model.save(rk_backbone_T)
    
    # backbone 图像特征提取模型
    trace_model = torch.jit.trace(backbone_net, torch.Tensor(1, 3, 255, 255))
    trace_model.save(rk_backbone_X)
    
    # head 模型
    trace_model = torch.jit.trace(head_net, (torch.Tensor(1, 96, 8, 8), torch.Tensor(1, 96, 16, 16)))
    trace_model.save(rk_head)

   
    convert(rk_backbone_T, [[1, 3, 127, 127]], "v3_backbone_127")
    convert(rk_backbone_X, [[1, 3, 255, 255]] , "v3_backbone_255")
    convert(rk_head, [[1, 96, 8, 8] ,[1, 96, 16, 16] ], "v3_head")

    

if __name__ == '__main__':
    main() 
