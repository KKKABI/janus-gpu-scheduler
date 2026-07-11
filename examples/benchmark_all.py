"""
Unified Opara benchmark script.
Only change `batch_size` below.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from Opara import GraphCapturer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# >>>  只需修改这一个变量  <<<
batch_size = 1
# ============================================================

cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')

def flush_cache():
    cache.zero_()

def run_parallel_graph(opara_model, inputs, iterations, warm_ups):
    time_list = []
    for _ in range(iterations):
        flush_cache()
        torch.cuda._sleep(1_000_000)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        output = opara_model(*inputs)
        end.record()
        end.synchronize()
        time_list.append(start.elapsed_time(end))
    output_str = ('Time of Opara:', f'{np.mean(time_list):.4f} ms', f'std: {np.std(time_list):.4f}')
    print('{:<30} {:<22} {:<20}'.format(*output_str))
    return output

def section(name):
    print('\n' + '=' * 70)
    print(f'  {name}  (batch_size={batch_size})')
    print('=' * 70)


# ============================================================
# 1. GoogLeNet
# ============================================================
section('GoogLeNet')
import torchvision
warm_ups, iterations = 10, 300
x = torch.randint(low=0, high=256, size=(batch_size, 3, 224, 224), dtype=torch.float32, device='cuda')
model = torchvision.models.googlenet().eval().cuda()
inputs = (x,)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, x


# ============================================================
# 2. Inception-v3
# ============================================================
section('Inception-v3')
warm_ups, iterations = 100, 300
x = torch.randint(low=0, high=256, size=(batch_size, 3, 299, 299), dtype=torch.float32, device='cuda')
model = torchvision.models.inception_v3(pretrained=True).eval().cuda()
inputs = (x,)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, x


# ============================================================
# 3. NASNet-Large
# ============================================================
section('NASNet-Large')
warm_ups, iterations = 100, 300
sys.path.insert(0, '/home/lyx/.conda/envs/Opara/lib/python3.10/site-packages')
import pretrainedmodels
x = torch.randint(low=0, high=256, size=(batch_size, 3, 331, 331), dtype=torch.float32, device='cuda')
model = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet').eval().cuda()
inputs = (x,)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, x


# ============================================================
# 4. DeepFM
# ============================================================
section('DeepFM')
warm_ups, iterations = 100, 300
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from NCF import DeepFM
cate_fea_nuniqs = [100 * (i + 1) for i in range(32)]
nume_fea_size   = 16
model = DeepFM(cate_fea_nuniqs, nume_fea_size, emb_size=8,
               hid_dims=[256, 128], num_classes=1,
               dropout=[0.2, 0.2]).eval().cuda()
X_sparse = torch.randint(0, 100, (batch_size, len(cate_fea_nuniqs)), device='cuda')
X_dense  = torch.rand(batch_size, nume_fea_size, device='cuda')
inputs = (X_sparse, X_dense)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, X_sparse, X_dense


# ============================================================
# 5. BERT-base
# ============================================================
section('BERT-base')
warm_ups, iterations = 10, 300
seq_length = 16
bert_model_path = '/public_0/ZYF/model/bert-base'
sys.path.append(bert_model_path)
from transformers import BertModel
model = BertModel.from_pretrained(bert_model_path).eval().cuda()
input_ids      = torch.randint(0, 30000, (batch_size, seq_length), dtype=torch.long, device='cuda')
attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device='cuda')
inputs = (input_ids, attention_mask)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, input_ids, attention_mask


# ============================================================
# 6. YOLOv8x
# ============================================================
section('YOLOv8x')
warm_ups, iterations = 10, 100
from ultralytics import YOLO
yolo_model = YOLO('/public_0/ZYF/model/YOLOv8/yolov8x.pt').model.eval().cuda()
x = torch.randn(batch_size, 3, 320, 320, device='cuda')
inputs = (x,)

class _BackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.model[0]
    def forward(self, x):
        return self.backbone(x)

backbone = _BackboneWrapper(yolo_model).eval().cuda()
opara = GraphCapturer.capturer(inputs, backbone)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del yolo_model, backbone, opara, x


# ============================================================
# 7. ConvNeXt-Base
# ============================================================
section('ConvNeXt-Base')
warm_ups, iterations = 10, 300
x = torch.randn(batch_size, 3, 224, 224, device='cuda')
model = torchvision.models.convnext_base(pretrained=False).eval().cuda()
inputs = (x,)
opara = GraphCapturer.capturer(inputs, model)
run_parallel_graph(opara, inputs, iterations, warm_ups)
del model, opara, x

print('\n' + '=' * 70)
print('All benchmarks done.')
print('=' * 70)
