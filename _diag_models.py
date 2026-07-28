"""Simulator diagnostic for remaining models"""
import sys, subprocess, re, json

PY = "/home/lyx/.conda/envs/opara/bin/python"

models = [
    {
        'name': 'Inception-v3',
        'setup': 'import torchvision; m=torchvision.models.inception_v3(weights=None).eval().cuda(); x=(torch.randn(1,3,299,299,device="cuda:0"),); inputs=x',
    },
    {
        'name': 'BERT',
        'setup': 'import sys; sys.path.insert(0,"/public_0/ZYF/model/bert-base"); from transformers import BertModel; m=BertModel.from_pretrained("/public_0/ZYF/model/bert-base").eval().cuda(); x=torch.randint(0,30522,(1,16),device="cuda"); inputs=(x,)',
    },
    {
        'name': 'NASNet',
        'setup': 'import pretrainedmodels; m=pretrainedmodels.__dict__["nasnetalarge"](num_classes=1000,pretrained="imagenet").eval().cuda(); x=(torch.randn(1,3,331,331,device="cuda:0"),); inputs=x',
    },
    {
        'name': 'YOLOv8x',
        'setup': 'from ultralytics import YOLO; m=YOLO("/public_0/ZYF/model/YOLOv8/yolov8x.pt").model.eval().cuda(); x=(torch.randn(1,3,320,320,device="cuda:0"),); inputs=x',
    },
]

for model_info in models:
    mname = model_info['name']
    setup = model_info['setup']

    for td_label, td in [('Static', False), ('TD', True)]:
        code = f'''
import sys; sys.path.insert(0,'/public_0/LYX/janus')
import torch, io
from contextlib import redirect_stdout
{setup}
f=io.StringIO()
with redirect_stdout(f):
    from Opara import GraphCapturer
    r=GraphCapturer.capturer(inputs,m,copy_outputs=False,alpha=0.9,selection_mode='cosine',time_domain={td})
try: r(*inputs); torch.cuda.synchronize()
except: pass
'''
        try:
            result = subprocess.run([PY, '-c', code], capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(f"{mname} {td_label}: TIMEOUT")
            continue
        except Exception as e:
            print(f"{mname} {td_label}: ERROR {e}")
            continue

        stderr = result.stderr
        data = {}
        for line in stderr.split('\n'):
            m = re.search(r'\[(ST|TD)\].*ready=(\d+)\s+total=(\d+)\s+feas=(\d+)', line)
            if m:
                rdy = int(m.group(2))
                t = int(m.group(3))
                f = int(m.group(4))
                if rdy not in data:
                    data[rdy] = {'tot': 0, 'feas': 0, 'calls': 0}
                data[rdy]['tot'] += t
                data[rdy]['feas'] += f
                data[rdy]['calls'] += 1

        if not data:
            print(f"\n{mname} {td_label}: NO DATA")
            continue

        total_enum = sum(v['tot'] for v in data.values())
        total_feas = sum(v['feas'] for v in data.values())
        rate = 100 * total_feas / total_enum if total_enum > 0 else 0

        print(f"\n{mname} {td_label}: enum={total_enum} feas={total_feas} rate={rate:.1f}%")
        print(f"  {'ready':>6} {'calls':>6} {'enum':>8} {'feas':>8} {'rate':>7}")
        for r in sorted(data.keys()):
            v = data[r]
            r2 = 100 * v['feas'] / v['tot'] if v['tot'] > 0 else 0
            print(f"  {r:>6} {v['calls']:>6} {v['tot']:>8} {v['feas']:>8} {r2:>6.1f}%")
