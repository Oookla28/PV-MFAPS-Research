#!/usr/bin/env python3
import argparse, json, copy, pickle, pathlib, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cifar100_lt_runner as r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--gpu',type=int,default=0); args=ap.parse_args()
    torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    dev=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    base=r.ensure_data(pathlib.Path(args.data_root)); x,y=r.read_batch(base/'train'); idx,counts=r.make_lt_indices(y,100,3407)
    ds=r.CifarLT(x[idx[:128]],y[idx[:128]],False,3407); loader=DataLoader(ds,batch_size=128,shuffle=False)
    xb,yb=next(iter(loader)); xb,yb=xb.to(dev),yb.to(dev)
    m0=r.CifarResNet(100).to(dev); state=copy.deepcopy(m0.state_dict());
    def run(legacy):
        m=r.CifarResNet(100).to(dev); m.load_state_dict(state); m.train(); opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9,weight_decay=5e-4); opt.zero_grad(set_to_none=True)
        feats=m.features(xb); logits=m.fc(feats); active=20; loss=r.loss_for('pv_mfaps',logits,yb,counts,0,dev); bn_restore={}
        if legacy:
            for _ in range(active):
                if not bn_restore:
                    for name,buf in m.named_buffers():
                        if name.endswith('running_mean') or name.endswith('running_var') or name.endswith('num_batches_tracked'):
                            bn_restore[name]=buf.detach().clone()
                m.train()
                with torch.no_grad(): aux_features=m.features(xb)
                aux=m.fc(aux_features.detach()); loss=loss+0.05*r.loss_for('pv_mfaps',aux,yb,counts,0,dev)
        else:
            aux=m.fc(feats.detach()); loss=loss+0.05*active*r.loss_for('pv_mfaps',aux,yb,counts,0,dev)
        loss.backward()
        for name,old_buf in bn_restore.items(): m.get_buffer(name).copy_(old_buf)
        before={k:v.detach().clone() for k,v in m.state_dict().items()}; opt.step(); after={k:v.detach().clone() for k,v in m.state_dict().items()}
        return float(loss.detach()),before,after
    torch.cuda.synchronize(); l0,b0,a0=run(False); torch.cuda.synchronize(); l1,b1,a1=run(True); torch.cuda.synchronize()
    diffs=[float((a0[k]-a1[k]).abs().max().item()) for k in a0]
    out={'status':'EXECUTED','loss_fused':l0,'loss_legacy':l1,'loss_abs_diff':abs(l0-l1),'state_max_abs_delta':max(diffs),'state_mean_max_abs_delta':float(np.mean(diffs)),'active':20,'held_reads':0,'semantics':'shared detached correction loss multiplied by active'}
    pathlib.Path(args.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
