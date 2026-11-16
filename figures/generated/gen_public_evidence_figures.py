#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = ROOT / 'artifacts' / 'results'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': .18,
    'grid.linewidth': .6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': .04,
})
C = {'ours':'#0072B2','baseline':'#8C8C8C','control':'#E69F00','accent':'#009E73','legacy':'#D55E00'}

def load(name):
    with (RES/name).open() as f: return json.load(f)

def save(fig, stem):
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.png', dpi=300)
    plt.close(fig)

def plantspd():
    d=load('plantspd_pr_pv_mfaps_summary.json')['matched_results']
    names=['Fixed-k','Effective\nnumber','PR-PV-MFAPS']
    keys=['fixed_k_matched_B02','effective_number','pr_pv_mfaps']
    macro=[d[k]['held_macro_f1'] for k in keys]
    tail=[d[k]['held_tail10_f1'] for k in keys]
    fig, ax = plt.subplots(1,2, figsize=(6.8,2.45), sharey=True)
    x=np.arange(len(names)); w=.66
    for a,vals,label in [(ax[0],macro,'HELD Macro-F1'),(ax[1],tail,'HELD Tail-10% F1')]:
        bars=a.bar(x, vals, width=w, color=[C['baseline'],C['control'],C['ours']], edgecolor='white')
        a.set_xticks(x, names)
        a.set_ylim(0, .55)
        a.set_ylabel(label)
        a.yaxis.set_major_formatter(PercentFormatter(1.0))
        for b,v in zip(bars, vals): a.text(b.get_x()+b.get_width()/2, v+.012, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax[0].set_xlabel('PlantSPD, 2 partitions × 3 seeds')
    ax[1].set_xlabel('same matched protocol')
    fig.tight_layout(w_pad=1.1)
    save(fig,'fig_plantspd_matched')

def cross_dataset():
    a=load('TRR_AUDITED_PREFIX10_VS_LA_SUMMARY.json')['rows']
    b=load('TRR_AUDITED_C10_PREFIX10_VS_LA_SUMMARY.json')['mean_std']
    # CIFAR100 summary rows
    c100={'LA_macro':a[1]['held_macro_mean'],'TRR_macro':a[0]['held_macro_mean'],'LA_tail':a[1]['held_tail10_mean'],'TRR_tail':a[0]['held_tail10_mean']}
    c10={'LA_macro':b['la_macro']['mean'],'TRR_macro':b['prefix_macro']['mean'],'LA_tail':b['la_tail10']['mean'],'TRR_tail':b['prefix_tail10']['mean']}
    fig, ax=plt.subplots(1,2,figsize=(6.8,2.5),sharey=True)
    for i,(title,d) in enumerate([('CIFAR-100-LT, IR=100',c100),('CIFAR-10-LT, IR=100',c10)]):
        x=np.arange(2); w=.34
        m=ax[i].bar(x-w/2,[d['LA_macro'],d['TRR_macro']],w,label='Macro-F1',color=C['baseline'])
        t=ax[i].bar(x+w/2,[d['LA_tail'],d['TRR_tail']],w,label='Tail-10% F1',color=C['ours'])
        ax[i].set_xticks(x,['LA','Stage-TRR-10']); ax[i].set_ylim(0,.9); ax[i].set_title(title)
        ax[i].yaxis.set_major_formatter(PercentFormatter(1.0)); ax[i].grid(axis='y')
        for bars in [m,t]:
            for bar in bars: ax[i].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f'{bar.get_height():.3f}',ha='center',fontsize=7)
    ax[0].set_ylabel('HELD score')
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', bbox_to_anchor=(.5,.0), ncol=2)
    fig.tight_layout(w_pad=1.2, rect=(0,.10,1,1)); save(fig,'fig_cross_dataset_stage_trr')

def long_matrix():
    d=load('cifar100_ir100_200_v2_summary.json')['matched_results']
    names=['ERM','BS','LA','LDAM-DRW','Stage-TRR']
    keys=['erm','balanced_softmax','la_control','ldam_drw','stage_trr10']
    fig,ax=plt.subplots(1,2,figsize=(6.8,2.55),sharey=True)
    x=np.arange(len(names)); colors=[C['baseline'],C['control'],C['control'],C['baseline'],C['ours']]
    for a,metric,label in [(ax[0],'macro_f1','Macro-F1'),(ax[1],'tail10_f1','Tail-10% F1')]:
        vals=[d[k][metric]['mean'] for k in keys]; errs=[d[k][metric]['std'] for k in keys]
        bars=a.bar(x,vals,yerr=errs,capsize=2.2,color=colors,edgecolor='white',error_kw={'elinewidth':.8})
        a.set_xticks(x,names); a.tick_params(axis='x', labelsize=7, rotation=18); a.set_ylim(0,.5); a.yaxis.set_major_formatter(PercentFormatter(1.0)); a.set_ylabel(label)
        for bar,v in zip(bars,vals): a.text(bar.get_x()+bar.get_width()/2,v+.015,f'{v:.3f}',ha='center',fontsize=6.7)
    ax[0].set_xlabel('CIFAR-100-LT IR=100, 200 epochs, mean ± std (3 seeds)')
    ax[1].set_xlabel('same ResNet-32 training contract')
    fig.tight_layout(w_pad=1.0); save(fig,'fig_cifar100_long_matrix')

def fused_system():
    d=load('fused_execution_summary.json'); eq=d['equivalence']; tm=d['sequential_timing']
    fig,axs=plt.subplots(1,3,figsize=(6.8,2.45))
    vals=[[eq['physical_encoder_forwards']['legacy'],eq['physical_encoder_forwards']['fused']], [tm['legacy_flops'],tm['fused_flops']], [tm['legacy_wall_seconds'],tm['fused_wall_seconds']]]
    labels=['Encoder\nforwards','FLOPs','Wall-clock\n(seconds)']
    ylabels=['count','log scale','seconds']
    for i,(a,v,lab,yl) in enumerate(zip(axs,vals,labels,ylabels)):
        bars=a.bar(['Legacy','Fused'],v,color=[C['legacy'],C['ours']],edgecolor='white')
        if i==1: a.set_yscale('log'); a.yaxis.set_major_formatter(ScalarFormatter())
        a.set_ylabel(yl); a.set_title(lab); a.grid(axis='y')
        for bar,val in zip(bars,v):
            txt=f'{val:.2e}' if i==1 else (f'{val:.1f}' if i==2 else f'{val:,}')
            a.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.08,txt,ha='center',fontsize=7)
    fig.tight_layout(w_pad=1.3); save(fig,'fig_fused_system_cost')

def solver():
    d=load('pv_mfaps_solver_benchmark.json')
    fig,ax=plt.subplots(1,2,figsize=(6.8,2.35))
    ax[0].bar(['Integer\ninstances','Quantized\ninstance'],[d['exactness']['test_cases'],1],color=[C['ours'],C['accent']],width=.6)
    ax[0].set_ylabel('verification count'); ax[0].set_ylim(0,14); ax[0].text(0,12.4,'12/12 exact',ha='center',fontsize=8)
    x=np.arange(2); vals=[d['scale_benchmark']['classes_69_wall_seconds'],d['scale_benchmark']['classes_200_wall_seconds']]
    ax[1].bar(x,vals,color=[C['ours'],C['control']],width=.6); ax[1].set_xticks(x,['69 classes','200 classes']); ax[1].set_ylabel('solver time (s)')
    for xx,v in zip(x,vals): ax[1].text(xx,v+.03,f'{v:.2f}',ha='center',fontsize=8)
    fig.tight_layout(w_pad=1.2); save(fig,'fig_solver_exactness_scale')

if __name__=='__main__':
    plantspd(); cross_dataset(); long_matrix(); fused_system(); solver()
    print('generated',len(list(OUT.glob('fig_*.pdf'))),'PDF figures in',OUT)
