import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from mden_battery.data.log_age import PrepConfig, resample_query, prepared_values, prepare_dataset, split_cells, validate_prepared_dataset
from mden_battery.data.window_batches import WindowBatchDataset
from mden_battery.data.article_preprocessing import WindowConfig
from mden_battery.feature_extraction import FeatureExtractionModule, fft_select_periods
from mden_battery.mamba import NativeMambaBlock, selective_scan_native
from mden_battery.model import MDEN, MDENConfig, JointPredictionHead
from mden_battery.loss import MDENJointLoss
from mden_battery.runtime import configure_precision, train_epoch_amp


def raw_frame(n=120, interval=30, shift=0):
    t = np.arange(n)*interval
    capacity = np.full(n, np.nan); capacity[0]=3; capacity[-1]=2.9
    return pl.DataFrame(dict(timestamp_s=t.astype(float), EFC=t/10000,
        v_raw_V=3+0.5*np.sin(t/600)+shift, i_raw_A=np.cos(t/500),
        t_cell_degC=25+np.sin(t/900), soc_est=50+30*np.sin(t/600), cap_aged_est_Ah=capacity))


def prep_fixture(tmp_path, cells=10):
    raw=tmp_path/'raw'; raw.mkdir(exist_ok=True)
    for i in range(cells):
        raw_frame(120,30, i*.01).write_csv(raw/f'cell_log_age_30s_P{i:03d}_1_S01_C01.csv',separator=';')
    return PrepConfig(raw, tmp_path/'prepared', cell_limit=cells, show_progress=False, chunk_rows=17)


def test_resample_rejects_internal_gap_invalid_row_and_partial_bin(tmp_path):
    cfg=PrepConfig(tmp_path,tmp_path,cell_limit=3);frame=raw_frame(45,2)
    good=resample_query(frame.lazy(),2,cfg).collect()
    assert good['_valid'].to_list()==[True,True,True]
    np.testing.assert_allclose(good['timestamp_s'],[30,60,90])
    assert good['soc_est'][0] == pytest.approx(frame['soc_est'][:15].mean())
    bad=resample_query(frame.filter(pl.col('timestamp_s')!=10).lazy(),2,cfg).collect()
    assert bad['_valid'].to_list()==[False,True,True]
    invalid=frame.with_columns(pl.when(pl.col('timestamp_s')==40).then(None).otherwise(pl.col('i_raw_A')).alias('i_raw_A'))
    assert resample_query(invalid.lazy(),2,cfg).collect()['_valid'].to_list()==[True,False,True]
    assert not resample_query(frame.head(44).lazy(),2,cfg).collect()['_valid'][-1]


def test_capacity_changes_while_efc_constant_and_does_not_extrapolate(tmp_path):
    cfg=PrepConfig(tmp_path,tmp_path,cell_limit=3)
    frame=resample_query(raw_frame().with_columns(pl.lit(0.).alias('EFC')).lazy(),30,cfg).collect()
    values, offsets, dropped=prepared_values(frame,(np.array([330.,3030.]),np.array([3.,2.7])),cfg)
    assert values[0,1]==330 and values[-1,1]==3030
    assert values[0,6] == pytest.approx(1) and values[-1,6]==pytest.approx(.9)
    assert np.ptp(values[:,6]) > .09 and dropped > 0


def test_resampling_cannot_hide_reversed_source_bins(tmp_path):
    cfg = PrepConfig(tmp_path, tmp_path, cell_limit=3)
    frame = raw_frame(900, 2)
    reversed_bins = pl.concat([frame.slice(15, 15), frame.head(15), frame.slice(30)])
    resampled = resample_query(reversed_bins.lazy(), 2, cfg).collect()
    with pytest.raises(ValueError, match='Non-increasing timestamps'):
        prepared_values(resampled, (np.array([2., 1800.]), np.array([3., 2.9])), cfg)


@pytest.mark.parametrize('n,expected',[(10,[6,2,2]),(30,[18,6,6]),(50,[30,10,10])])
def test_dynamic_disjoint_splits(tmp_path,n,expected):
    splits=split_cells([str(i) for i in range(n)],PrepConfig(tmp_path,tmp_path,cell_limit=n))
    assert [list(splits.values()).count(k) for k in ('train','val','test')]==expected


def test_etl_cache_train_scaler_and_batch_window_boundaries(tmp_path):
    cfg=prep_fixture(tmp_path);generation,reused=prepare_dataset(cfg);assert not reused
    same,reused=prepare_dataset(cfg);assert reused and same==generation
    index=validate_prepared_dataset(generation,cfg)
    features=np.concatenate([np.load(generation/r['path']) for r in index.to_dicts() if r['split']=='train'])[:,:5]
    np.testing.assert_allclose(features.mean(0),np.zeros(5),atol=2e-6)
    np.testing.assert_allclose(features.std(0),np.ones(5),atol=2e-6)
    dataset=WindowBatchDataset(generation,split='train',config=WindowConfig(),batch_size=17,shuffle=True)
    batch,meta=dataset.fetch([0,dataset.total_windows-1],metadata=True)
    for i in range(2):
        row=dataset.index.row(int(meta['source'][i]),named=True);array=np.load(generation/row['path']);start=meta['start'][i]
        np.testing.assert_array_equal(batch['x'][i],array[start:start+32,:5])
        np.testing.assert_array_equal(batch['y_soc'][i],array[start+32:start+40,5])
        np.testing.assert_array_equal(batch['y_soh'][i],array[start+32:start+40,6])
    assert sum(len(b['x']) for b in dataset)==dataset.total_windows
    assert dataset.epoch==0
    assert not Path(index.row(0,named=True)['path']).is_absolute()
    # A corrupt cached scaler must fail before it can be used for inference.
    import json
    scaler_path = generation / 'scaler.json'
    scaler = json.loads(scaler_path.read_text())
    scaler['std'][0] = float('inf')
    scaler_path.write_text(json.dumps(scaler))
    with pytest.raises(ValueError, match='Invalid train scaler'):
        validate_prepared_dataset(generation, cfg)


def test_fft_and_model_inference_do_not_depend_on_batch():
    torch.manual_seed(7);t=torch.arange(32).float()
    x=torch.stack([torch.sin(t*2*torch.pi/32),40*torch.sin(t*12*torch.pi/32)])[:,:,None].repeat(1,1,5)
    assert fft_select_periods(x,1).frequency_indices.tolist()==[[1],[6]]
    model=MDEN(MDENConfig(fem_hidden_channels=8,fem_groups=2,top_k=2,mamba_d_state=4)).eval()
    with torch.no_grad(): together=model(x);alone=model(x[:1])
    for task in ['soc','soh']:
        torch.testing.assert_close(together[task][:1],alone[task],atol=2e-6,rtol=2e-5)


def test_fem_has_no_unpublished_residual_by_default():
    layer=FeatureExtractionModule(5,top_k=2,hidden_channels=8,groups=2)
    for p in layer.parameters(): torch.nn.init.zeros_(p)
    x=torch.randn(3,32,5);torch.testing.assert_close(layer(x),torch.zeros_like(x))


def test_parallel_scan_equals_serial_outputs_and_gradients():
    torch.manual_seed(3);shape=(2,7,3)
    values=[torch.randn(*shape,dtype=torch.double),torch.rand(*shape,dtype=torch.double)*.1,
            -torch.rand(3,4,dtype=torch.double),torch.randn(2,7,4,dtype=torch.double),
            torch.randn(2,7,4,dtype=torch.double),torch.randn(3,dtype=torch.double)]
    a=[v.clone().requires_grad_() for v in values];b=[v.clone().requires_grad_() for v in values]
    serial=selective_scan_native(*a,mode='serial');parallel=selective_scan_native(*b,mode='parallel')
    torch.testing.assert_close(serial,parallel,atol=1e-11,rtol=1e-10)
    serial.square().sum().backward();parallel.square().sum().backward()
    for x,y in zip(a,b): torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-9)


def test_figure4_norms_and_no_residual_and_dt_initialization():
    block=NativeMambaBlock(5);dt=torch.nn.functional.softplus(block.dt_proj.bias)
    assert dt.min()>=.001 and dt.max()<=.1
    assert hasattr(block,'gate_norm') and hasattr(block,'output_norm')
    with torch.no_grad(): block.out_proj.weight.zero_()
    torch.testing.assert_close(block(torch.randn(2,32,5)),torch.zeros(2,32,5))


def test_optimized_head_equals_full_time_head():
    head=JointPredictionHead(5,8,32);soc,soh=torch.randn(3,32,5),torch.randn(3,32,5)
    old=head.net(torch.cat([soc,soh],-1).transpose(1,2))[:,:,-1]
    torch.testing.assert_close(old,torch.cat(head(soc,soh),1))


def test_loss_formula_fp32_and_gradient_to_features():
    loss=MDENJointLoss(5)
    with torch.no_grad(): loss.soc_uncertainty.linear.weight.fill_(.2)
    features=torch.randn(3,32,5,requires_grad=True)
    pred={'soc':torch.rand(3,8),'soh':torch.rand(3,8),'soc_fused':features,'soh_fused':features}
    y_soc,y_soh=torch.rand(3,8),torch.rand(3,8);result=loss(pred,y_soc,y_soh)
    s=loss.soc_uncertainty(features);h=loss.soh_uncertainty(features)
    expected=(((pred['soc']-y_soc)**2).mean(1)/(2*s*s)+((pred['soh']-y_soh)**2).mean(1)/(2*h*h)+s.log1p()+h.log1p()).mean()
    torch.testing.assert_close(result['loss'],expected)
    result['loss'].backward();assert features.grad.abs().sum()>0 and result['loss'].dtype==torch.float32


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3));self.config=SimpleNamespace(horizon=1)
    def forward(self,x):
        pred=self.weight*x[:,0,:1];return {'soc':pred,'soh':pred}


class TinyLoss(torch.nn.Module):
    def forward(self,p,s,h):
        a=(p['soc']-s).square().mean();b=(p['soh']-h).square().mean()
        return {'loss':a+b,'mse_soc':a,'mse_soh':b}


def test_unequal_microbatches_equal_single_batch_gradient():
    one=TinyModel();two=copy.deepcopy(one);loss=TinyLoss()
    opt1=torch.optim.SGD(one.parameters(),lr=.01);opt2=torch.optim.SGD(two.parameters(),lr=.01)
    batch={'x':torch.arange(5.).reshape(5,1,1),'y_soc':torch.ones(5,1),'y_soh':torch.zeros(5,1)}
    micros=[{k:v[:4] for k,v in batch.items()},{k:v[4:] for k,v in batch.items()}]
    precision=configure_precision('cpu')
    train_epoch_amp(one,loss,[batch],opt1,precision.scaler(),precision,gradient_clip_norm=100)
    train_epoch_amp(two,loss,micros,opt2,precision.scaler(),precision,accumulation_steps=3,gradient_clip_norm=100)
    torch.testing.assert_close(one.weight,two.weight)


def test_grouped_fem_equals_independent_branches_with_duplicate_periods():
    torch.manual_seed(11)
    layer=FeatureExtractionModule(5,top_k=16,hidden_channels=8,groups=2,conv_rounds=1)
    x=torch.randn(2,32,5,requires_grad=True)
    y=layer(x);selection=fft_select_periods(x,16);weights=selection.scores.softmax(-1)
    naive=[]
    for sample in range(2):
        out=0
        for j in range(16):
            period=int(selection.periods[sample,j])
            branch=layer._to_2d(x[sample:sample+1],period)[0]
            out=out+layer._to_1d(layer.rounds(branch),32)*weights[sample,j]
        naive.append(out)
    reference=torch.cat(naive)
    torch.testing.assert_close(y,reference,atol=3e-7,rtol=1e-5)
    first=torch.autograd.grad(y.sum(),x,retain_graph=True)[0]
    second=torch.autograd.grad(reference.sum(),x)[0]
    torch.testing.assert_close(first,second,atol=3e-7,rtol=1e-5)


def test_checkpoint_keeps_best_per_task_and_roundtrips_prediction_and_rng(tmp_path):
    import json
    from mden_battery.checkpoints import new_training_state, complete_epoch, save_epoch, load_inference, resume_training
    from mden_battery.runtime import build_optimizer
    cfg=prep_fixture(tmp_path);generation,_=prepare_dataset(cfg)
    config=MDENConfig(fem_hidden_channels=8,fem_groups=2,top_k=1,soc_depth=1,shared_depth=1,soh_depth=1,conv_rounds=1)
    model=MDEN(config).eval();criterion=MDENJointLoss(5)
    precision=configure_precision('cpu');optimizer=build_optimizer(model,criterion)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer);scaler=precision.scaler()
    state=new_training_state()
    metrics={'loss':1.,'rmse_soc':.02,'rmse_soh':.08,'balanced_rmse':.06}
    improved=complete_epoch(state,metrics,metrics,epoch=1,learning_rate=.001,seconds=1,monitor='loss')
    effective=dict(monitor='loss',batch_size=8,accumulation_steps=1,seed=42)
    run=tmp_path/'run'
    save_epoch(run,model,criterion,optimizer,scheduler,scaler,state,config=effective,generation=generation,improved=improved)
    expected_rng=torch.rand(4)
    x=torch.randn(2,32,5)
    loaded,loaded_loss,payload=load_inference(run/'best_soc.pt')
    with torch.no_grad():
        torch.testing.assert_close(model(x)['soc'],loaded(x)['soc'])
    resumed=resume_training(run/'last.pt',model,criterion,optimizer,scheduler,scaler,effective,generation)
    torch.testing.assert_close(expected_rng,torch.rand(4))
    with pytest.raises(ValueError, match='Resume mismatch: amp'):
        resume_training(run/'last.pt',model,criterion,optimizer,scheduler,scaler,
                        dict(effective, amp='torch.float16'),generation)
    metrics2={'loss':.8,'rmse_soc':.04,'rmse_soh':.03,'balanced_rmse':.035}
    improved=complete_epoch(state,metrics2,metrics2,epoch=2,learning_rate=.001,seconds=1,monitor='loss')
    assert 'soc' not in improved and state['best_epoch']['soc']==1 and 'joint' in improved


def test_article_does_not_mix_cells_or_bridge_nan():
    import pandas as pd
    from mden_battery.data.article_preprocessing import compute_article_labels, SlidingWindowDataset
    frames=[]
    for cell in ('a','b'):
        n=90
        frames.append(pd.DataFrame(dict(cell_id=cell,cycle=np.ones(n),time_s=np.arange(n)*30.,
                     current_A=np.ones(n)*(1 if cell=='a' else 2),voltage_V=np.ones(n)*3.5,temperature_C=np.ones(n)*25)))
    labels=compute_article_labels(pd.concat(frames),rated_capacity_ah=3)
    assert labels.groupby('cell_id').first()['soc'].tolist()==[0,0]
    labels.loc[45,'voltage_V']=np.nan
    ds=SlidingWindowDataset(labels,['cycle','time_s','voltage_V','current_A','temperature_C'])
    assert all(torch.isfinite(ds[i]['x']).all() for i in range(len(ds)))
    assert len(ds)==9  # a: 2 valid windows, b: 7; no compression across the NaN


@pytest.mark.parametrize('shape',[(2,5,1,32),(2,5,4,8),(2,5,6,6)])
def test_linear_conv_fusion_preserves_padding_bias_outputs_and_all_gradients(shape):
    from mden_battery.feature_extraction import ConvRound
    torch.manual_seed(23)
    fused=ConvRound(5,hidden_channels=8,groups=2).double()
    ref=copy.deepcopy(fused);ref.fuse_linear=False
    x=torch.randn(*shape,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    a=fused(x);b=ref(y)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)
    a.square().sum().backward();b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-11,rtol=1e-10)
    for p,q in zip(fused.parameters(),ref.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-9)
