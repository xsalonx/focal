from argparse import ArgumentParser

import numpy as np
import os
from pathlib import Path
from tqdm.auto import trange, tqdm

import torch
import torch.nn.functional as F

from focal.utils.metrics import Metrics
from focal.utils.train import get_last_epoch, get_all_saved_epochs
from utilss.image_analyzis.energy_characteristics import get_image_from_characteristics

from focal.models.approx.models.model1 import ParticleShowerNet as model1, get_optimizer as get_optimizer1
from focal.models.approx.models.model2 import ParticleShowerNet as model2
from focal.models.approx.models.model3 import ParticleShowerNet as Model3
from focal.models.approx.models.model4 import ParticleShowerNet as model4
from focal.models.approx.models.model5 import ParticleShowerNet as model5

def get_model_and_optimizer(name):
    models_lookup = {
        1: (model1, get_optimizer1),
        2: (model2, get_optimizer1),
        3: (Model3, get_optimizer1),
        4: (model4, get_optimizer1),
        5: (model5, get_optimizer1),
    }

    return models_lookup[name]


from focal.models.approx.enhancer.model1 import UnetEnhancer as EnhancerModel1
from focal.models.approx.enhancer.model2 import UNetEnhancer as EnhancerModel2

def get_enhancer_model_and_optimizer(name):
    models_lookup = {
        1: (EnhancerModel1, get_optimizer1), # sic
        2: (EnhancerModel2, get_optimizer1),
    }

    return models_lookup[name]


from focal.models.approx.utils import *

from pprint import pprint
from tqdm import tqdm

def build_parser():
    args = ArgumentParser()

    args.add_argument("--name", "-n", required=True, type=str)
    args.add_argument("--model-id", "--mid", required=True, type=int)
    args.add_argument("--latent-dim", type=int, default=16)
    args.add_argument("--learning-rate", "--lr", type=float)
    args.add_argument("--train-batch-size", "--tbs", required=True, type=int)


    args.add_argument("--enhancer-name", "-en", required=True, type=str)
    args.add_argument("--enhancer-model-id", "--emid", required=True, type=int)
    args.add_argument("--enhancer-learning-rate", "--elr", type=float)
    args.add_argument("--enhancer-train-batch-size", "--etbs", required=True, type=int)
    args.add_argument("--enhancer-base-channel", "--ebc", default=16, required=False, type=int)

    args.add_argument("--eval-batch-size", "--ebs", default=1024, required=False, type=int)
    args.add_argument("--test-batch-size", "--tsbs", default=1024, required=False, type=int)

    args.add_argument("--epochs", "--ep", "-e", required=False, default=500, type=int)

    args.add_argument("--data-dir", "--dd", type=str)
    args.add_argument("--trainings-dir", "--td", type=str)

    args.add_argument("--early-loaders-to-cuda", "--elcuda", action="store_true", default=False)

    args.add_argument("--subset-percentage", "--subpct", type=float)
    args.add_argument("--samples-no", default=32, type=int)

    args.add_argument("--no-wandb", "--nowb", action="store_true", default=False)
    args.add_argument("--group", "--gr", required=False, type=str)

    args.add_argument("--n-rep-test", "--nrt", "-r", default=1, type=int)

    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--val-freq", type=int, default=2, help="Validate every N epochs")

    args.add_argument("--no-resume", default=False, action="store_true")
    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e
from collections import defaultdict

def approx_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['model_id']}-{kwargs['latent_dim']}-{kwargs['train_batch_size']}-{sci_e(kwargs['learning_rate'])}"

def enhancer_extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['enhancer_name']}-{kwargs['enhancer_model_id']}-{kwargs['enhancer_base_channel']}-{kwargs['enhancer_train_batch_size']}-{sci_e(kwargs['enhancer_learning_rate'])}"




from focal.utils.eval import eval_step
from focal.utils.data import load_data_scope

if __name__ == '__main__':
    args = build_parser().parse_args()
    pprint(vars(args))
    enhancer_name = enhancer_extend_name_with_params(**vars(args))
    print(enhancer_name)

    # 1️⃣  Data ----------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_device = device if args.early_loaders_to_cuda else 'cpu'
    
    generator=torch.manual_seed(42)


    data_scope_args = {**vars(args)}
    data_scope_args['train_batch_size'] = args.enhancer_train_batch_size
    data_scope = load_data_scope(
        device=data_device,
        **data_scope_args,
        data_load_config=[
            ['particles', 'standard'],
            ['responses', '1log+1', lambda p: p[:, None]],
            ['characteristics', '1log+1'],
            ['centroids', 'none'],
        ]
    )


    enhancer_checkpoints_dir = Path(args.trainings_dir).resolve() / enhancer_name / 'checkpoints'
    os.makedirs(enhancer_checkpoints_dir, exist_ok=True)
    
# Loading approx
    apporx_name = approx_extend_name_with_params(**vars(args))
    print(apporx_name)
    approx_checkpoints_dir = Path(args.trainings_dir).resolve() / apporx_name / 'checkpoints'
    approx_model_class, _ = get_model_and_optimizer(args.model_id)
    margs = { 'latent_dim': args.latent_dim } if args.model_id >= 3 else {}
    approx_model = approx_model_class(**margs).to(device)
    apporx_last_epoch = get_last_epoch(approx_checkpoints_dir)
    load_approx_model(approx_checkpoints_dir, approx_model, apporx_last_epoch)
    for p in approx_model.parameters():
        p.requires_grad_(False)

# Loading Enhancer
    EnhancerClass, get_enhancer_opt = get_enhancer_model_and_optimizer(args.enhancer_model_id)
    enhancer_model = EnhancerClass(base_ch=args.enhancer_base_channel).to(device)
    optimizer = get_enhancer_opt(enhancer_model, args.enhancer_learning_rate)

    if not args.no_resume:
        last_epoch = get_last_epoch(enhancer_checkpoints_dir)
        if last_epoch is not None:
            load_approx_model(enhancer_checkpoints_dir, enhancer_model, last_epoch)
            print(f"Loaded enhancer model from epoch {last_epoch}")
        else:
            last_epoch = 0

    config = vars(args)
    config['model_type'] = str(type(EnhancerClass))
    config['model_family'] = 'enhancer'
    config['approx_name'] = apporx_name
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(enhancer_model)
    
    metrics = Metrics(
        job_type='train',
        name=enhancer_name,
        trainings_dir=args.trainings_dir,
        use_wandb=not args.no_wandb,
        config=config)


    sample_particles, sample_responses, _, sample_centroids = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)
    sample_centroids = torch.Tensor(sample_centroids).to(device)
    characterisitc_transform = data_scope['scalers'][2]

    def approx_model_forward_wrp(cond):
        if args.model_id >= 3:
            return approx_model(
                torch.randn(cond.size(0), args.latent_dim, device=device),
                cond.to(device)
            )
        else:
            return approx_model(cond.to(device))

    def pipeline(cond, _r, _c, centroids):
        with torch.no_grad():
            approx_responses = characterisitc_transform.transform(np.stack([
                get_image_from_characteristics(centroid, characterisitc_transform.inverse_transform(characteristic))
                for centroid, characteristic
                in tqdm(zip(centroids, approx_model_forward_wrp(cond).squeeze().cpu().numpy()))
            ])[:, None])
            print(approx_responses.shape)
        return enhancer_model(torch.Tensor(approx_responses).to(device))
    
    def eval_pipeline(*batch):
        return pipeline(*batch).detach().cpu().numpy().transpose(0, 2, 3, 1)
    

    if not args.ftest:
        approx_model.eval()
        for epoch in trange(last_epoch, args.epochs, desc=f'Epochs {args.name}'):
            enhancer_model.train()
            
            for batch_idx, batch in enumerate(tqdm(data_scope['dataloaders']['train'], desc="Train loader")):
                _cond, responses, *_ = batch

                optimizer.zero_grad()

                enhanced_approx_responses = pipeline(*batch)

                loss = F.mse_loss(enhanced_approx_responses, responses.to(device))
                loss.backward()
                optimizer.step()

                metrics.add({ 'mse_loss': loss.detach().item()}, 'train')

            metrics.log(epoch)


            enhancer_model.eval()
            with torch.no_grad():
                metrics.current_epoch = epoch
                eval_step(
                    eval_pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'validation',
                    metrics
                )

                generated_samples = eval_pipeline(sample_particles, None, None, sample_centroids)
                metrics.plot_responses(
                        [sample_responses, generated_samples],
                        epoch,
                        labels=["Original", "Generated"],
                    )

                
            save_approx_model(enhancer_checkpoints_dir, enhancer_model, epoch)
            metrics.log(epoch)            


        print(f'Testing after {args.epochs} epochs of trainig')
        with torch.no_grad():
            metrics.current_epoch = epoch
            eval_step(eval_pipeline, data_scope['dataloaders']['test'], data_scope, 'test', metrics)
            metrics.log(args.epochs)
            print(f'Finished training of {enhancer_name}')

        print("Training finished ✔️")

    else:
        enhancer_model.eval()
        from focal.utils.train import find_closest_epoch_file 
        saved_epochs = [find_closest_epoch_file(enhancer_checkpoints_dir, args.ftest)]
        for epoch in tqdm(saved_epochs, desc=f'Epochs {enhancer_name}'):
            load_approx_model(enhancer_checkpoints_dir, enhancer_model, epoch=epoch)
            with torch.no_grad():
                metrics.current_epoch = epoch
                eval_step(
                    eval_pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'ftest',
                    metrics
                )
            metrics.log(epoch, logAsEpoch=True)

