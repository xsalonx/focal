from argparse import ArgumentParser

import numpy as np
import os
from pathlib import Path
from tqdm.auto import trange, tqdm

import torch
import torch.nn.functional as F

from focal.utils.metrics import Metrics
from focal.utils.train import get_last_epoch
from utilss.image_analyzis.energy_characteristics import get_image_from_characteristics

from focal.models.approx.models.model1 import ParticleShowerNet as model1, get_optimizer as get_optimizer1
from focal.models.approx.models.model2 import ParticleShowerNet as model2
from focal.models.approx.models.model3 import ParticleShowerNet as model3
from focal.models.approx.models.model4 import ParticleShowerNet as model4
from focal.models.approx.models.model5 import ParticleShowerNet as model5

def get_model_and_optimizer(name):
    models_lookup = {
        1: (model1, get_optimizer1),
        2: (model2, get_optimizer1),
        3: (model3, get_optimizer1),
        4: (model4, get_optimizer1),
        5: (model5, get_optimizer1),
    }

    return models_lookup[name]


from focal.models.approx.utils import *

from pprint import pprint
from tqdm import tqdm

def build_parser():
    args = ArgumentParser()

    args.add_argument("--name", "-n", required=True, type=str)
    args.add_argument("--model-id", "--mid", required=True, type=int)
    args.add_argument("--learning-rate", "--lr", type=float)
    args.add_argument("--latent_dim", type=int, default=16)

    args.add_argument("--train-batch-size", "--tbs", required=True, type=int)
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
    args.add_argument("--responses-scaler", "--rssc", default="1log+1", type=str)

    args.add_argument("--ftest", default=None, type=int)

    return args


from focal.utils.formats import sci_e
from collections import defaultdict

def extend_name_with_params(**kwargs):
    def keep_nonone(d: dict) -> dict:
        return { k: v for k, v in d.items() if v is not None }
    kwargs = defaultdict(lambda: '', keep_nonone(kwargs))
    return f"{kwargs['name']}-{kwargs['model_id']}-{kwargs['latent_dim']}-{kwargs['train_batch_size']}-{sci_e(kwargs['learning_rate'])}"



from focal.utils.eval import eval_step
from focal.utils.data import load_data_scope

def KL_loss_function(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

if __name__ == '__main__':
    args = build_parser().parse_args()
    pprint(vars(args))
    name = extend_name_with_params(**vars(args))
    print(name)

    # 1️⃣  Data ----------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_device = device if args.early_loaders_to_cuda else 'cpu'
    
    generator=torch.manual_seed(42)

    data_scope = load_data_scope(
        device=data_device,
        **vars(args),
        data_load_config=[
            ['particles', 'standard'],
            ['responses', args.responses_scaler, lambda p: p[:, None]],
            ['characteristics', args.responses_scaler],
            ['centroids', 'none'],
        ]
    )


    checkpoints_dir = Path(args.trainings_dir).resolve() / name / 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    model_class, get_optimizer = get_model_and_optimizer(args.model_id)

    margs = { 'latent_dim': args.latent_dim } if args.model_id >= 3 else {}
    model = model_class(**margs).to(device)
    optimizer = get_optimizer(model, args.learning_rate)

    config = vars(args)
    config['model_type'] = str(type(model))
    config['model_family'] = 'approx'
    from focal.utils.train import get_num_parameters
    config['model_num_parameters'] = get_num_parameters(model)
    
    metrics = Metrics(
        job_type='train',
        name=name,
        trainings_dir=args.trainings_dir,
        use_wandb=not args.no_wandb,
        config=config)

    if not args.no_resume:
        last_epoch = get_last_epoch(checkpoints_dir)
        if last_epoch is not None:
            load_approx_model(checkpoints_dir, model, last_epoch)
            print(f"Loaded approx model from epoch {last_epoch}")
        else:
            last_epoch = 0


    sample_particles, sample_responses, _, sample_centroids = data_scope['samples']
    sample_particles = torch.Tensor(sample_particles).to(device)
    sample_responses = sample_responses.transpose(0, 2, 3, 1)
    sample_centroids = torch.Tensor(sample_centroids).to(device)
    characterisitc_transform = data_scope['scalers'][2]

    def model_forward_wrp(cond):
        if args.model_id >= 3:
            return model(
                torch.randn(cond.size(0), args.latent_dim, device=device),
                cond.to(device)
            )
        else:
            return model(cond.to(device))

    def pipeline(cond, _r, _c, centroids):
        return characterisitc_transform.transform(np.stack([
            get_image_from_characteristics(centroid, characterisitc_transform.inverse_transform(characteristic))
            for centroid, characteristic
            in tqdm(zip(centroids, model_forward_wrp(cond).squeeze().cpu().numpy()))
        ])[:, None].transpose(0, 2, 3, 1))
                

    

    if not args.ftest:

        for epoch in trange(last_epoch, args.epochs, desc=f'Epochs {args.name}'):
            model.train()
            
            for batch_idx, batch in enumerate(tqdm(data_scope['dataloaders']['train'], desc="Train loader")):
                cond, _r, characteristics, *_ = batch

                cond = cond.to(device)
                characteristics = characteristics.to(device)
                optimizer.zero_grad()

                pred_characteristics = model_forward_wrp(cond)
                if isinstance(pred_characteristics, tuple):
                    pred_characteristics, mu, logvar = pred_characteristics
                    kl_loss = KL_loss_function(mu, logvar)
                else:
                    kl_loss = 0

                loss = F.mse_loss(pred_characteristics, characteristics) + kl_loss
                loss.backward()
                optimizer.step()

                metrics.add({ 'mse_loss': loss.detach().item()}, 'train')

            metrics.log(epoch)


            model.eval()
            with torch.no_grad():
                metrics.current_epoch = epoch
                eval_step(
                    pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'validation',
                    metrics,
                )

                generated_samples = pipeline(sample_particles, None, None, sample_centroids)
                metrics.plot_responses(
                        [sample_responses, generated_samples],
                        epoch,
                        labels=["Original", "Generated"],
                    )

            
            save_approx_model(checkpoints_dir, model, epoch)
            metrics.log(epoch)            

        print(f'Testing after {args.epochs} epochs of trainig')
        with torch.no_grad():
            metrics.current_epoch = epoch
            eval_step(pipeline, data_scope['dataloaders']['test'], data_scope, 'test', metrics)
            metrics.log(args.epochs)
            print(f'Finished training of {name}')

        print("Training finished ✔️")

    else:
        from focal.utils.train import get_all_saved_epochs

        from focal.utils.train import find_closest_epoch_file 
        saved_epochs = [find_closest_epoch_file(checkpoints_dir, args.ftest)]
        model.eval()
        with torch.no_grad():
            for epoch in tqdm(saved_epochs, desc=f'Epochs {name}'):
                load_approx_model(checkpoints_dir, model, epoch)
                metrics.current_epoch = epoch
                eval_step(
                    pipeline,
                    data_scope['dataloaders']['test'],
                    data_scope,
                    'ftest',
                    metrics
                )
                metrics.log(epoch, logAsEpoch=True)
            