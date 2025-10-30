from collections import defaultdict
from pprint import pprint
import typing
import math

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from utilss.plot import plot_responses_rows
import os


SLURM_VAR_NAMES = [
    'SLURM_STEP_NUM_TASKS',
    'SLURM_TASKS_PER_NODE',
    'SLURM_JOB_UID',
    'SLURM_JOB_START_TIME',
    'SLURM_CLUSTER_NAME',
    'SLURM_JOB_END_TIME',
    'SLURM_CPUS_ON_NODE',
    'SLURM_JOB_CPUS_PER_NODE',
    'SLURM_JOB_NUM_NODES',
    'SLURM_JOBID',
    'SLURM_NTASKS',
    'SLURM_MEM_PER_NODE',
    'SLURM_NPROCS',
    'SLURM_NNODES',
    'SLURM_JOB_ID',
    ]

def keep_numeric_values(d: dict) -> dict:
    """Return a copy of d with only keys whose values are numeric (int/float), excluding bools."""
    return { k: v for k, v in d.items() if isinstance(v, (int, float)) }


SPEC_RUN_TAG_SUFFIX = os.environ['SPEC_RUN_TAG_SUFFIX'] if 'SPEC_RUN_TAG_SUFFIX' in os.environ else ''

class Metrics:
    def __init__(self,
                 job_type,
                 name,
                 trainings_dir: str=None,
                 project='focal-v3',
                 config=None,
                 group=None,
                 use_wandb=True,
                 **kwargs,
    ):
        self.metrics = defaultdict(list)
        self.current_epoch = None
        self.config = None
        self.run_name = name

        self.use_wandb = use_wandb
        self.wandb = wandb

        if config is not None and 'data_dir' in config:
            valid_parts = [p for p in str(config['data_dir']).split('/') if p]
            dataset_variant = '/'.join(valid_parts[-2:])
            dataset_family = '/'.join(valid_parts[-2:-1])
            config['dataset_name'] = dataset_variant
            config['dataset_family'] = dataset_family
            self.config = config

        if use_wandb:
            print("WB init")
            self.run = wandb.init(
                project=project,
                settings=wandb.Settings(console="wrap"),
                name=SPEC_RUN_TAG_SUFFIX + name,
                job_type=job_type,
                dir=trainings_dir,
                id=name,
                group=group,
                config=config,
                resume='allow',
            )

# SLURM env
            for varname in SLURM_VAR_NAMES:
                if varname in os.environ:
                    value = os.environ[varname]

                    job_list = self.run.summary.get(varname, [])
                    if value not in job_list:
                        job_list.append(value)
                        self.run.summary[varname] = job_list

            if 'SLURM_JOB_ID' in os.environ:
                self.run.summary["num_restarts"]  = len(self.run.summary.get('SLURM_JOB_ID', [])) - 1 

    def add(self, metrics, suffix='', section='', no_append=False):
        for name, value in metrics.items():
            if not no_append:
                self.metrics[f"{section + '/' if section else ''}{name}{'_' + suffix if suffix else ''}"].append(value)
            else:
                self.metrics[f"{section + '/' if section else ''}{name}{'_' + suffix if suffix else ''}"] = value


    def log(self, step, logAsEpoch=True, silent=False):
        logs = { metric: jnp.array(values).mean().item() if not isinstance(values, wandb.Image) else values
                   for metric, values in self.metrics.items()}

        if self.use_wandb:
            if logAsEpoch:
                logs['epoch'] = step
                wandb.log(logs)
            else:
                wandb.log(logs, step=step)

        if not silent:
            print(f'Logged metrics of step {step}')
            pprint(logs)

        self.metrics = defaultdict(list)

    def plot_responses(
            self,
            images,
            step=None,
            tag='',
            section='',
            true_coordinates=None,
            found_cordinates=None,
            labels: typing.Sequence[str] | None = None,
        ):

        fig, _axs = plot_responses_rows(
            images=images,
            true_coordinates=true_coordinates,
            found_cordinates=found_cordinates,
            labels=labels,
            clip_negative=True,
        )

        if self.use_wandb:
            self.add({ 'generated': wandb.Image(fig) }, suffix=tag, section=section, no_append=True)
        else:
            plt.show()

        plt.close(fig)

    def plot_energy_ratio_distributions(
            self,
            PRR,
            section='',
            suffix='',
        ):

        P, R, R_gen = PRR
        pmax = P.max()
        fig, axs = plt.subplots(3, 2, figsize=(3 * 2 + 2, 2 * 2 + 2), dpi=200)
        axs[0, 0].scatter(P / pmax, R / R.max(), s=0.1)
        axs[0, 0].set_title('[R vs P] Original scaled individually', fontsize=8)
        axs[0, 1].scatter(P / pmax, R_gen / R_gen.max(), s=0.1)
        axs[0, 1].set_title('[R vs P] Generated scaled individually', fontsize=8)

        axs[1, 0].scatter(P / pmax, R / pmax, s=0.1)
        axs[1, 0].set_title('[R vs P] Original scaled by P.max', fontsize=8)
        axs[1, 1].scatter(P / pmax, R_gen / pmax, s=0.1)
        axs[1, 1].set_title('[R vs P] Generated scaled by P.max', fontsize=8)
        

        axs[2, 0].hist(R, bins=100)
        axs[2, 0].set_title('[R hist] Original responses total energy', fontsize=8)
        axs[2, 1].hist(R_gen, bins=100)
        axs[2, 1].set_title('[R hist] Generated responses total energy', fontsize=8)


        plt.tight_layout()

        if self.use_wandb:
            self.add({ 'energy_ratio_distributions': wandb.Image(fig) }, suffix=suffix, section=section, no_append=True)

        else:
            plt.show()

        plt.close(fig)
