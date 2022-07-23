from .base import Config, Model, ModelFactory, DataFactory, BaseExperiment
import wandb
import os
from shutil import copy2

# The default value is "redirect" which doesn't work well for sweeps
os.environ["WANDB_CONSOLE"] = "wrap"


class WandbExperiment(BaseExperiment):
    def __init__(
        self, config: Config, modelFactory: ModelFactory,
        dataFactory: DataFactory
    ):
        super().__init__(config, modelFactory, dataFactory)

    def init(self, tags: list, job_type: str):
        wandb.init(save_code=True, tags=tags, job_type=job_type)
        print('wandb_train run %s (%s)' % (wandb.run.name, wandb.run.id))
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/apmp", summary="max")
        wandb.define_metric("per_sample_time", summary="mean")
        wandb.define_metric("test/loss", summary="min")
        wandb.define_metric("test/acc", summary="max")
        wandb.define_metric("per_sample_time", summary="mean")

        # Priority for wandb.config because it could come from a sweep
        wandb.config.update(self.config)
        self.config = wandb.config

    def train(self):
        self.init(['train'], 'train')
        super().train()

    def test(self):
        self.init(['test'], 'test')
        super().test()

    def visual(self):
        self.init(['visual'], 'visual')
        super().visual()

    def modelFactory(self):
        return modelFactoryWrapper(super().modelFactory())

    def resume(self, model: Model) -> int:
        initial_epoch = 0
        if wandb.run.resumed:
            print('Resuming from checkpoint')
            try:
                wandb.restore(self.config.checkpoint_path)
            except ValueError as e:
                print('Failed to restore checkpoint with error: %s' % e)
                print('Experiment will start from epoch 0')
            else:
                checkpoint = load(self.config.checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                initial_epoch = checkpoint['epoch'] + 1
        return initial_epoch

    def logMetrics(self, log: dict):
        wandb.log(log)
        super().logMetrics(log)

    def saveCheckpoint(self, model: Model, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        save(checkpoint, self.config.checkpoint_path)
        wandb.save(self.config.checkpoint_path)

        if (epoch + 1) % self.config.save_every_epochs == 0:
            epoch_checkpoint_path = '%s/%d.pickle' % (
                self.config.epoch_checkpoint_dir, epoch
            )
            copy2(self.config.checkpoint_path, epoch_checkpoint_path)
            wandb.save(epoch_checkpoint_path)

        # If it is a SLURM job, save the SLURM log file
        slurm_log_path = ''
        if os.environ.get('SLURM_ARRAY_JOB_ID'
                         ) and os.environ.get('SLURM_ARRAY_TASK_ID'):
            slurm_log_path = os.path.join(
                os.environ['SLURM_SUBMIT_DIR'], 'slurm-%s_%s.out' % (
                    os.environ.get('SLURM_ARRAY_JOB_ID'),
                    os.environ.get('SLURM_ARRAY_TASK_ID')
                )
            )
        elif os.environ.get('SLURM_JOB_ID'):
            slurm_log_path = os.path.join(
                os.environ['SLURM_SUBMIT_DIR'],
                'slurm-%s.out' % (os.environ.get('SLURM_JOB_ID'))
            )

        if slurm_log_path and os.path.exists(slurm_log_path):
            log_path = os.path.join(self.config.logs_dir, 'slurm.log')
            ensure_parent_dir_exists(log_path)
            copy2(slurm_log_path, log_path)
            wandb.save(log_path)

    def saveModel(self, model: Model):
        if self.config.save_onnx:
            model_onnx_path = os.path.join(
                self.config.model_dir, self.config.model_filename_onnx
            )
            ensure_parent_dir_exists(model_onnx_path)
            model.to_onnx(model_onnx_path)
            wandb.save(model_onnx_path)

        model_pt_path = os.path.join(
            self.config.model_dir, self.config.model_filename_pt
        )
        save(model.state_dict(), model_pt_path)
        wandb.save(model_pt_path)

        artifact_name = model.name()
        print('Saving model to artifact: %s' % artifact_name)
        model_artifact = wandb.Artifact(
            artifact_name,
            type="model",
            # description="trained inception v3",
            metadata=self.config.__dict__
        )

        if self.config.save_onnx:
            model_artifact.add_file(model_onnx_path)
        model_artifact.add_file(model_pt_path)
        wandb.log_artifact(model_artifact)

    def loadModel(self, model: Model):
        artifact_name = model.name() + ':latest'
        print('Loading artifact: %s' % artifact_name)
        model_artifact = wandb.use_artifact(artifact_name)
        model_dir = model_artifact.download()
        model_pt_path = os.path.join(model_dir, self.config.model_filename_pt)
        state_dict = load(model_pt_path)
        model.load_state_dict(state_dict)


def modelFactoryWrapper(modelFactory):
    def wrapper(config: Config):
        model = modelFactory(config, )
        # log_freq is 1 because we log metrics once every epoch
        # this could change if we want to log metrics more frequently
        # and don't want to log gradients at the same frequency
        wandb.watch(model.model(), model.criterion(), log="all", log_freq=1)
        return model

    return wrapper


#### Helpers ####
import pickle
from pathlib import Path
import torch
import numpy
import random
import subprocess


def ensure_dir_exists(dir):
    path_to_dir = Path(dir)
    path_to_dir.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir_exists(filename):
    path_to_file = Path(filename)
    parent_directory_of_file = path_to_file.parent
    parent_directory_of_file.mkdir(parents=True, exist_ok=True)


def save(data_dict, filename):
    ensure_parent_dir_exists(filename)
    with open(filename, 'wb') as file_handle:
        torch.save(data_dict, file_handle)


def load(filename):
    with open(filename, 'rb') as handle:
        return torch.load(handle, map_location=torch.device('cpu'))
