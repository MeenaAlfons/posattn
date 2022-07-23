import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib
import gc
import sys
from pathlib import Path

# Imported for convenience: from my_wandb import Data, Model, Config
from .base import Data, Config, Model

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from utils import measure

from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str


class Model(Model):
    """
    Provides a base model for torch implementations.
    """
    def __init__(self, device, accumulation_steps=1, profile=False):
        self._device = device
        self._accumulation_steps = accumulation_steps
        self._profile = profile

    def model(self):
        pass

    def optimizer(self):
        pass

    def scheduler(self):
        return None

    def criterion(self):
        pass

    def dummy_input(self, batch_size):
        pass

    def name(self):
        pass

    def params_count(self):
        return sum(
            p.numel() for p in self.model().parameters() if p.requires_grad
        )

    def flops_count(self, batch_size=1):
        self.model().to(self._device)
        input = self.dummy_input(batch_size).to(self._device)
        flops = FlopCountAnalysis(self.model(), input)
        print(f"Flops count for batch_size={batch_size}:")
        print(flop_count_str(flops))
        return flops.total()

    def run_scheduler(self, epoch, loss):
        if not self.scheduler():
            return
        scheduler_name = self.scheduler().__class__.__name__
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler().step(loss)
        else:
            self.scheduler().step()

    def train_epoch(self, dataloader, beforeBatch, afterBatch):
        self.model().to(self._device)
        self.model().train()

        running_loss = 0.0
        total = 0
        self.optimizer().zero_grad(set_to_none=True)
        for i, data in enumerate(tqdm(dataloader)):
            beforeBatch({'batch': i})
            inputs, labels = data[0].to(self._device), data[1].to(self._device)
            inputs.requires_grad = False

            measure('forward')

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True
            ) if self._profile else contextlib.nullcontext() as prof:
                outputs = self.model()(inputs)
            report_prof(prof)

            loss = self.criterion()(outputs, labels)
            measure('backward')
            loss.backward(retain_graph=False)
            measure()

            if (i + 1) % self._accumulation_steps == 0:
                self.optimizer().step()
                self.optimizer().zero_grad(set_to_none=True)

            running_loss += loss.detach().item() * labels.size(0)
            total += labels.size(0)

            afterBatch({
                'batch': i,
                'loss': running_loss / total,
            })
            del loss
            del outputs
            del inputs
            del labels
            gc.collect()

        loss = running_loss / total
        return {
            'loss': loss,
        }

    def evaluate(self, dataloader):
        self.model().to(self._device)
        self.model().eval()

        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in tqdm(dataloader):
                inputs, labels = data[0].to(self._device), data[1].to(
                    self._device
                )

                outputs = self.model()(inputs)

                loss = self.criterion()(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        loss = running_loss / total
        acc = correct / total
        return {
            'loss': loss,
            'acc': acc,
        }

    def to_onnx(self, filename='model.onnx'):
        self.model().to('cpu')
        self.model().eval()
        input_names = ["actual_input"]
        output_names = ["output"]

        torch.onnx.export(
            self.model(),
            self.dummy_input(batch_size=2).to('cpu'),
            filename,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=12,
        )

    def state_dict(self):
        self.model().to('cpu')
        self.model().eval()
        return {
            'model_state_dict': self.model().state_dict(),
            'optimizer_state_dict': self.optimizer().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model().load_state_dict(state_dict['model_state_dict'])
        self.optimizer().load_state_dict(state_dict['optimizer_state_dict'])


def report_prof(prof):
    if prof is not None:
        print("Profile sort_by= self_cpu_memory_usage")
        print(
            prof.key_averages().table(
                sort_by="self_cpu_memory_usage", row_limit=10
            )
        )

        print("Profile sort_by= cpu_memory_usage")
        print(
            prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)
        )

        print("Profile sort_by= self_cuda_memory_usage")
        print(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=10
            )
        )

        print("Profile sort_by= cuda_memory_usage")
        print(
            prof.key_averages().table(
                sort_by="cuda_memory_usage", row_limit=10
            )
        )
