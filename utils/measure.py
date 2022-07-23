import time
import torch
import psutil


def MegaBytes(bytes):
    return bytes / (1024.0 * 1024.0)


class _MeasureClass:
    def __init__(self):
        self.enabled = False
        self.last_measure = None

    def __call__(self, name=None):
        if not self.enabled:
            return

        current_time = time.time()
        current_gpu_memory = torch.cuda.memory_allocated(
        ) if torch.cuda.is_available() else 0

        current_memory_info = psutil.Process().memory_info()

        if self.last_measure is not None:
            print(
                '{} took {:.3f} seconds'.format(
                    self.last_measure['name'],
                    current_time - self.last_measure['start_time']
                )
            )
            if torch.cuda.is_available():
                print(
                    '{} consumed {:.2f} MB'.format(
                        self.last_measure['name'],
                        MegaBytes(
                            current_gpu_memory - self.last_measure['gpu_memory']
                        )
                    )
                )
                print(
                    'current gpu memory: {:.2f} MB'.format(
                        MegaBytes(current_gpu_memory)
                    )
                )
            else:
                print(
                    '{} consumed rss= {:.2f} MB vms= {:.2f} MB'.format(
                        self.last_measure['name'],
                        MegaBytes(
                            current_memory_info.rss -
                            self.last_measure['memory_info'].rss
                        ),
                        MegaBytes(
                            current_memory_info.vms -
                            self.last_measure['memory_info'].vms
                        )
                    )
                )
                print(
                    'current rss= {:.2f} MB vms= {:.2f} MB'.format(
                        MegaBytes(current_memory_info.rss),
                        MegaBytes(current_memory_info.vms)
                    )
                )

        if name is not None:
            self.last_measure = {
                'name': name,
                'start_time': time.time(),
                'gpu_memory': current_gpu_memory,
                'memory_info': current_memory_info
            }
        else:
            self.last_measure = None


measure = _MeasureClass()
"""
Use `measure` to measure the time and the memory cosumed between two moments in the execution of the code.

Example::

    measure('cooking')
    cook_food()
    measure('eating')
    eat_food()
    measure()

Output::

    cooking took 1.2 seconds
    cooking consumed 205000 bytes
    current gpu memory: 205000 bytes
    eating took 0.7 seconds
    eating consumed 103000 bytes
    current gpu memory: 308000 bytes
    
"""
