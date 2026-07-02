import gc
import argparse
import subprocess
import sys
from pathlib import Path
from repair.args import RepairTask

script = Path(__file__).parent / "ProRepair" / "repair_rob.py"

exe_map = {
    0: ("mnist 9x100", 'nonzero'),
    1: ("mnist 9x100", 'all'),
    2: ("mnist_256x4", 'nonzero'),
    3: ("mnist_256x4", 'all'),
}

# repair task
repair_task_map = {
    0: RepairTask.LocalRobustness,
}
repair_task_id = 0

'''
max ndims for mnist: 28*28 = 784
max ndims for cifar10: 3*32*32 = 3072
'''

parser = argparse.ArgumentParser(description="Run ProRepair with an experiment setting from exe_map.")
parser.add_argument(
    "--exe-id",
    type=int,
    default=1,
    choices=sorted(exe_map.keys()),
    help="Key into exe_map selecting the model/pick combination.",
)
args = parser.parse_args()

N_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
ndims_list = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

model_name, pick = exe_map[args.exe_id]  # 'all' or 'nonzero'

if pick == 'all':
    N_values = N_list
    ndims_values = [10]
else:
    N_values = [1]
    ndims_values = ndims_list

for N in N_values:
    for ndims in ndims_values:
        cmd = [
            sys.executable,
            str(script),
            "--dataset", "MNIST",
            "--model", model_name,
            "--repair_task", repair_task_map[repair_task_id].value,
            "--num_runs", "20",
            "--N", str(N),
            "--eps", "1",  # value/255
            "--pick", pick,
            "--ndims", str(ndims),
            "--device", "cpu",
            "--dtype", "float32",
        ]

        subprocess.run(cmd, check=True)
        gc.collect()
