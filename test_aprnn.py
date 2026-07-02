import gc
import argparse
import subprocess
import sys
from pathlib import Path
from repair.args import RepairTask

script = Path(__file__).parent / "APRNN" / "eval_7_aprnn.py"

net_map = {
    0: "mnist 9x100",
    1: "mnist_256x4",
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

parser = argparse.ArgumentParser(description="Run APRNN eval_7 across a range of ndims.")
parser.add_argument(
    "--net-id",
    type=int,
    default=0,
    choices=sorted(net_map.keys()),
    help="Key into net_map selecting which network to evaluate.",
)
args = parser.parse_args()

ndims_list = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

for ndims in ndims_list:
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "MNIST",
        "--net", net_map[args.net_id],
        "--repair_task", repair_task_map[repair_task_id].value,
        "--num_runs", "20",
        "--eps", "1",  # value/255
        "--pick", "nonzero",  # 'all' or 'nonzero'
        "--ndims", str(ndims),
        "--device", "cpu",
    ]

    subprocess.run(cmd, check=True)
    gc.collect()
