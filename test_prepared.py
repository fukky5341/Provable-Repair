import gc
import argparse
import time
from PREPARED.repair import repair_prepared
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairMode, RepairTask


# experiment map (model, dataset, perturbation_pick, repair_task, num_runs)
exe_map = {
    0: ( "mnist 9x100", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20 ),
    1: ( "mnist 9x100", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20 ),
    2: ( "mnist_256x4", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20 ),
    3: ( "mnist_256x4", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20 ),
    4: ( "cifar10_cnn1", Dataset.CIFAR10, 'all', RepairTask.CorruptionAndPerturbation, 10 ),
    5: ( "gtsrb_cnn", Dataset.GTSRB, 'all', RepairTask.AdversarialAndPerturbation, 10 ),
}

# ndims_list
ndims_map = {
    0: [5, 6, 7, 8, 9, 10, 12, 14],
    1: [10],
    2: [5, 6, 7, 8, 9, 10, 12, 14],
    3: [10],
    4: [10],
    5: [10]
}

# N_list
N_list_map = {
    0: [1],
    1: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    2: [1],
    3: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    4: [12, 14, 16, 18, 20],
    5: [4, 6, 8, 10, 12]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-id", type=int, default=1, choices=sorted(exe_map.keys()))
    cli_args = parser.parse_args()

    model_name, dataset, pick, repair_task, num_runs = exe_map[cli_args.exe_id]
    num_v_polys_list = N_list_map[cli_args.exe_id] if pick == 'all' else [1]
    perturbation_ndim_list = ndims_map[cli_args.exe_id] if pick == 'nonzero' else [10]

    # current time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/test_log_{timestamp}.txt"

    for num_v_polys in num_v_polys_list:
        for perturbation_ndim in perturbation_ndim_list:
            args = RepairArgs(
                repair_task=repair_task,
                model_name=model_name,
                dataset=dataset,
                logfile=logfile,
            )

            # repair settings
            args.repair_mode = RepairMode.PREPARED
            args.target_label = None
            args.num_runs = num_runs
            args.inp_eps = 1.0
            args.num_v_polys = num_v_polys
            args.perturbation_pick = pick
            args.perturbation_ndim = perturbation_ndim
            args.modifiable_range = 0.5

            # layer to repair
            start_last_layer_idx_map = {
                "mnist 9x100": (12, None),
                "mnist_256x4": (7, None),
                "cifar10_cnn1": (5, None),
                "gtsrb_cnn": (5, None),
            }
            args.repair_start_layer_idx, args.repair_last_layer_idx = \
                start_last_layer_idx_map[model_name]

            repair_prepared(args)
            gc.collect()
