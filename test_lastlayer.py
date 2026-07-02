import gc
import argparse
import time
from LastLayerRepair.repair import repair_last_layer
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

    for vpolys_num in num_v_polys_list:
        for perturbation_ndim in perturbation_ndim_list:
            args = RepairArgs(
                repair_task=repair_task,
                model_name=model_name,
                dataset=dataset,
                logfile=logfile,
            )

            # repair settings
            args.repair_mode = RepairMode.LastLayer_ibp
            args.target_label = None
            args.num_runs = num_runs
            args.inp_eps = 1.0
            args.num_v_polys = vpolys_num
            args.perturbation_pick = pick
            args.perturbation_ndim = perturbation_ndim

            # optimization settings
            args.obj_regularization = "l2"  # fixed to l2
            args.lambda_strength_dist = 1000
            args.lambda_strength_rate = 1
            args.lambda_reg = 1.0
            args.modifiable_range = 0.5
            args.use_priority_weight = False

            # maximum number of iterations for the repair optimization
            args.max_repair_steps = 10

            repair_last_layer(args)
            gc.collect()
