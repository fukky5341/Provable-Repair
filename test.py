import argparse
import time
import gc
from PREPARED.repair import repair_prepared
from repair.repair import repair_test
from FL.util import FLmode
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairMode, RepairTask
from LPsolver.solver import ObjectiveType


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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exe-id", type=int, default=1, choices=sorted(exe_map.keys()))

    # repair settings
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--inp_eps", type=float, default=1.0)  # would be divided by 255 for normalization

    # optimization
    parser.add_argument("--lambda_strength_dist", type=float, default=1000)
    parser.add_argument("--lambda_strength_rate", type=float, default=1)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--modifiable_range", type=float, default=0.5)

    # FL
    parser.add_argument("--perform_FL", type=bool, default=False)
    parser.add_argument("--fl_k_ratio", type=float, default=0.01)
    parser.add_argument("--pareto_round", type=int, default=3)

    return parser.parse_args()



# run the repair test
if __name__ == "__main__":

    # current time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/test_log_{timestamp}.txt"

    # cli
    cli_args = parse_args()

    model_name, dataset, pick, repair_task, num_runs = exe_map[cli_args.exe_id]
    num_v_polys_list = N_list_map[cli_args.exe_id] if pick == 'all' else [1]
    perturbation_ndim_list = ndims_map[cli_args.exe_id] if pick == 'nonzero' else [10]

    for vpolys_num in num_v_polys_list:
        for perturbation_ndim in perturbation_ndim_list:

            args = RepairArgs(repair_task=repair_task,
                            model_name=model_name, 
                            dataset=dataset, 
                            logfile=logfile)
            
            args.debug = False

            # repair settings
            args.repair_mode = RepairMode.LP_ibp
            args.target_label = cli_args.target_label
            args.inp_eps = cli_args.inp_eps
            args.num_v_polys = vpolys_num
            args.perturbation_pick = pick
            args.perturbation_ndim = perturbation_ndim
            args.max_iterations = 10
            args.max_subrepair_loops = 4 # replace with 1 if you want to run targeted sub-repair

            # layer to repair
            if args.repair_mode == RepairMode.LP_hard or args.repair_mode == RepairMode.LP_ibp:
                start_last_layer_idx_map = {
                    "mnist 9x100": (10, 14),
                    "mnist_256x4": (3, 7),
                    "cifar10_cnn1": (2, 5),
                    "gtsrb_cnn": (2, 5),
                }
                args.repair_start_layer_idx, args.repair_last_layer_idx = \
                    start_last_layer_idx_map[model_name]
            else:
                raise ValueError(f"Unsupported repair mode: {args.repair_mode}")

            # optimization settings
            args.obj_regularization = "l2"  # fixed to l2
            args.obj_type = ObjectiveType.POLYTOPE_DISTANCE
            # args.obj_type = ObjectiveType.SLACK_INTERVAL
            args.lambda_strength_dist = cli_args.lambda_strength_dist
            args.lambda_strength_rate = cli_args.lambda_strength_rate
            args.lambda_reg = cli_args.lambda_reg
            args.modifiable_range = cli_args.modifiable_range
            args.use_priority_weight = False

            # FL parameters
            args.perform_FL = cli_args.perform_FL
            args.flmode = FLmode.ARACHNE
            args.fl_k_ratio = cli_args.fl_k_ratio
            args.pareto_round = cli_args.pareto_round

            repair_test(args)
            
            gc.collect()
