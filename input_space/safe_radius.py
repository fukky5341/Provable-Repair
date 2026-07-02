import time
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # for imports

from input_space import safe_center
from repair.logging import logging
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairMode, RepairTask
from net.util import get_net
from input_space.generate_input import repair_corrupted_points, maximal_safe_shrink, check_violation, clean_points, adv_dataset

class dummySpec:
    def __init__(self, C):
        self.C = C

net_dataset_map = {
    0: ( "mnist 9x100", Dataset.MNIST ),
    1: ( "mnist_256x4", Dataset.MNIST ),
    2: ( "mnist_256x6", Dataset.MNIST ),
    3: ( "mnist_conv", Dataset.MNIST ),
    4: ( "cifar10_cnn1", Dataset.CIFAR10 ),
    5: ( "cifar10_cnn2", Dataset.CIFAR10 ),
    6: ( "gtsrb_cnn", Dataset.GTSRB ),
}

def collect_safe_radii_corruption(repair_task, target_ids=None):
    net_id = 1
    model_name, dataset = net_dataset_map[net_id]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/safe_radius_log_{timestamp}.txt"

    args = RepairArgs(
        repair_task=repair_task,
        logfile=logfile,
        model_name=model_name,
        dataset=dataset,
    )

    net, _norm, _denorm = get_net(args)
    
    collect_num = 300
    args.inp_eps = 1./255.

    repaired_indices, repaired_points, base_points = repair_corrupted_points(args, net)

    if target_ids is not None:
        collect_num = len(target_ids)

    safe_input_spaces = {}
    missed = []
    for _i in range(min(collect_num, len(base_points.images))):
        if target_ids is not None:
            i = target_ids[_i]
        else:
            i = _i
        print(f"Processing {_i+1}/{min(collect_num, len(base_points.images))}...")
        time_start = time.time()
        data_id = base_points.indices[i]
        data_id_n = repaired_points.indices[i]
        assert data_id == data_id_n, f"Data ID mismatch: {data_id} vs {data_id_n}"
        x = base_points.images[i].unsqueeze(0).to(device=args.device, dtype=args.dtype)
        label = base_points.labels[i].item()
        num_classes = net(x).shape[1]
        pre_lb = torch.clamp(x - args.inp_eps, 0.0, 1.0).squeeze(0)
        pre_ub = torch.clamp(x + args.inp_eps, 0.0, 1.0).squeeze(0)
        C = torch.eye(num_classes, device=args.device, dtype=args.dtype)
        C = C[label:label+1] - torch.cat([C[:label], C[label+1:]], dim=0)
        spec = dummySpec(C=C)
        if check_violation(net, pre_lb, pre_ub, spec):
            new_lb, new_ub, _ = maximal_safe_shrink(net, pre_lb, pre_ub, spec, iters=15)
        else:
            new_lb, new_ub = pre_lb, pre_ub
        time_end = time.time()

        if new_lb is None or new_ub is None:
            missed.append((i, data_id))
            print(f"  Missed case for data index {data_id}.")
            safe_input_spaces[data_id] = {
                "lb": None,
                "ub": None,
                "time": time_end - time_start,
            }
        else:
            print(f"  Safe radius found. Time taken: {time_end - time_start:.2f} seconds.")
            safe_input_spaces[data_id] = {
                "lb": new_lb.cpu(),
                "ub": new_ub.cpu(),
                "time": time_end - time_start,
            }

    # save results as a torch file
    save_dir = "safe_radii"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if target_ids is not None:
        # ids_str = "_".join(str(tid) for tid in target_ids)
        # torch.save(safe_input_spaces, f"{save_dir}/{model_name}_target{ids_str}_num{collect_num}.pt")
        torch.save(safe_input_spaces, f"{save_dir}/{model_name}_target_num{collect_num}.pt")
    else:
        torch.save(safe_input_spaces, f"{save_dir}/{model_name}_num{collect_num}.pt")

    # log missed cases
    if len(missed) > 0:
        print("Missed cases (order index, data index):")
        for idx, data_id in missed:
            print(f"  {idx}: {data_id}")
    else:
        print("No missed cases.")

def collect_safe_radii_adv(repair_task, target_ids=None):
    net_id = 1
    model_name, dataset = net_dataset_map[net_id]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/safe_radius_log_{timestamp}.txt"

    args = RepairArgs(
        repair_task=repair_task,
        logfile=logfile,
        model_name=model_name,
        dataset=dataset,
    )

    net, _norm, _denorm = get_net(args)
    
    collect_num = 50
    args.inp_eps = 1./255.

    if args.repair_task == RepairTask.AdversarialAndPerturbation or args.repair_task == RepairTask.LocalRobustness:
        clean_data = clean_points(args)
        adv_data = adv_dataset(args, clean_data, net, eps=args.inp_eps, samples_num=collect_num, target_label=args.target_label)

    if target_ids is not None:
        collect_num = len(target_ids)

    safe_input_spaces = {}
    missed = []
    for _i in range(min(collect_num, len(adv_data.images))):
        if target_ids is not None:
            i = target_ids[_i]
        else:
            i = _i
        print(f"Processing {_i+1}/{min(collect_num, len(adv_data.images))}...")
        time_start = time.time()
        data_id = adv_data.indices[i].item()
        x = adv_data.ori_images[i].unsqueeze(0).to(device=args.device, dtype=args.dtype)
        label = adv_data.labels[i].item()
        num_classes = net(x).shape[1]
        pre_lb = torch.clamp(x - args.inp_eps, 0.0, 1.0).squeeze(0)
        pre_ub = torch.clamp(x + args.inp_eps, 0.0, 1.0).squeeze(0)
        C = torch.eye(num_classes, device=args.device, dtype=args.dtype)
        C = C[label:label+1] - torch.cat([C[:label], C[label+1:]], dim=0)
        spec = dummySpec(C=C)
        if check_violation(net, pre_lb, pre_ub, spec):
            new_lb, new_ub, _ = maximal_safe_shrink(net, pre_lb, pre_ub, spec, iters=20)
        else:
            new_lb, new_ub = pre_lb, pre_ub
        time_end = time.time()

        if new_lb is None or new_ub is None:
            missed.append((i, data_id))
            print(f"  Missed case for data index {data_id}.")
            safe_input_spaces[data_id] = {
                "label": label,
                "center": adv_data.ori_images[i].cpu(),
                "adv_center": adv_data.images[i].cpu(),
                "lb": None,
                "ub": None,
                "time": time_end - time_start,
            }
        else:
            print(f"  Safe radius found. Time taken: {time_end - time_start:.2f} seconds.")
            safe_input_spaces[data_id] = {
                "label": label,
                "center": adv_data.ori_images[i].cpu(),
                "adv_center": adv_data.images[i].cpu(),
                "lb": new_lb.cpu(),
                "ub": new_ub.cpu(),
                "time": time_end - time_start,
            }

    # save results as a torch file
    save_dir = "safe_radii"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if target_ids is not None:
        # ids_str = "_".join(str(tid) for tid in target_ids)
        # torch.save(safe_input_spaces, f"{save_dir}/{model_name}_target{ids_str}_num{collect_num}.pt")
        torch.save(safe_input_spaces, f"{save_dir}/adv_{model_name}_target_num{collect_num}.pt")
    else:
        torch.save(safe_input_spaces, f"{save_dir}/adv_{model_name}_num{collect_num}.pt")

    # log missed cases
    if len(missed) > 0:
        print("Missed cases (order index, data index):")
        for idx, data_id in missed:
            print(f"  {idx}: {data_id}")
    else:
        print("No missed cases.")

def main():
    # repair_task = RepairTask.CorruptionAndPerturbation
    repair_task = RepairTask.AdversarialAndPerturbation

    time_start = time.time()
    numbers = [i for i in range(0, 300)]
    # target_ids = numbers[0: 0+100]
    target_ids = None
    if repair_task == RepairTask.CorruptionAndPerturbation:
        collect_safe_radii_corruption(repair_task, target_ids)
    else:
        collect_safe_radii_adv(repair_task, target_ids)
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start:.2f} seconds.")

if __name__ == "__main__":
    main()
    print("Safe radius collection completed.")