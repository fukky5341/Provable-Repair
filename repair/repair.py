import torch
import torch.nn as nn
import copy
import time
import os
import gc
import random

from net.util import get_net
from input_space.generate_input import ( repair_regions, train_dataloader, evaluation_data )
from input_space.region import RegionStatus
from LPsolver.solver import ( LPSolver, ObjectiveType )
from FL.fl import FL_selection
from .subspace import build_safe_subspaces
from .bound import ( get_concrete_bounds, check_violation_lp, bounds_lp )
from .args import RepairArgs, RepairMode, RepairTask
from .logging import logging
from .util import RepairStatus


# ==============
# helper functions
# ==============
def accuracy(dnn, dataloader):
    dnn.eval()
    correct = 0
    total = 0
    device = next(dnn.parameters()).device

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1)  # fix here

            outputs = dnn(inputs)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc, correct, total



def check_subspace_violation(net, lb, ub, subspace, logfile=None):
    ''' return True if the subspace is violated, False if the subspace is satisfied '''
    out_lb, out_ub, _ = get_concrete_bounds(net, lb, ub)
    subspace_lb, subspace_ub = subspace.lb, subspace.ub
    lb_violation_dim = (out_lb[-1] < subspace_lb).nonzero(as_tuple=True)[0]
    ub_violation_dim = (out_ub[-1] > subspace_ub).nonzero(as_tuple=True)[0]
    if len(lb_violation_dim) > 0 or len(ub_violation_dim) > 0:
        logging(f'  The number of violating dimensions: {len(lb_violation_dim)} (lb), {len(ub_violation_dim)} (ub)', line_break=0, logfile=logfile)
        return True
    else:
        logging('  No violation detected.', line_break=0, logfile=logfile)
        return False


def is_dnn_repaired(args: RepairArgs, repaired_dnn, repaired_layer_idx, repair_region_pairs):
    '''
    evaluate the repaired DNN
    Note: this evaluation is still approximate

    return:
        - is_violated:
            True if the repaired DNN satisfies all specifications
            False if the repaired DNN violates any specification
        - Pcenter_violation:
            True if the center point of any base region is misclassified after repair
            False otherwise

    '''
    is_violated = False
    Pcenter_violation = False
    for pair in repair_region_pairs:
        Nregions = pair.repaired_regions
        Pregion = pair.base_region
        Plb, Pub = Pregion.lb, Pregion.ub

        # check by subspace
        if repaired_layer_idx < len(repaired_dnn) - 1:
            logging(f"Checking subspace violation", logfile=args.logfile)
            sub_net = repaired_dnn[:repaired_layer_idx+1]
            logging(f"  base region ...", logfile=args.logfile)
            Psubspace = Pregion.subspace
            p_is_violated = check_subspace_violation(sub_net, Plb, Pub, Psubspace, logfile=args.logfile)
            n_is_violated = False
            for _nidx, Nregion in enumerate(Nregions):
                logging(f"  repaired region {_nidx + 1}/{len(Nregions)} ...", logfile=args.logfile)
                Nlb, Nub = Nregion.lb, Nregion.ub
                Nsubspace = Pregion.subspace if Nregion.status == RegionStatus.negative else Nregion.subspace
                n_is_violated = n_is_violated or check_subspace_violation(sub_net, Nlb, Nub, Nsubspace, logfile=args.logfile)
                if not p_is_violated and not n_is_violated:
                    Nregion.status = RegionStatus.positive
                    continue

        # check by using whole repaired DNN
        logging(f"Checking violation on the whole repaired DNN", logfile=args.logfile)
        logging(f"  base region ...", logfile=args.logfile)
        p_is_violated = check_violation_lp(repaired_dnn, Plb, Pub, Pregion.spec, logfile=args.logfile, debug=args.debug)
        n_is_violated = False
        for _nidx, Nregion in enumerate(Nregions):
            logging(f"  repaired region {_nidx + 1}/{len(Nregions)} ...", logfile=args.logfile)
            Nlb, Nub = Nregion.lb, Nregion.ub
            n_is_violated = n_is_violated or check_violation_lp(repaired_dnn, Nlb, Nub, Nregion.spec, logfile=args.logfile, debug=args.debug)

        logging(f"Status check after repair for data id {Nregion.data_id}:", logfile=args.logfile)
        if p_is_violated:
            logging(f"  Warning: Base region with data id {Pregion.data_id} is violating the specification after repair.", logfile=args.logfile)
            p_x = Pregion.center_point
            p_label = Pregion.spec.target_label
            if repaired_dnn(p_x).argmax().item() == p_label:
                logging(f"  However, the center point of the base region is still correctly classified.", logfile=args.logfile)
            else:
                logging(f"  The center point of the base region is misclassified after repair", logfile=args.logfile)
                Pcenter_violation = True
        else:
            logging(f"  Base region is still satisfying.", line_break=0, logfile=args.logfile)
        if n_is_violated:
            Nregion.status = RegionStatus.negative
            logging(f"  Repaired region is still violating.",line_break=0, logfile=args.logfile)
        else:
            Nregion.status = RegionStatus.positive
            logging(f"  Repaired region is no longer violating.", line_break=0, logfile=args.logfile)
        is_violated = is_violated or n_is_violated
    
    if is_violated:
        logging(f"Repaired DNN violates the specification for some repaired regions.", logfile=args.logfile)
        return False, Pcenter_violation
    else:
        logging(f"Repaired DNN satisfies the specification for all repaired regions.", logfile=args.logfile)
        return True, Pcenter_violation

def update_repaired_layer(dnn, layer_idx, new_params):
    new_W, new_B = new_params
    layer = dnn[layer_idx]

    layer.weight.data = new_W.to(layer.weight.device)
    layer.bias.data = new_B.to(layer.bias.device)


# ==============
# LP repair
# ==============

def repair_params_by_lp(args: RepairArgs, dnn, repaired_layer_idx, repair_region_pairs, tr_dataloader, last_layer_repair=False):
    if not last_layer_repair:
        repaired_net = dnn[:repaired_layer_idx+1]
    else:
        repaired_net = dnn
    layer = dnn[repaired_layer_idx]
    solver = LPSolver(modifiable_range=args.modifiable_range, weight=layer.weight, bias=layer.bias)

    # fault localization
    if args.perform_FL and not last_layer_repair:
        time_fl_start = time.time()
        logging(f"Fault localization for layer {repaired_layer_idx}...", border="bottom", logfile=args.logfile)
        weight_mask, bias_mask = FL_selection(args, dnn, repaired_net, repaired_layer_idx, layer, repair_region_pairs, tr_dataloader, last_layer_repair=last_layer_repair)
        solver.set_parameter_mask(args, weight_mask, bias_mask)
        time_fl_end = time.time()
        logging(f"  Selected {weight_mask.sum().item()}/{weight_mask.numel()} parameters for repair (weight), {bias_mask.sum().item()}/{bias_mask.numel()} parameters for repair (bias)", logfile=args.logfile)
        logging(f"  Fault localization time: {time_fl_end - time_fl_start:.2f} seconds", logfile=args.logfile)
    else:
        solver.set_repairable_parameters_all()


    logging(f'Solving the LP model to get repaired parameters for layer {repaired_layer_idx}...', border="bottom", logfile=args.logfile)

    # add constraints or objective
    time_build_start = time.time()
    if not last_layer_repair:
        for pair in repair_region_pairs:
            Nregion_set = pair.repaired_regions
            Pregion = pair.base_region

            Plb, Pub = Pregion.lb, Pregion.ub
            Psubspace = Pregion.subspace
            # build net structure in the solver
            if args.repair_mode == RepairMode.LP_hard:
                # add constraints to ensure the output of the base region is in the safe subspace
                _, _ = solver.build_net_with_subspace(repaired_net, Plb, Pub, Psubspace)
            elif args.repair_mode == RepairMode.LP_ibp:
                # add ibp constraints for the base region
                _, _, Pbounder = get_concrete_bounds(repaired_net, Plb, Pub, save_coeffs=False)
                solver.set_objective_ibp(args, repaired_net, Pbounder, Pregion, Psubspace, obj_weight=args.obj_weight_positive)
                del Pbounder

            for Nregion in Nregion_set:
                if Nregion.status == RegionStatus.positive:
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    Nsubspace = Nregion.subspace
                    if args.repair_mode == RepairMode.LP_hard:
                        # add constraints to ensure the output of the repaired region is in the safe subspace
                        _, _ = solver.build_net_with_subspace(repaired_net, Nlb, Nub, Nsubspace)
                    elif args.repair_mode == RepairMode.LP_ibp:
                        # add ibp constraints for the repaired region
                        _, _, Nbounder = get_concrete_bounds(repaired_net, Nlb, Nub, save_coeffs=False)
                        solver.set_objective_ibp(args, repaired_net, Nbounder, Nregion, Nsubspace)
                        del Nbounder
                elif Nregion.status == RegionStatus.negative:
                    # if last_layer_repair:
                    #     Nlb, Nub = Nregion.lb, Nregion.ub
                    #     Nsubspace = Pregion.subspace
                    #     _, _ = solver.build_net_with_subspace(repaired_net, Nlb, Nub, Nsubspace)
                    # else:
                    #     # set the objective to minimize the distance between the repaired region and the safe subspace
                    #     objective_type = args.obj_type
                    #     solver.set_objective(Pvars, repaired_net, Nregion, Psubspace, obj_type=objective_type)

                    # set the objective to minimize the distance between the repaired region and the safe subspace
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    _, _, Nbounder = get_concrete_bounds(repaired_net, Nlb, Nub, save_coeffs=False)
                    if args.repair_mode == RepairMode.LP_hard:
                        solver.set_objective(args, repaired_net, Nbounder, Nregion, Psubspace)
                    elif args.repair_mode == RepairMode.LP_ibp:
                        solver.set_objective_ibp(args, repaired_net, Nbounder, Nregion, Psubspace)
                    del Nbounder
                else:
                    raise ValueError(f"Unsupported region status: {Nregion.status}")
    else:
        for pair in repair_region_pairs:
            Nregion_set = pair.repaired_regions
            Pregion = pair.base_region

            if args.repair_task == RepairTask.LocalCounterexample:
                pass
            else:  # todo
                # -- Pregion --
                Plb, Pub = Pregion.lb, Pregion.ub
                _, _, Pbounder = get_concrete_bounds(repaired_net, Plb, Pub, save_coeffs=False)
                solver.set_objective_lastlayer(args, repaired_net, Pbounder, Pregion)
                del Pbounder

            # -- Nregion --
            for Nregion in Nregion_set:
                Nlb, Nub = Nregion.lb, Nregion.ub
                _, _, Nbounder = get_concrete_bounds(repaired_net, Nlb, Nub, save_coeffs=False)
                solver.set_objective_lastlayer(args, repaired_net, Nbounder, Nregion)
                del Nbounder
    
    if args.obj_regularization == "l1":
        solver.build_objective_l1(
            strength_dist=args.lambda_strength_dist,
            lambda_reg=args.lambda_reg
        )
    elif args.obj_regularization == "l2":
        solver.build_objective_l2(
            strength_dist=args.lambda_strength_dist,
            lambda_reg=args.lambda_reg
        )
    else:
        raise ValueError(f"Unsupported regularization type: {args.obj_regularization}")
    
    time_build_end = time.time()
    logging(f"LP model built ({time_build_end - time_build_start:.2f} seconds). Starting optimization...", logfile=args.logfile)

    time_opt_start = time.time()
    repaired_params = solver.repair_solve()
    time_opt_end = time.time()
    logging(f"LP optimization completed.", logfile=args.logfile)
    logging(f"  time: {time_opt_end - time_opt_start:.2f} seconds", line_break=0, logfile=args.logfile)

    solver.dispose()
    del solver
    if '_' in locals():
        del _
    gc.collect()

    # update the repaired layer with repaired parameters
    if repaired_params is not None:
        logging(f'LP solver found repaired parameters for layer {repaired_layer_idx}. Updating the layer with repaired parameters.', logfile=args.logfile)
        update_repaired_layer(dnn, repaired_layer_idx, repaired_params)
        return RepairStatus.PROCEEDING
    else:
        logging(f"LP solver failed to find a solution for layer {repaired_layer_idx}. Skipping this layer.", logfile=args.logfile)
        return RepairStatus.FAILED


# ===================================
# repair layer
# ===================================

def repair_layer(args: RepairArgs, dnn, repaired_layer_idx, repair_region_pairs, tr_dataloader, last_layer_repair=False):

    if not last_layer_repair:
        # --- collect safe subspaces ---
        logging(f'Collecting safe subspaces for the repaired layer {repaired_layer_idx}...', border="bottom", logfile=args.logfile)
        repair_region_pairs = build_safe_subspaces(args, dnn, repaired_layer_idx, repair_region_pairs, logfile=args.logfile)

        # debug
        if args.debug:
            logging(f'Checking subspace validation (by base region)', logfile=args.logfile)
            for pair in repair_region_pairs:
                Pregion = pair.base_region
                logging(f"data id {Pregion.data_id}...", logfile=args.logfile)
                Plb, Pub = Pregion.lb, Pregion.ub
                psubspace = Pregion.subspace
                check_subspace_violation(dnn[:repaired_layer_idx+1], Plb, Pub, psubspace, logfile=args.logfile)

            logging(f'Satisfication of subspace constraints on the repaired region before repair', logfile=args.logfile)
            for pair in repair_region_pairs:
                Pregion = pair.base_region
                for _nid, Nregion in enumerate(pair.repaired_regions):
                    logging(f"data id {Nregion.data_id}, process {(_nid + 1)}/{len(pair.repaired_regions)}", logfile=args.logfile)
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    psubspace = Pregion.subspace
                    check_subspace_violation(dnn[:repaired_layer_idx+1], Nlb, Nub, psubspace, logfile=args.logfile)
    else:
        logging(f"Last layer repair: skipping safe subspace construction.", border="bottom", logfile=args.logfile)

    # --- repair operation ---
    if args.repair_mode in (RepairMode.LP_hard, RepairMode.LP_ibp):
        # --- get repaired parameters by solving the lp model ---
        repair_status = repair_params_by_lp(args, dnn, repaired_layer_idx, repair_region_pairs, tr_dataloader, last_layer_repair=last_layer_repair)
    else:
        raise ValueError(f"Unsupported repair mode: {args.repair_mode}")

    return repair_status



# ===============
# Repair test
# ===============
# layer set to repair
def repair_layer_set(dnn, start_layer_idx, last_layer_idx):
    layer_set = []
    for i in range(start_layer_idx, last_layer_idx + 1):
        if isinstance(dnn[i], nn.Linear):
            layer_set.append(i)
        elif isinstance(dnn[i], nn.Conv2d):
            layer_set.append(i)
        else:
            continue
    return layer_set

# layer list of sets to repair
def repair_layer_list_of_sets(dnn, start_layer_idx, last_layer_idx):
    linear_layers = repair_layer_set(dnn, start_layer_idx, last_layer_idx)
    num_linear_layers = len(linear_layers)
    list_of_sets = []
    for i in range(num_linear_layers):
        list_of_sets.append(linear_layers[i:])
    list_of_sets.append([-1])  # add the last layer repair set
    return list_of_sets[::-1]  # reverse the order so that we start from the last layer


# termination condition for repair loop
def termination_condition(repair_status):
    if repair_status == RepairStatus.REPAIRED:
        return True
    else:
        return False



def repair_test(args: RepairArgs):

    # --- Load network ---
    repaired_dnn, _norm, _denorm = get_net(args)
    original_dnn = copy.deepcopy(repaired_dnn)

    # --- repaired layer index ---
    repair_start_layer_idx = args.repair_start_layer_idx
    repair_last_layer_idx = args.repair_last_layer_idx
    # repair_last_layer_idx = len(repaired_dnn) - 1
    if args.max_subrepair_loops > 1:
        subrepair_set_list = repair_layer_list_of_sets(repaired_dnn, repair_start_layer_idx, repair_last_layer_idx)
        subrepair_set_list = subrepair_set_list[:args.max_subrepair_loops] if len(subrepair_set_list) > args.max_subrepair_loops else subrepair_set_list
    else:
        subrepair_set_list = [repair_layer_set(repaired_dnn, repair_start_layer_idx, repair_last_layer_idx)]

    # inp eps
    int_eps = int(args.inp_eps)
    args.inp_eps = int_eps / 255.0

    # --- positive samples for fault localization collected from training data ---
    tr_dataloader = train_dataloader(args)

    total_num = args.num_runs * args.num_v_polys

    repair_region_pairs_all, additional_time = repair_regions(
        args=args,
        dnn=repaired_dnn,
        total_num=total_num,
        normalize_input=_norm
    )  # list, list
    # divide pairs
    repair_region_pairs_list = []
    if len(repair_region_pairs_all) < total_num:
        # randomly sample pairs if the total number of pairs is smaller than num_runs * num_v_polys
        random.seed(0)
        # num_runs list, each containing num_v_polys pairs
        for _ in range(args.num_runs):
            tmp_sampled_list = random.sample(repair_region_pairs_all, args.num_v_polys)
            sampled_list = copy.deepcopy(tmp_sampled_list)
            repair_region_pairs_list.append(sampled_list)
    else:
        for i in range(args.num_runs):
            start_idx = i * args.num_v_polys
            end_idx = (i + 1) * args.num_v_polys
            repair_region_pairs_list.append(repair_region_pairs_all[start_idx:end_idx])
        
    if args.perturbation_pick == 'nonzero':
        log_name = f"eps{int_eps}_N{args.num_v_polys}_ndims{args.perturbation_ndim}"
    elif args.perturbation_pick == 'all':
        log_name = f"eps{int_eps}_N{args.num_v_polys}"
    else:
        raise ValueError(f"Unsupported perturbation pick method: {args.perturbation_pick}")
    if args.perform_FL:
        log_name += f"_{args.flmode.name}{args.fl_k_ratio}_pr{args.pareto_round}"
        log_dir = f"result/FL/{args.model_name}/{args.repair_task.name}/{args.perturbation_pick}"
    else:
        log_dir = f"result/ours/{args.model_name}/{args.repair_task.name}/{args.perturbation_pick}"
    os.makedirs(log_dir, exist_ok=True)
    args.logfile = f"{log_dir}/{log_name}.txt"
    logfile = args.logfile

    with open(logfile, "w") as f:
        f.write("=== Repair Test Log ===\n")
        # experiment mode and repair mode
        f.write(f"Repair task: {args.repair_task.value}\n")
        f.write(f"Repair mode: {args.repair_mode.value}\n")
        # number of runs
        f.write(f"Number of runs: {args.num_runs}\n")
        # network
        f.write(f"Network: {args.model_name}\n")
        f.write("\n")
        # repair parameters
        f.write(f"Target label: {args.target_label}\n")
        f.write(f"Number of v-polytopes: {args.num_v_polys}\n")
        f.write(f"Perturbation distance (eps): {args.inp_eps}\n")
        f.write(f"Perturbation pick: {args.perturbation_pick}\n")
        if args.perturbation_pick == 'nonzero':
            f.write(f"Perturbation ndim: {args.perturbation_ndim}\n")
        f.write("\n")
        # repaired layer indices
        f.write(f"# subrepair loops: {args.max_subrepair_loops}\n")
        for i, subrepair_set in enumerate(subrepair_set_list):
            f.write(f"subrepair set {i + 1}: {list(subrepair_set)}\n")
        f.write("\n")
        # optimization settings
        f.write(f"Modifiable range: {args.modifiable_range}\n")
        f.write(f"Objective parameter regularization: {args.obj_regularization}\n")
        f.write(f"Objective type: {args.obj_type}\n")
        f.write(f"Strength of distance term in objective: {args.lambda_strength_dist}\n")
        f.write(f"Strength increase rate for distance term: {args.lambda_strength_rate}\n")
        f.write(f"Strength of regularization term: {args.lambda_reg}\n")
        f.write(f"Use priority weight: {args.use_priority_weight}\n")
        f.write(f"Objective weight for positive region (if not using priority weight): {args.obj_weight_positive}\n")
        f.write("\n")
        # FL settings
        f.write(f"FL performed: {args.perform_FL}\n")
        if args.perform_FL:
            f.write(f"FL mode: {args.flmode}\n")
            f.write(f"FL k ratio: {args.fl_k_ratio}\n")
            f.write(f"Pareto rounds: {args.pareto_round}\n")
        f.write("\n")
        # additional time for input space construction
        f.write(f"Additional time for input space construction: {additional_time:.2f} seconds\n")
        f.write(f"Average time per repair region: {additional_time / len(repair_region_pairs_all):.2f} seconds\n")
        f.write("\n\n")
    
    results = []
    # --- perform repair for each set of repair regions ---
    for run_idx in range(len(repair_region_pairs_list)):
        repair_region_pairs = repair_region_pairs_list[run_idx]
        logging(f"=== Repair run {run_idx + 1}/{args.num_runs} ===", border="both", logfile=logfile)

        data_ids = [pair.base_region.data_id for pair in repair_region_pairs]
        logging(f"Data ids for this repair run: {data_ids}", logfile=logfile)

        # repaired network
        dnn = copy.deepcopy(repaired_dnn)

        # evaluation data (test set)
        acc_dataloader, gen_dataloader_all = evaluation_data(args, repair_region_pairs)

        if args.debug:
            # logging how much original network is violating
            logging(f"Spec lower bound of original network", border="bottom", logfile=logfile)
            for pair in repair_region_pairs:
                for _nid, Nregion in enumerate(pair.repaired_regions):
                    logging(f"  data id {Nregion.data_id} ({_nid + 1}/{len(pair.repaired_regions)})", logfile=logfile)
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    _ = bounds_lp(original_dnn, Nlb, Nub, Nregion.spec, logfile=logfile)

        time_repair_start = time.time()
        
        # --- Repair simultaneously ---
        if args.repair_mode in (RepairMode.LP_hard, RepairMode.LP_ibp):
            repair_status = RepairStatus.PROCEEDING
            last_layer_idx = len(dnn) - 1
            if subrepair_set_list[0][0] == -1:
                subrepair_set_list[0][0] = last_layer_idx  # replace -1 with the last layer index
            # --- subrepair loop ---
            for subrepair_idx, subrepair_set in enumerate(subrepair_set_list):
                if repair_status == RepairStatus.REPAIRED:
                    break
                logging(f"Subrepair loop {subrepair_idx + 1}/{len(subrepair_set)}", border_type="=", line_break=2, border="both", logfile=logfile)

                # --- subrepair ---
                step = 0
                while not termination_condition(repair_status) and step < args.max_iterations:
                    for repaired_layer_idx in subrepair_set:
                        logging(f"Repairing layer {repaired_layer_idx} with strength {args.lambda_strength_dist}", line_break=2, border="both", logfile=logfile)

                        tmp_dnn = copy.deepcopy(dnn)

                        # repair layer
                        if repaired_layer_idx == last_layer_idx:
                            last_layer_repair = True
                        else:
                            last_layer_repair = False
                        repair_status = repair_layer(args, dnn, repaired_layer_idx, repair_region_pairs,
                                                    tr_dataloader, last_layer_repair=last_layer_repair)
                        step += 1
                        # --- evaluate the repaired DNN ---
                        if repair_status == RepairStatus.PROCEEDING:
                            logging(f'Evaluating the repaired DNN after repairing layer {repaired_layer_idx}...', border="bottom", logfile=logfile)
                            is_repaired, Pcenter_violation = is_dnn_repaired(args, dnn, repaired_layer_idx, repair_region_pairs)
                            if is_repaired:
                                logging(f"Step {step}: Successfully repaired the DNN!", logfile=logfile)
                                repair_status = RepairStatus.REPAIRED
                                break
                            else:
                                if Pcenter_violation:
                                    logging(f"Step {step}: Center point violation. Skip this layer...", logfile=logfile)
                                    del dnn
                                    dnn = tmp_dnn
                                logging(f"Step {step}: Layer {repaired_layer_idx} failed. Trying the next layer...", logfile=logfile)
                    
                    if repair_status == RepairStatus.PROCEEDING and repaired_layer_idx != last_layer_idx:
                        logging(f"Repairing layer {last_layer_idx} with strength {args.lambda_strength_dist}", line_break=2, border="both", logfile=logfile)

                        tmp_dnn = copy.deepcopy(dnn)
                        repair_status = repair_layer(args, dnn, last_layer_idx, repair_region_pairs, tr_dataloader, last_layer_repair=True)
                        step += 1
                        if repair_status == RepairStatus.PROCEEDING:
                            logging(f'Step {step}: Evaluating the repaired DNN after repairing the last layer...', border="bottom", logfile=logfile)
                            is_repaired, Pcenter_violation = is_dnn_repaired(args, dnn, last_layer_idx, repair_region_pairs)
                            if is_repaired:
                                logging(f"Step {step}: Successfully repaired the DNN!", logfile=logfile)
                                repair_status = RepairStatus.REPAIRED
                                break
                            else:
                                if Pcenter_violation:
                                    logging(f"Step {step}: Center point violation for the last layer repair. Skip this layer...", logfile=logfile)
                                    del dnn
                                    dnn = tmp_dnn
                                logging(f"Step {step}: Repair unsuccessful for the last layer. Restarting the next iteration with increased strength...", logfile=logfile)

                    if not termination_condition(repair_status):
                        # increase the strength of distance term in the objective for the next iteration
                        args.lambda_strength_dist *= args.lambda_strength_rate

        time_repair_end = time.time()
        logging(f"Total repair time: {time_repair_end - time_repair_start:.2f} seconds", logfile=logfile)
        logging(f"Total steps: {step}", logfile=logfile)

        # --- evaluate the repaired DNN ---
        if repair_status == RepairStatus.REPAIRED:
            logging('Evaluating the repaired DNN after repair...', border="bottom", logfile=logfile)
            # original network accuracy
            original_acc, original_correct, original_total = accuracy(original_dnn, acc_dataloader)
            logging(f"Original DNN accuracy on the evaluation dataset: {original_acc:.4f} ({original_correct}/{original_total})", logfile=logfile)
            # repaired network accuracy
            repaired_acc, repaired_correct, repaired_total = accuracy(dnn, acc_dataloader)
            logging(f"Repaired DNN accuracy on the evaluation dataset: {repaired_acc:.4f} ({repaired_correct}/{repaired_total})", logfile=logfile)
            results.append((original_acc, repaired_acc))

            # generalization check
            for gen_id, gen_dataloaders in gen_dataloader_all:
                logging(f"Generalization check for data id {gen_id}...", logfile=logfile)
                
                for gen_dist, gen_dataloader in gen_dataloaders:
                    gen_acc0, gen_correct0, gen_total0 = accuracy(original_dnn, gen_dataloader)
                    gen_acc, gen_correct, gen_total = accuracy(dnn, gen_dataloader)
                    logging(f"Repaired DNN accuracy on the generalization dataset (dist={gen_dist:.4f}): \n{gen_acc0:.4f} ({gen_correct0}/{gen_total0}) -> {gen_acc:.4f} ({gen_correct}/{gen_total})", logfile=logfile)
        else:
            logging("Repair failed. No repaired DNN to evaluate.", logfile=logfile)
            results.append((None, None))

        del dnn, gen_dataloader_all, acc_dataloader
        if '_' in locals():
            del _
        gc.collect()

    # summary of results
    logging("=== Summary of Repair Results ===", border="both", logfile=logfile)
    for run_idx, (original_acc, repaired_acc) in enumerate(results):
        if original_acc is not None and repaired_acc is not None:
            logging(f"Run {run_idx + 1}: Original accuracy = {original_acc}, Repaired accuracy = {repaired_acc}", logfile=logfile)
        else:
            logging(f"Run {run_idx + 1}: Repair failed. No accuracy results.", logfile=logfile)
