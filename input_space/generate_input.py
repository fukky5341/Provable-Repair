import torch
import numpy as np
import copy
import time

from experiments import ( mnist, cifar10, gtsrb, acas )

from .region import Region, RegionStatus, RegionPair
from .dataset import Dataset
from repair.bound import check_violation
from repair.args import RepairTask, RepairMode
from .advex import adv_dataset
from repair.logging import logging


def _normalize_positions(keep_pos):
    if isinstance(keep_pos, torch.Tensor):
        keep_pos = keep_pos.detach().cpu().tolist()
    elif isinstance(keep_pos, np.ndarray):
        keep_pos = keep_pos.tolist()
    return [int(i) for i in keep_pos]


def _subset_field(field, keep_pos):
    if isinstance(field, torch.Tensor):
        idx = torch.tensor(keep_pos, device=field.device, dtype=torch.long)
        return field.index_select(0, idx).clone()
    elif isinstance(field, np.ndarray):
        return field[np.array(keep_pos, dtype=np.int64)].copy()
    else:
        return [field[i] for i in keep_pos]


def subset_by_positions(points, keep_pos):
    """
    Keep entries by local positions inside the current dataset object.
    """
    keep_pos = _normalize_positions(keep_pos)

    # IMPORTANT: use deepcopy, not points.copy()
    subset = copy.deepcopy(points)

    subset.images = _subset_field(points.images, keep_pos)
    subset.labels = _subset_field(points.labels, keep_pos)

    if hasattr(points, "indices") and points.indices is not None:
        subset.indices = _subset_field(points.indices, keep_pos)

    return subset


def subset_by_original_indices(points, selected_indices):
    selected_set = set(int(idx) for idx in selected_indices)

    keep_pos = [
        i for i, orig_idx in enumerate(points.indices)
        if int(orig_idx) in selected_set
    ]

    return subset_by_positions(points, keep_pos)


def filter_by_label(points, target_label):
    if target_label is None:
        return points

    if isinstance(points.labels, torch.Tensor):
        keep_pos = (points.labels == target_label).nonzero(as_tuple=True)[0]
    elif isinstance(points.labels, np.ndarray):
        keep_pos = np.where(points.labels == target_label)[0]
    else:
        keep_pos = [i for i, y in enumerate(points.labels) if int(y) == target_label]

    return subset_by_positions(points, keep_pos)


@torch.no_grad()
def split_by_misclassification(points, dnn):
    logits = dnn(points.images)
    preds = logits.argmax(dim=1)

    if isinstance(points.labels, torch.Tensor):
        labels = points.labels.to(device=preds.device, dtype=preds.dtype)
    elif isinstance(points.labels, np.ndarray):
        labels = torch.from_numpy(points.labels).to(device=preds.device, dtype=preds.dtype)
    else:
        labels = torch.tensor(points.labels, device=preds.device, dtype=preds.dtype)

    labels = labels.reshape(-1).to(dtype=preds.dtype)

    correct_mask = (preds == labels)
    wrong_mask = ~correct_mask

    pos_idx = correct_mask.nonzero(as_tuple=True)[0]
    neg_idx = wrong_mask.nonzero(as_tuple=True)[0]

    pos_points = subset_by_positions(points, pos_idx)
    neg_points = subset_by_positions(points, neg_idx)

    return pos_points, neg_points


def train_dataset(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)


def train_dataloader(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)


def clean_points(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)


def damaged_points(args):
    if args.dataset == Dataset.MNIST:
        return (
            mnist.datasets.MNIST_C(corruption='fog', split='test')
            .reshape(args.input_shape)
            .to(device=args.device, dtype=args.dtype)
        )
    elif args.dataset == Dataset.CIFAR10:
        return (
            cifar10.datasets.Dataset('fog', 'test')
            .reshape(args.input_shape)
            .to(device=args.device, dtype=args.dtype)
        )


def neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points):
    """
    Match by original dataset index, and force the same ordering in both outputs.
    """
    clean_map = {int(idx): i for i, idx in enumerate(pos_clean_points.indices)}
    damaged_map = {int(idx): i for i, idx in enumerate(neg_damaged_points.indices)}

    common_indices = sorted(set(clean_map.keys()) & set(damaged_map.keys()))

    clean_pos = [clean_map[idx] for idx in common_indices]
    damaged_pos = [damaged_map[idx] for idx in common_indices]

    clean_original_points = subset_by_positions(pos_clean_points, clean_pos)
    repaired_points = subset_by_positions(neg_damaged_points, damaged_pos)

    # overwrite indices with aligned original indices
    clean_original_points.indices = common_indices
    repaired_points.indices = common_indices

    return common_indices, repaired_points, clean_original_points


def repair_corrupted_points(args, dnn):
    c_points = clean_points(args)
    d_points = damaged_points(args)

    if args.target_label is not None:
        f_c_points = filter_by_label(c_points, args.target_label)
        f_d_points = filter_by_label(d_points, args.target_label)
    else:
        f_c_points = c_points
        f_d_points = d_points

    pos_clean_points, neg_clean_points = split_by_misclassification(f_c_points, dnn)
    pos_damaged_points, neg_damaged_points = split_by_misclassification(f_d_points, dnn)

    repaired_indices, repaired_points, base_points = \
        neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points)

    return repaired_indices, repaired_points, base_points


def maximal_safe_shrink(net, lb, ub, spec, iters=20):
    center = (lb + ub) / 2
    width = (ub - lb) / 2

    low = 0.0      # always safe (point)
    high = 1.0     # current box (unsafe)

    best = 0.0

    for _ in range(iters):
        mid = (low + high) / 2

        new_lb = center - mid * width
        new_ub = center + mid * width

        if check_violation(net, new_lb, new_ub, spec):
            high = mid   # still unsafe → shrink more
        else:
            best = mid   # safe → can expand
            low = mid

        if high - low < 1e-4:  # Convergence threshold
            break
    
    if best == 0.0:
        # raise ValueError("Failed to find a safe shrink. The original box might already be safe or the specification might be unsatisfiable.")
        return None, None, best

    new_lb = center - best * width
    new_ub = center + best * width

    return new_lb, new_ub, best


# ===== construct safe region with the largest radius =====
def find_largest_safe_radius(dnn, center_point, spec, max_radius, iters=20):
    low = 0.0
    high = max_radius
    best_radius = 0.0

    for step in range(iters):
        # debug
        time_step_start = time.time()

        mid = (low + high) / 2
        lb = (center_point - mid).squeeze(0)
        ub = (center_point + mid).squeeze(0)

        if check_violation(dnn, lb, ub, spec):
            high = mid  # Violation occurs, reduce the radius
        else:
            best_radius = mid  # No violation, try a larger radius
            low = mid

        # debug
        time_step_end = time.time()
        print(f"Binary search step {step+1}/{iters}, low={low:.4f}, high={high:.4f}, best={best_radius:.4f}, time={time_step_end - time_step_start:.2f}s")
        
        if high - low < 1e-4:  # Convergence threshold
            break
    
    return best_radius


# ===== prepare repair box based on the perturbation_pick setting =====
def prepare_repair_box(center_point, inp_eps, perturbation_pick, perturbation_ndim):
    if perturbation_pick == 'nonzero':
        flat_image = center_point.flatten()
        non_zero_indices = torch.nonzero(flat_image, as_tuple=True)[0]
        num_elements = min(perturbation_ndim, len(non_zero_indices))
        indices_to_perturb = non_zero_indices[:num_elements]
        mask = torch.zeros_like(flat_image, dtype=torch.bool)
        mask[indices_to_perturb] = True
        mask = mask.view_as(center_point)
        lb = torch.where(mask, center_point - inp_eps, center_point)
        ub = torch.where(mask, center_point + inp_eps, center_point)
    elif perturbation_pick == 'all':  # 'all'
        lb = center_point - inp_eps
        ub = center_point + inp_eps
    else:
        raise ValueError(f"Invalid perturbation_pick value: {perturbation_pick}")

    lb = torch.clamp(lb, min=0, max=1).squeeze(0)
    ub = torch.clamp(ub, min=0, max=1).squeeze(0)

    return lb, ub


# ===== construct repair regions based on the experiment setting =====

# generate repair regions for CorruptionAndPerturbation experiment
def repair_regions_corruption_and_perturbation(args, dnn, total_num):
    device = args.device
    dtype = args.dtype
    neg_eps = args.inp_eps
    pos_eps = args.inp_eps

    repaired_indices, repaired_points, base_points = repair_corrupted_points(args, dnn)

    # safe radii
    data = torch.load(f"safe_radii/{args.model_name}_num300.pt")
    if total_num > len(data):
        total_num = len(data)

    additional_time = 0.0

    pair_list = []
    for i in range(len(repaired_points)):
        # repaired region
        center_poi_neg = repaired_points.images[i].unsqueeze(0).to(device=device, dtype=dtype)
        Nlb, Nub = prepare_repair_box(center_poi_neg, neg_eps, args.perturbation_pick, args.perturbation_ndim)
        repaired_region = Region(
            center_point = center_poi_neg,
            lb = Nlb,
            ub = Nub,
            target_label = repaired_points.labels[i].item(),
            data_id=repaired_indices[i],
            status=RegionStatus.negative
        )
        # base region
        center_poi_pos = base_points.images[i].unsqueeze(0).to(device=device, dtype=dtype)
        base_region = Region(
            center_point = center_poi_pos,
            lb = torch.clamp(center_poi_pos - pos_eps, min=0, max=1).squeeze(0),
            ub = torch.clamp(center_poi_pos + pos_eps, min=0, max=1).squeeze(0),
            target_label = base_points.labels[i].item(),
            data_id=repaired_indices[i],
            status=RegionStatus.positive
        )
        # add spec
        num_classes = dnn(base_points.images[:1]).shape[1]  # assuming output shape is (1, num_classes)
        repaired_region.add_spec(num_classes, repaired_region.target_label, device=device, dtype=dtype)
        base_region.add_spec(num_classes, base_region.target_label, device=device, dtype=dtype)

        # check if base region does not violate the specification
        if args.repair_mode not in (RepairMode.PREPARED, RepairMode.LastLayer_ibp): 
            Pdata_id = base_region.data_id
            tmp_time = data[Pdata_id]['time']
            new_lb = data[Pdata_id]['lb']
            new_ub = data[Pdata_id]['ub']
            if new_lb is None or new_ub is None:
                raise ValueError(f"Safe radius not found for data id {Pdata_id}.")
            base_region.lb = torch.clamp(new_lb.to(device=device, dtype=dtype), min=0, max=1)
            base_region.ub = torch.clamp(new_ub.to(device=device, dtype=dtype), min=0, max=1)
            additional_time += tmp_time

        pair_list.append(RegionPair(repaired_regions=[repaired_region], base_region=base_region))

        if len(pair_list) >= total_num:
            break

    return pair_list, additional_time


def repair_regions_adv_and_perturbation(args, dnn, total_num):
    '''
    1. Generate adversarial examples from clean points based on the given:
        - target_label
        - inp_eps (strength of perturbation)
    2. Construct pairs of repaired regions and base regions
        - repaired region: centered at the generated adversarial example, with radius inp_eps
        - base region: centered at the corresponding clean point
            - but, the input region with a radius of inp_eps includes buggy region
            - so we search for the largest radius that does not violate the specification
    '''
    device = args.device
    dtype = args.dtype

    # safe radii
    data = torch.load(f"safe_radii/adv_{args.model_name}.pt")
    collect_num = total_num if len(data) >= total_num else len(data)

    additional_time = 0.0

    pair_list = []
    for i, (data_id, item) in enumerate(data.items()):
        if i >= collect_num:
            break
        # repaired region
        if args.repair_task == RepairTask.AdversarialAndPerturbation:
            center_poi_neg = item['adv_center'].unsqueeze(0).to(device=device, dtype=dtype)
        elif args.repair_task == RepairTask.LocalRobustness:
            center_poi_neg = item['center'].unsqueeze(0).to(device=device, dtype=dtype)
        repaired_region = Region(
            center_point = center_poi_neg,
            lb = torch.clamp(center_poi_neg - args.inp_eps, min=0, max=1).squeeze(0),
            ub = torch.clamp(center_poi_neg + args.inp_eps, min=0, max=1).squeeze(0),
            target_label = item['label'],
            data_id=data_id,
            status=RegionStatus.negative
        )
        # add spec
        num_classes = dnn(center_poi_neg).shape[1]  # assuming output shape is (1, num_classes)
        repaired_region.add_spec(num_classes, repaired_region.target_label, device=device, dtype=dtype)

        # base region
        center_poi_pos = item['center'].unsqueeze(0).to(device=device, dtype=dtype)
        if args.repair_mode not in (RepairMode.PREPARED, RepairMode.LastLayer_ibp):
            additional_time += item['time']
        lb = item['lb'].to(device=device, dtype=dtype)
        ub = item['ub'].to(device=device, dtype=dtype)
        base_region = Region(
            center_point = center_poi_pos,
            lb = torch.clamp(lb, min=0, max=1),
            ub = torch.clamp(ub, min=0, max=1),
            target_label = item['label'],
            data_id=data_id,
            status=RegionStatus.positive
        )
        # add spec
        base_region.add_spec(num_classes, base_region.target_label, device=device, dtype=dtype)

        pair_list.append(RegionPair(repaired_regions=[repaired_region], base_region=base_region))

    return pair_list, additional_time


def repair_regions_local_counterexample(args, dnn, total_num):
    '''
    1. find counterexample from the test set
    2. find the safe region with the center point
    3. construct repaired region based on the counterexample and the safe region
    '''
    device = args.device
    dtype = args.dtype
    clean_data = clean_points(args)
    pos_xs, neg_xs = split_by_misclassification(clean_data, dnn)

    # safe centers
    data = torch.load(f"safe_centers/{args.model_name}_num300.pt")
    if total_num > len(data):
        total_num = len(data)

    # # debug
    # for i in range(100):
    #     image = neg_xs.images[i]
    #     label = neg_xs.labels[i]
    #     pred = dnn(image.unsqueeze(0)).argmax(dim=1).item()
    #     print(f"Counterexample data id: {neg_xs.indices[i]}, label: {label}, pred: {pred}")

    additional_time = 0.0

    pair_list = []
    for i in range(min(total_num, len(neg_xs.images))):
        counterexample = neg_xs.images[i].unsqueeze(0).to(device=device, dtype=dtype)
        target_label = neg_xs.labels[i].item()

        # add perturbation based on the perturbation_pick setting
        Nlb, Nub = prepare_repair_box(counterexample, args.inp_eps, args.perturbation_pick, args.perturbation_ndim)

        repaired_region = Region(
            center_point=counterexample,
            lb=Nlb,
            ub=Nub,
            target_label=target_label,
            data_id=neg_xs.indices[i].item(),
            status=RegionStatus.negative
        )
        repaired_region.add_spec(num_classes=dnn(counterexample).shape[1], target_label=target_label, device=device, dtype=dtype)

        if args.repair_mode not in (RepairMode.PREPARED, RepairMode.LastLayer_ibp):
            x_id = neg_xs.indices[i].item()
            safe_center = data[x_id]['safe_center']
            radius = data[x_id]['radius']
            timecost = data[x_id]['time']
            additional_time += timecost
        else:  # for PREPARED and LastLayer_ibp, we don't use base_region
            safe_center = counterexample
            radius = args.inp_eps

        base_region = Region(
            center_point=safe_center,
            lb=torch.clamp(safe_center - radius, min=0, max=1).squeeze(0),
            ub=torch.clamp(safe_center + radius, min=0, max=1).squeeze(0),
            target_label=target_label,
            data_id=neg_xs.indices[i].item(),
            status=RegionStatus.positive
        )
        base_region.add_spec(num_classes=dnn(safe_center).shape[1], target_label=target_label, device=device, dtype=dtype)

        pair_list.append(RegionPair(repaired_regions=[repaired_region], base_region=base_region))

    return pair_list, additional_time


# def repair_regions_local_rbst(args, dnn):
#     '''
#     1. Generate adversarial examples from clean points based on the given:
#         - target_label
#         - inp_eps (strength of perturbation)
#     2. Construct repaired regions so that each repaired region contains an generated adversarial example
#     '''
#     device = args.device
#     dtype = args.dtype
#     clean_data = clean_points(args)
#     adv_data = adv_dataset(args, clean_data, dnn, eps=args.inp_eps, samples_num=args.num_v_polys, target_label=args.target_label)

#     region_list = []
#     for i in range(len(adv_data)):
#         center_poi = adv_data.ori_images[i].unsqueeze(0).to(device=device, dtype=dtype)
#         region = Region(
#             center_point=center_poi,
#             lb = (center_poi - args.inp_eps).squeeze(0),
#             ub = (center_poi + args.inp_eps).squeeze(0),
#             target_label = adv_data.labels[i].item(),
#             data_id = adv_data.indices[i].item(),
#             status=RegionStatus.negative
#         )

#         # add spec
#         num_classes = dnn(center_poi).shape[1]  # assuming output shape is (1, num_classes)
#         region.add_spec(num_classes, region.target_label, device=device, dtype=dtype)

#         region_list.append(region)

#     return [], region_list


# def repair_regions_local(args, normalize_input):
#     '''
#     returns
#         - pair_list: None
#         - region_list
#     '''
#     device = args.device
#     dtype = args.dtype
#     props = acas.properties.applicable_properties_for_model(args.acasxu_net_key, device=device, dtype=dtype)
    
#     region_list = []
#     for no, prop in props:
#         boxes = prop.get_input_polytopes(
#             normalize_input=normalize_input,
#             device=device, dtype=dtype
#         )
#         # check box num
#         if boxes.shape[0] == 2:
#             lb = boxes[0, :, 0]
#             ub = boxes[0, :, 1]
#             lb2 = boxes[1, :, 0]
#             ub2 = boxes[1, :, 1]
#         elif boxes.shape[0] == 1:
#             lb = boxes[0, :, 0]
#             ub = boxes[0, :, 1]
#             lb2 = None
#             ub2 = None
#         else: raise ValueError(f"Unexpected number of boxes: {boxes.shape[0]}")

#         region = Region(
#             lb=lb,
#             ub=ub,
#             lb2=lb2,
#             ub2=ub2,
#             data_id=no,
#             status=RegionStatus.unprocessed
#         )
#         # add spec
#         region.add_spec_acasxu(prop, device=device, dtype=dtype)
#         region_list.append(region)
#     return None, region_list


def repair_regions(args, dnn, total_num, normalize_input=None):
    '''
    returns
    - pair_list: list of RegionPair, where each RegionPair contains a repaired_region and a base_region
    - region_list: list of Region, where each Region is a repaired_region without a corresponding base_region
    '''
    if args.repair_task == RepairTask.CorruptionAndPerturbation:
        return repair_regions_corruption_and_perturbation(args, dnn, total_num)
    # elif args.repair_task == RepairTask.GlobalACASXu:
    #     return repair_regions_local(args, normalize_input=normalize_input)
    elif args.repair_task in ( RepairTask.AdversarialAndPerturbation, RepairTask.LocalRobustness ):
        return repair_regions_adv_and_perturbation(args, dnn, total_num)
    # elif args.repair_task == RepairTask.LocalRobustness:
    #     return repair_regions_local_rbst(args, dnn)
    elif args.repair_task == RepairTask.LocalCounterexample:
        return repair_regions_local_counterexample(args, dnn, total_num)
    else:
        raise NotImplementedError(f"Repair experiment {args.repair_task} is not implemented.")
    
def sample_annulus_points(center, inner_eps, outer_eps, num_points):
    """
    Sample points:
        inner_eps < ||x - center||_inf <= outer_eps
    """
    torch.manual_seed(0)  # for reproducibility
    device = center.device
    inp_shape = center.shape[1:]
    dim = center.numel()

    # flatten center
    center_flat = center.view(1, -1)

    # 1. random direction in [-1, 1]
    noise = torch.rand((num_points, dim), device=device) * 2 - 1  # [-1,1]

    # 2. normalize to L_inf = 1
    noise = noise / noise.abs().max(dim=1, keepdim=True)[0]

    # 3. sample radius
    r = torch.rand((num_points, 1), device=device) * (outer_eps - inner_eps) + inner_eps

    # 4. perturbation
    perturb = noise * r

    # 5. reshape back
    perturb = perturb.view(num_points, *inp_shape)

    sampled_points = center + perturb

    # 6. clamp
    sampled_points = torch.clamp(sampled_points, 0.0, 1.0)

    return sampled_points


def evaluation_data(args, repair_region_pairs):
    '''
    returns
    - acc_dataloader: dataloader for accuracy evaluation, containing clean test data
    - gen_dataloader: dataloader for generalization evaluation
    '''
    c_points = clean_points(args)
    acc_dataloader = c_points.dataloader(batch_size=100, shuffle=False)

    denom = 255.
    pixel_vals = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    gen_distance_list = [args.inp_eps + (v / denom) for v in pixel_vals]

    gen_dataloader_all = []
    data_num = 50  # todo: if you change this, also change the number in APRNN/eval_7_aprnn/gen_data()
    
    for pair in repair_region_pairs:
        Nregions = pair.repaired_regions
        for region in Nregions:
            gen_dataloader_list = []
            counter_x = region.center_point  # (1, input_dim)
            label = region.target_label
            data_id = region.data_id

            if args.repair_task in (RepairTask.CorruptionAndPerturbation, 
                                          RepairTask.LocalRobustness, 
                                          RepairTask.AdversarialAndPerturbation,
                                          RepairTask.LocalCounterexample):
                '''
                collect random points whose distance from the center point: args.inp_eps < distance ≤ gen_distance
                '''
                for gen_distance in gen_distance_list:
                    if gen_distance <= args.inp_eps:
                        continue
                    
                    # generate random points with random distance from args.inp_eps to gen_distance
                    random_points = sample_annulus_points(counter_x, args.inp_eps, gen_distance, data_num)

                    # create dataset and dataloader
                    gen_dataloader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(random_points, torch.full((data_num,), label, device=counter_x.device, dtype=torch.long)),
                        batch_size=100, shuffle=False
                    )
                    gen_dataloader_list.append((gen_distance, gen_dataloader))
            else:
                raise NotImplementedError(f"Repair experiment {args.repair_task} is not implemented for generalization evaluation.")
            
            gen_dataloader_all.append((data_id, gen_dataloader_list))

    return acc_dataloader, gen_dataloader_all