
import torch

from network_bound.bounder import IndividualBounds
from network_bound.utill import filter_violated_spec, update_violated_mask
from LPsolver.solver import LPSolver
from .logging import logging


def get_concrete_bounds(
        net, in_lb, in_ub,
        optimize_alpha=True,
        alpha_steps=20,
        alpha_lr=1e-1,
        save_coeffs=False,
        verbose=False,
        ):
    """ returns concrete lower and upper bounds """

    bounder = IndividualBounds(
        net = net,
        lb_inp = in_lb,
        ub_inp = in_ub,
        device = in_lb.device,
    )

    lbs, ubs = bounder.run(
        optimize_alpha=optimize_alpha,
        alpha_steps=alpha_steps,
        alpha_lr=alpha_lr,
        save_coeffs=save_coeffs,
        verbose=verbose
    )

    return lbs, ubs, bounder


# === bound lp ===
def bounds_lp(net, lb, ub, spec, logfile=None):
    # get bounds from backsubstitution
    
    lbs, ubs, bounder = get_concrete_bounds(net, lb, ub)
    out_lb, out_ub = lbs[-1], ubs[-1]
    violated_mask, spec_lb = filter_violated_spec(
                C=spec.C,
                out_lb=out_lb,
                out_ub=out_ub,
            )
    if not violated_mask.any():
        logging.info("Skip LP check since no violation detected by backsubstitution.")
        del bounder
        return

    # lp
    C_lp = spec.C[violated_mask]
    solver = LPSolver(enable_repair=False)
    model, vars = solver.build_net(net, lb, ub, bounder)
    for i in range(C_lp.shape[0]):
        c = C_lp[i]
        c_np = c.detach().cpu().numpy()
        out_var = vars[-1]
        obj_expr = c_np @ out_var
        c_lb = solver.minimize(obj_expr)
        print(f"spec dimension {i}: c_lb = {c_lb}")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"    spec dimension {i}: c_lb = {c_lb}\n")
    solver.dispose()
    del solver, bounder


# === violation check ===

def check_violation(net, lb, ub, spec, lp_check=False):
    '''
    returns:
        is_violated: bool
    '''
    # approximate checking by backsubstitution and dual optimization

    # --- check violation by backsubstitution bounds ---
    lbs, ubs, bounder = get_concrete_bounds(net, lb, ub)
    out_lb, out_ub = lbs[-1], ubs[-1]
    ini_C = spec.C
    violated_mask_back, spec_lb = filter_violated_spec(
                C=ini_C,
                out_lb=out_lb,
                out_ub=out_ub,
            )
    
    if not violated_mask_back.any():
        # no violation
        if lp_check:
            return False, None, bounder
        else:
            del bounder
            return False
    else:
        is_violated = violated_mask_back.any()
        if lp_check:
            return is_violated, violated_mask_back, bounder
        else:
            del bounder
            return is_violated

def check_violation_lp(net, lb, ub, spec, logfile=None, debug=True):
    '''
    return is_violated: bool
    '''
    is_violated, violated_mask, bounder = check_violation(net, lb, ub, spec, lp_check=True)
    if not is_violated:
        print("No violation detected by backsubstitution or dual optimization.")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write("  No violation detected by backsubstitution or dual optimization.\n")
        return False
    
    violated_dim = violated_mask.nonzero(as_tuple=True)[0]
    print(f"Potential violation detected in spec dimensions: {violated_dim.tolist()} by backsubstitution/dual optimization. Checking with LP solver...")
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(f"  Potential violation detected in spec dimensions: {violated_dim.tolist()} by backsubstitution/dual optimization. Checking with LP solver...\n")
    
    # lp check
    solver = LPSolver(enable_repair=False)
    model, vars = solver.build_net(net, lb, ub, bounder)
    is_violated = False
    for dim in violated_dim:
        c = spec.C[dim.item()]
        c_np = c.detach().cpu().numpy()
        out_var = vars[-1]
        obj_expr = c_np @ out_var
        c_lb = solver.minimize(obj_expr)
        if c_lb < 0:
            if debug:
                print(f"dimension {dim.item()} is violated with LP check: c_lb = {c_lb}")
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"    spec dimension {dim.item()}: c_lb = {c_lb}\n")
                is_violated = True
                continue
            else:
                is_violated = True
                solver.dispose()
                del solver
                return is_violated
        else:
            if debug:
                print(f"dimension {dim.item()} is NOT violated with LP check: c_lb = {c_lb}")
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"    spec dimension {dim.item()}: c_lb = {c_lb}\n")
    solver.dispose()
    del solver
    return is_violated