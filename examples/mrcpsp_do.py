from typing import Tuple, Dict, NamedTuple, Optional

import numpy as np
from cpmpy.solvers.solver_interface import ExitStatus

from cpmpy.solvers import CPM_ortools
from cpmpy import cpm_array, intvar, Model, Cumulative, boolvar
import cpmpy as cp
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution, logger, logging
from discrete_optimization.rcpsp.rcpsp_utils import plot_task_gantt, plot_ressource_view, plt
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from cpmpy.tools.mus import mus
from cpmpy.tools.explain.mcs import mcs
logger.setLevel(logging.DEBUG)


class ConfigHardSoft(NamedTuple):
    resource_renewable: str
    resource_non_renewable: str
    precedence: str
    modes_alloc: str
    modes_to_resource: str
    modes_to_duration: str
    makespan: str


def model_mrcpsp(rcpsp_model: RCPSPModel, config: Optional[ConfigHardSoft] = None) -> Tuple[cp.Model, Dict]:
    if config is None:
        config = ConfigHardSoft(resource_renewable="soft",
                                resource_non_renewable="hard",
                                precedence="hard",
                                modes_alloc="hard",
                                modes_to_duration="hard",
                                modes_to_resource="hard", makespan="hard")
    all_modes = [(t, rcpsp_model.index_task[t], m, rcpsp_model.mode_details[t][m])
                 for t in rcpsp_model.tasks_list for m in rcpsp_model.mode_details[t]]
    mode_index_per_task = {}
    for i in range(len(all_modes)):
        if all_modes[i][1] not in mode_index_per_task:
            mode_index_per_task[all_modes[i][1]] = set()
        mode_index_per_task[all_modes[i][1]].add(i)
    resource_capacities = cpm_array([rcpsp_model.get_max_resource_capacity(r)
                                     for r in rcpsp_model.resources_list])
    res_needs = np.array([[rcpsp_model.mode_details[x[0]][x[2]].get(r, 0)
                          for r in rcpsp_model.resources_list]
                         for x in all_modes])
    resource_needs_modes = cpm_array(res_needs)
    non_renewable = {r: r in rcpsp_model.non_renewable_resources for r in rcpsp_model.resources_list}
    duration_modes = cpm_array([rcpsp_model.mode_details[x[0]][x[2]]["duration"]
                                for x in all_modes])
    max_res_need = np.max(res_needs)
    max_duration = max(duration_modes)
    nb_jobs = rcpsp_model.n_jobs
    nb_modes = len(all_modes)
    nb_res = len(rcpsp_model.resources_list)
    ub_makespan = sum([max([rcpsp_model.mode_details[t][m]["duration"]
                            for m in rcpsp_model.mode_details[t]])
                       for t in rcpsp_model.tasks_list])
    starts = intvar(lb=0, ub=ub_makespan, shape=nb_jobs, name="start")
    ends = intvar(lb=0, ub=ub_makespan, shape=nb_jobs, name="end")
    durations = intvar(lb=0, ub=max_duration, shape=nb_jobs, name="duration")
    resource_consumption = intvar(lb=0, ub=max_res_need, shape=(nb_jobs, nb_res),
                                  name="res_consumption")
    modes_activated = boolvar(shape=nb_modes, name="modes")
    hard, soft = [], []
    model = Model()
    for i in range(nb_modes):
        i_task = all_modes[i][1]
        model += modes_activated[i].implies(durations[i_task] == duration_modes[i])
        eval(config.modes_to_duration).append(model.constraints[-1])
        model.constraints[-1].set_description(f"Duration of task {i_task}"
                                              f" when modes{i} is activated")
        model += modes_activated[i].implies(cp.all(resource_consumption[i_task, :] == resource_needs_modes[i, :]))
        eval(config.modes_to_resource).append(model.constraints[-1])
        model.constraints[-1].set_description(f"constraint res consumption of task {i_task}"
                                              f" when modes{i} is activated")
    for i_task in mode_index_per_task:
        # 1 mode per task
        model += cp.any([modes_activated[x] for x in mode_index_per_task[i_task]])
        eval(config.modes_alloc).append(model.constraints[-1])
        model.constraints[-1].set_description(f"At least one mode for task {i_task}")
        model += cp.sum([modes_activated[x] for x in mode_index_per_task[i_task]]) == 1
        model.constraints[-1].set_description(f"Exactly one mode for task {i_task}")
        eval(config.modes_alloc).append(model.constraints[-1])
    model += ends == starts+durations
    hard.append(model.constraints[-1])
    # Precedence constraints
    for t in rcpsp_model.successors:
        i_t = rcpsp_model.index_task[t]
        for succ in rcpsp_model.successors[t]:
            i_succ = rcpsp_model.index_task[succ]
            model += starts[i_succ] >= ends[i_t]
            eval(config.precedence).append(model.constraints[-1])
            model.constraints[-1].set_description(f"{i_succ} should be done after {i_t}")
    for r in range(nb_res):
        if non_renewable[rcpsp_model.resources_list[r]]:
            model += cp.sum(resource_consumption[:, r]) <= resource_capacities[r]
            eval(config.resource_non_renewable).append(model.constraints[-1])
            model.constraints[-1].set_description(f"Non renewable resource {rcpsp_model.resources_list[r]} capacity respected")
        else:
            model += Cumulative(start=starts,
                                duration=durations,
                                end=ends,
                                demand=resource_consumption[:, r],
                                capacity=resource_capacities[r])
            eval(config.resource_renewable).append(model.constraints[-1])
            model.constraints[-1].set_description(f"Renewable resource {rcpsp_model.resources_list[r]} capacity respected")
    makespan = starts[rcpsp_model.index_task[rcpsp_model.sink_task]]
    model.minimize(makespan)
    return model, {"starts": starts, "ends": ends, "modes": modes_activated,
                   "all_modes": all_modes, "makespan": makespan,
                   "hard": hard, "soft": soft}


def model_mrcpsp_slack(rcpsp_model: RCPSPModel, config: Optional[ConfigHardSoft] = None) -> Tuple[cp.Model, Dict]:
    if config is None:
        config = ConfigHardSoft(resource_renewable="soft",
                                resource_non_renewable="hard",
                                precedence="hard",
                                modes_alloc="hard",
                                modes_to_duration="hard",
                                modes_to_resource="hard",
                                makespan="hard")
    all_modes = [(t, rcpsp_model.index_task[t], m, rcpsp_model.mode_details[t][m])
                 for t in rcpsp_model.tasks_list for m in rcpsp_model.mode_details[t]]
    mode_index_per_task = {}
    for i in range(len(all_modes)):
        if all_modes[i][1] not in mode_index_per_task:
            mode_index_per_task[all_modes[i][1]] = set()
        mode_index_per_task[all_modes[i][1]].add(i)
    resource_capacities = cpm_array([rcpsp_model.get_max_resource_capacity(r)
                                     for r in rcpsp_model.resources_list])
    res_needs = np.array([[rcpsp_model.mode_details[x[0]][x[2]].get(r, 0)
                          for r in rcpsp_model.resources_list]
                         for x in all_modes])
    resource_needs_modes = cpm_array(res_needs)
    non_renewable = {r: r in rcpsp_model.non_renewable_resources for r in rcpsp_model.resources_list}
    duration_modes = cpm_array([rcpsp_model.mode_details[x[0]][x[2]]["duration"]
                                for x in all_modes])
    max_res_need = np.max(res_needs)
    max_duration = max(duration_modes)
    nb_jobs = rcpsp_model.n_jobs
    nb_modes = len(all_modes)
    nb_res = len(rcpsp_model.resources_list)
    ub_makespan = sum([max([rcpsp_model.mode_details[t][m]["duration"]
                            for m in rcpsp_model.mode_details[t]])
                       for t in rcpsp_model.tasks_list])
    starts = intvar(lb=0, ub=ub_makespan, shape=nb_jobs, name="start")
    ends = intvar(lb=0, ub=ub_makespan, shape=nb_jobs, name="end")
    durations = intvar(lb=0, ub=max_duration, shape=nb_jobs, name="duration")
    resource_consumption = intvar(lb=0, ub=max_res_need, shape=(nb_jobs, nb_res),
                                  name="res_consumption")
    modes_activated = boolvar(shape=nb_modes, name="modes")
    hard, soft = [], []
    model = Model()
    for i in range(nb_modes):
        i_task = all_modes[i][1]
        model += modes_activated[i].implies(durations[i_task] == duration_modes[i])
        eval(config.modes_to_duration).append(model.constraints[-1])
        model.constraints[-1].set_description(f"Duration of task {i_task}"
                                              f" when modes{i} is activated")
        model += modes_activated[i].implies(cp.all(resource_consumption[i_task, :] == resource_needs_modes[i, :]))
        eval(config.modes_to_resource).append(model.constraints[-1])
        model.constraints[-1].set_description(f"constraint res consumption of task {i_task}"
                                              f" when modes{i} is activated")
    for i_task in mode_index_per_task:
        # 1 mode per task
        model += cp.any([modes_activated[x] for x in mode_index_per_task[i_task]])
        eval(config.modes_alloc).append(model.constraints[-1])
        model.constraints[-1].set_description(f"At least one mode for task {i_task}")
        model += cp.sum([modes_activated[x] for x in mode_index_per_task[i_task]]) == 1
        model.constraints[-1].set_description(f"Exactly one mode for task {i_task}")
        eval(config.modes_alloc).append(model.constraints[-1])
    model += ends == starts+durations
    hard.append(model.constraints[-1])
    # Precedence constraints
    slack_precedence = []
    for t in rcpsp_model.successors:
        i_t = rcpsp_model.index_task[t]
        for succ in rcpsp_model.successors[t]:
            if config.precedence == "soft":
                slack_precedence.append(intvar(lb=0, ub=2,
                                               name=f"slack_succ_{t,succ}"))
            i_succ = rcpsp_model.index_task[succ]
            if config.precedence == "soft":
                model += starts[i_succ] >= ends[i_t]-slack_precedence[-1]
            else:
                model += starts[i_succ] >= ends[i_t]
            eval(config.precedence).append(model.constraints[-1])
            model.constraints[-1].set_description(f"{i_succ} should be done after {i_t}")
    slack_resources = []
    for r in range(nb_res):
        if non_renewable[rcpsp_model.resources_list[r]]:
            if config.resource_non_renewable == "soft":
                slack_resources.append(intvar(0, 10, name=f"slack_{rcpsp_model.resources_list[r]}"))
                model += cp.sum(resource_consumption[:, r]) <= resource_capacities[r]+slack_resources[-1]
            else:
                model += cp.sum(resource_consumption[:, r]) <= resource_capacities[r]
                eval(config.resource_non_renewable).append(model.constraints[-1])
                model.constraints[-1].set_description(f"Non renewable resource {rcpsp_model.resources_list[r]} capacity respected")
        else:
            if config.resource_renewable == "soft":
                slack_resources.append(intvar(0, 10, name=f"slack_{rcpsp_model.resources_list[r]}"))
                model += Cumulative(start=starts,
                                    duration=durations,
                                    end=ends,
                                    demand=resource_consumption[:, r],
                                    capacity=resource_capacities[r]+slack_resources[-1])
            else:
                model += Cumulative(start=starts,
                                    duration=durations,
                                    end=ends,
                                    demand=resource_consumption[:, r],
                                    capacity=resource_capacities[r])
            eval(config.resource_renewable).append(model.constraints[-1])
            model.constraints[-1].set_description(f"Renewable resource {rcpsp_model.resources_list[r]} capacity respected")
    makespan = starts[rcpsp_model.index_task[rcpsp_model.sink_task]]
    objs = [makespan]
    if config.precedence == "soft":
        objs += [cp.sum(slack_precedence)]
    if config.resource_non_renewable == "soft" or config.resource_renewable == "soft":
        objs += [cp.sum(slack_resources)]
    model.minimize(sum(objs))
    return model, {"starts": starts, "ends": ends, "modes": modes_activated,
                   "all_modes": all_modes, "makespan": makespan,
                   "hard": hard, "soft": soft,
                   "slack_precedence": slack_precedence,
                   "slack_resources": slack_resources}


def solve_mrcpsp():
    """Solve original MRCPSP problem"""
    file = [f for f in get_data_available() if "j1010_10.mm" in f][0]
    rcpsp_model = parse_file(file)
    model, data = model_mrcpsp(rcpsp_model=rcpsp_model)
    solver = CPM_ortools(model)
    solver.solve(time_limit=10, log_search_progress=True)
    modes_dict = {}
    schedule = {t: {"start_time": 0, "end_time": 0}
                for t in rcpsp_model.tasks_list}
    for i in range(data["starts"].shape[0]):
        t = rcpsp_model.tasks_list[i]
        schedule[t]["start_time"] = int(data["starts"][i].value())
        schedule[t]["end_time"] = int(data["ends"][i].value())
    for i in range(len(data["all_modes"])):
        t = data["all_modes"][i][0]
        m = data["all_modes"][i][2]
        if data["modes"][i].value():
            modes_dict[t] = m
    print("Modes : ", modes_dict)
    modes = [modes_dict[t] for t in rcpsp_model.tasks_list_non_dummy]
    solution = RCPSPSolution(problem=rcpsp_model,
                             rcpsp_schedule=schedule,
                             rcpsp_modes=modes)
    rcpsp_model.satisfy(solution)
    print(rcpsp_model.satisfy(solution), rcpsp_model.evaluate(solution))
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plt.show()


def solve_mrcpsp_inf():
    """Create artificial infeasible instance, compute mus and mcs"""
    file = [f for f in get_data_available() if "j1010_10.mm" in f][0]
    rcpsp_model = parse_file(file)
    config = ConfigHardSoft(resource_renewable="soft",
                            resource_non_renewable="soft",
                            precedence="hard",
                            modes_alloc="hard",
                            modes_to_duration="hard",
                            modes_to_resource="hard",
                            makespan="hard")
    model, data = model_mrcpsp(rcpsp_model=rcpsp_model,
                               config=config)
    model += data["makespan"] <= 16  # 17 is optimal
    data[config.makespan].append(model.constraints[-1])
    solver = CPM_ortools(model)
    solver.solve(time_limit=10, log_search_progress=True)
    assert solver.status().exitstatus == ExitStatus.UNSATISFIABLE

    conflicts = mus(soft=data["soft"], hard=data["hard"], solver="ortools")
    print('---Conflict---')
    for c in conflicts:
        print(c)
    corrections = mcs(soft=data["soft"], hard=data["hard"], solver="ortools")
    print('---Correction subset---')
    for c in corrections:
        print(c)
    new_model = Model([c for c in model.constraints if c not in corrections])
    new_model.minimize(data["makespan"])
    new_model.solve(solver="ortools")
    modes_dict = {}
    schedule = {t: {"start_time": 0, "end_time": 0}
                for t in rcpsp_model.tasks_list}
    for i in range(data["starts"].shape[0]):
        t = rcpsp_model.tasks_list[i]
        schedule[t]["start_time"] = int(data["starts"][i].value())
        schedule[t]["end_time"] = int(data["ends"][i].value())
    for i in range(len(data["all_modes"])):
        t = data["all_modes"][i][0]
        m = data["all_modes"][i][2]
        if data["modes"][i].value():
            modes_dict[t] = m
    print("Modes : ", modes_dict)
    modes = [modes_dict[t] for t in rcpsp_model.tasks_list_non_dummy]
    solution = RCPSPSolution(problem=rcpsp_model,
                             rcpsp_schedule=schedule,
                             rcpsp_modes=modes)
    rcpsp_model.satisfy(solution)
    print(rcpsp_model.satisfy(solution), rcpsp_model.evaluate(solution))
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plt.show()


def mrcpsp_slack():
    file = [f for f in get_data_available() if "j1010_10.mm" in f][0]
    rcpsp_model = parse_file(file)
    config = ConfigHardSoft(resource_renewable="soft",
                            resource_non_renewable="soft",
                            precedence="soft",
                            modes_alloc="hard",
                            modes_to_duration="hard",
                            modes_to_resource="hard",
                            makespan="hard")
    model, data = model_mrcpsp_slack(rcpsp_model=rcpsp_model,
                                     config=config)
    model += data["makespan"] <= 16  # 17 is optimal
    model.solve(solver="ortools", time_limit=10)
    print("Obj value : ", model.objective_.value())
    print("slack precedence", [x.value() for x in data["slack_precedence"]])
    print("slack res", [x.value() for x in data["slack_resources"]])
    modes_dict = {}
    schedule = {t: {"start_time": 0, "end_time": 0}
                for t in rcpsp_model.tasks_list}
    for i in range(data["starts"].shape[0]):
        t = rcpsp_model.tasks_list[i]
        schedule[t]["start_time"] = int(data["starts"][i].value())
        schedule[t]["end_time"] = int(data["ends"][i].value())
    for i in range(len(data["all_modes"])):
        t = data["all_modes"][i][0]
        m = data["all_modes"][i][2]
        if data["modes"][i].value():
            modes_dict[t] = m
    print("Modes : ", modes_dict)
    modes = [modes_dict[t] for t in rcpsp_model.tasks_list_non_dummy]
    solution = RCPSPSolution(problem=rcpsp_model,
                             rcpsp_schedule=schedule,
                             rcpsp_modes=modes)
    rcpsp_model.satisfy(solution)
    print(rcpsp_model.satisfy(solution), rcpsp_model.evaluate(solution))
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plt.show()


if __name__ == "__main__":
    mrcpsp_slack()