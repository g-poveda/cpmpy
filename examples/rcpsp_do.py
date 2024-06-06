import os

from cpmpy.solvers import CPM_ortools
from cpmpy import cpm_array, intvar, Model, Cumulative
import cpmpy as cp
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution, logger, logging
from discrete_optimization.rcpsp.rcpsp_utils import plot_task_gantt, plot_ressource_view, plt
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from cpmpy.tools.mus import mus
from cpmpy.tools.explain.mcs import mcs
logger.setLevel(logging.DEBUG)


def inf_rcpsp():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    rcpsp_model: RCPSPModel = parse_file(file)
    durations = cpm_array([rcpsp_model.mode_details[t][1]["duration"]
                           for t in rcpsp_model.tasks_list])
    resource_needs = cpm_array(
        [[rcpsp_model.mode_details[t][1].get(r, 0) for r in rcpsp_model.resources_list]
         for t in rcpsp_model.tasks_list])
    resource_capacities = cpm_array([rcpsp_model.get_max_resource_capacity(r)
                                     for r in rcpsp_model.resources_list])
    nb_resource = len(rcpsp_model.resources_list)
    nb_jobs = len(durations)
    max_duration = sum(durations)  # dummy upper bound, could be improved
    # Variables
    start_time = intvar(0, max_duration, shape=nb_jobs, name="start")
    end_time = intvar(0, max_duration, shape=nb_jobs, name="end")
    # need to include it so that the mus function works
    model = Model()
    config_constr = {"resource": "soft",
                     "precedence": "hard",
                     "makespan": "hard"}
    soft, hard = [], []
    model += end_time == start_time+durations
    hard.append(model.constraints[-1])
    # Precedence constraints
    for t in rcpsp_model.successors:
        i_t = rcpsp_model.index_task[t]
        for succ in rcpsp_model.successors[t]:
            i_succ = rcpsp_model.index_task[succ]
            model += start_time[i_succ] >= end_time[i_t]
            eval(config_constr["precedence"]).append(model.constraints[-1])
            model.constraints[-1].set_description(f"{succ} has to be done after {t}")
    # Cumulative resource constraint
    for r in range(nb_resource):
        model += Cumulative(start=start_time,
                            duration=durations,
                            end=end_time,
                            demand=resource_needs[:, r],
                            capacity=resource_capacities[r])
        eval(config_constr["resource"]).append(model.constraints[-1])
        model.constraints[-1].set_description(f"Cumulative constraint {r},"
                                              f" of capacity {resource_capacities[r]}")
    makespan = start_time[rcpsp_model.index_task[rcpsp_model.sink_task]]
    model.minimize(makespan)
    model += makespan <= 42  # 43 is the optimal makespan
    eval(config_constr["makespan"]).append(model.constraints[-1])
    conflicts = mus(soft=soft, hard=hard, solver="exact")
    print('---Conflict---')
    for c in conflicts:
        print(c)
    corrections = mcs(soft=soft, hard=hard, solver="exact")
    print('---Correction subset---')
    for c in corrections:
        print(c)
    new_model = Model([c for c in model.constraints if c not in corrections])
    new_model.solve(solver="ortools")
    solution = RCPSPSolution(problem=rcpsp_model,
                             rcpsp_schedule={rcpsp_model.tasks_list[i]:
                                                 {"start_time": int(start_time[i].value()),
                                                  "end_time": int(end_time[i].value())}
                                             for i in range(rcpsp_model.n_jobs)})
    rcpsp_model.satisfy(solution)
    print(rcpsp_model.satisfy(solution), rcpsp_model.evaluate(solution))
    print("Makespan when relaxed constraint :", makespan.value())
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plt.show()


def rcpsp_slack():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    rcpsp_model: RCPSPModel = parse_file(file)
    durations = cpm_array([rcpsp_model.mode_details[t][1]["duration"]
                           for t in rcpsp_model.tasks_list])
    resource_needs = cpm_array(
        [[rcpsp_model.mode_details[t][1].get(r, 0) for r in rcpsp_model.resources_list]
         for t in rcpsp_model.tasks_list])
    resource_capacities = cpm_array([rcpsp_model.get_max_resource_capacity(r)
                                     for r in rcpsp_model.resources_list])
    slack_resource = intvar(lb=0, ub=2, shape=resource_capacities.shape,
                            name="slack_res")
    nb_resource = len(rcpsp_model.resources_list)
    nb_jobs = len(durations)
    max_duration = sum(durations)  # dummy upper bound, could be improved
    # Variables
    start_time = intvar(0, max_duration, shape=nb_jobs, name="start")
    end_time = intvar(0, max_duration, shape=nb_jobs, name="end")
    # need to include it so that the mus function works
    model = Model()
    soft, hard = [], []
    model += end_time == start_time+durations
    # Precedence constraints
    successor = cpm_array([[rcpsp_model.index_task[t], rcpsp_model.index_task[succ]]
                          for t in rcpsp_model.successors for succ in rcpsp_model.successors[t]])
    slack_succ = intvar(0, 10, shape=(successor.shape[0],), name="slack_succ_constraint")
    for i in range(successor.shape[0]):
        model += start_time[successor[i, 1]] >= end_time[successor[i, 0]]-slack_succ[i]
    # Cumulative resource constraint
    for r in range(nb_resource):
        model += Cumulative(start=start_time,
                            duration=durations,
                            end=end_time,
                            demand=resource_needs[:, r],
                            capacity=resource_capacities[r]+slack_resource[r])
        model.constraints[-1].set_description(f"Cumulative constraint {r},"
                                              f" of capacity {resource_capacities[r]}")
    makespan = start_time[rcpsp_model.index_task[rcpsp_model.sink_task]]
    model.minimize(10*cp.sum(slack_resource)+cp.sum(slack_succ))
    model += makespan <= 42  # 43 is the optimal makespan
    # model.solve(solver="ortools")
    solver = CPM_ortools(model)
    solver.solve(time_limit=10, log_search_progress=True)
    solution = RCPSPSolution(problem=rcpsp_model,
                             rcpsp_schedule={rcpsp_model.tasks_list[i]:
                                                 {"start_time": int(start_time[i].value()),
                                                  "end_time": int(end_time[i].value())}
                                             for i in range(rcpsp_model.n_jobs)})
    rcpsp_model.satisfy(solution)
    print(rcpsp_model.satisfy(solution), rcpsp_model.evaluate(solution))
    print("Makespan when relaxed constraint :", makespan.value())
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=solution)
    plt.show()


if __name__ == "__main__":
    rcpsp_slack()