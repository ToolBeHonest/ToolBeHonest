import numpy as np
from itertools import chain
from tabulate import tabulate

subtask_groups = {
    "MNT": ["single_step", "multi_step_wo_rep", "multi_step_w_rep"],
    "PT": ["os", "web"],
    "LFT": ["iter", "best"]
}
subtask_groups["overall"] = list(chain.from_iterable(subtask_groups.values()))

def flatten(data):
    flat_list = []
    
    def _flatten(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            for sub_item in item:
                _flatten(sub_item)
        else:
            flat_list.append(item)
    
    _flatten(data)
    return flat_list

def calculate_metrics(metrics, task, level):
    if level == "level_1":
        return np.mean([m["unsolvable"] for m in metrics[task][level]])
    elif level == "level_2":
        return np.mean([m["unsolvable"]["progress_rate"] for m in metrics[task][level]])
    elif level == "level_3":
        progress_rate = np.mean([m["unsolvable"]["progress_rate"] for m in metrics[task][level]])
        scorers = np.mean([
            np.mean(list(chain.from_iterable([item] if not isinstance(item, list) else item for item in m["unsolvable"]["scorers"])))
            for m in metrics[task][level]
        ])
        return {"progress_rate": progress_rate, "scorers": scorers}
    return None

def calculate_group_metrics(metrics, tasks, level):
    if level == "level_1":
        return [m["unsolvable"] for task in tasks for m in metrics[task][level]]
    elif level == "level_2":
        return [m["unsolvable"]["progress_rate"] for task in tasks for m in metrics[task][level]]
    elif level == "level_3":
        progress_rate = [m["unsolvable"]["progress_rate"] for task in tasks for m in metrics[task][level]]
        scorers = [
            np.mean(list(chain.from_iterable([item] if not isinstance(item, list) else item for item in m["unsolvable"]["scorers"])))
            for task in tasks for m in metrics[task][level]
        ]
        return {"progress_rate": progress_rate, "scorers": scorers}
    return []

def calculate_group_metrics_embedding(metrics, tasks, level):
    if level == "level_1":
        return np.mean([m["unsolvable"] for task in tasks for m in metrics[task][level]])
    elif level == "level_2":
        return {"progress_rate": np.mean([m["unsolvable"]["progress_rate"] for task in tasks for m in metrics[task][level]])}
    elif level == "level_3":
        progress_rate = np.mean([m["unsolvable"]["progress_rate"] for task in tasks for m in metrics[task][level]])
        scorers = np.mean([
            np.mean(list(chain.from_iterable([item] if not isinstance(item, list) else item for item in m["unsolvable"]["scorers"])))
            for task in tasks for m in metrics[task][level]
        ])
        return {"progress_rate": progress_rate, "scorers": scorers}
    return None

def calculate_group_metrics_vs(metrics, tasks, level, solvability="solvable"):
    if level == "level_1":
        return [m[solvability] for task in tasks for m in metrics[task][level]]
    else:
        return [m[solvability]["progress_rate"] for task in tasks for m in metrics[task][level]]

def calculate_hallu_analysis(metrics, tasks, level):
    count_dict = {
        "non_existent_tools": 0, 
        "solvability_hallu": 0, 
        "wrong_tools": 0, 
        "correct": 0,
        "wrong_unsolvable_index": 0,
        "wrong_reasoning": 0
    }
    for condition in [m["unsolvable"]["condition"] for task in tasks for m in metrics[task][level]]:
        count_dict[condition] += 1
    return count_dict

def calculate_overall_level_metrics(metrics, tasks):
    all_scores = []

    for level in ["level_1", "level_2", "level_3"]:
        scores = calculate_group_metrics(metrics, tasks, level)
        if level != "level_3":
            all_scores.extend(scores)
        else:
            all_scores.extend(scores["progress_rate"])
            all_scores.extend(scores["scorers"])
    
    overall_avg = np.mean(all_scores)
    
    return overall_avg

def calculate_subtask_results(metrics):
    results_dict = {}
    for sub_task, task_metrics in metrics.items():
        task_dict = {}
        for level, _ in task_metrics.items():
            level_dict = {}
            level_results = calculate_metrics(metrics, sub_task, level)
            if isinstance(level_results, dict):
                level_dict.update({f"{k}": v for k, v in level_results.items()})
            else:
                level_dict = level_results
            task_dict[level] = level_dict
        results_dict[sub_task] = task_dict
    return results_dict
def calculate_group_results(metrics):
    results_dict = {}
    for group, subtasks in subtask_groups.items():
        group_dict = {}
        if group == "overall":
            overall_avg = calculate_overall_level_metrics(metrics, subtasks)
            group_dict["overall"] = overall_avg
        else:
            for level in ["level_1", "level_2", "level_3"]:
                if level == "level_1":
                    group_dict["Level 1 Exact Match"] = np.mean(calculate_group_metrics(metrics, subtasks, level))
                elif level == "level_2":
                    group_dict["Level 2 Progress Rate"] = np.mean(calculate_group_metrics(metrics, subtasks, level))
                elif level == "level_3":
                    result = calculate_group_metrics(metrics, subtasks, level)
                    group_dict["Level 3 Progress Rate"] = np.mean(result["progress_rate"])
                    group_dict["Level 3 Matching Score"] = np.mean(result["scorers"])
        
        results_dict[group] = group_dict
    return results_dict

def calculate_analysis_results(metrics):
    statistics = {}
    for group, subtasks in subtask_groups.items():
        for level in ["level_2", "level_3"]:
            count_dict = calculate_hallu_analysis(metrics, subtasks, level)
            level_name = level.replace("l", "L").replace("_", " ")
            group_name = group.replace("overall", "Overall")
            statistics[f"{level_name} / {group_name}"] = {}
            statistics[f"{level_name} / {group_name}"]["Non-existent Tools"] = count_dict["non_existent_tools"]
            statistics[f"{level_name} / {group_name}"]["Solvability Hallucination"] = count_dict["solvability_hallu"]
            statistics[f"{level_name} / {group_name}"]["Wrong Tools"] = count_dict["wrong_tools"]
            statistics[f"{level_name} / {group_name}"]["Wrong UnsolvableQuery Index"] = count_dict["wrong_unsolvable_index"]
            statistics[f"{level_name} / {group_name}"]["Wrong Tool Reasoning"] = count_dict["wrong_reasoning"]
            statistics[f"{level_name} / {group_name}"]["Correct"] = count_dict["correct"]
    return statistics

def print_table(sub_task_results_dict, group_results_dict, analysis_results_dict):
    sub_task_table = []
    for task, results in sub_task_results_dict.items():
        sub_task_table.append([task, 'Level 1 Exact Match', round(results['level_1'], 2)])
        sub_task_table.append([task, 'Level 2 Progress Rate', round(results['level_2'], 2)])
        sub_task_table.append([task, 'Level 3 Progress Rate', round(results['level_3']['progress_rate'], 2)])
        sub_task_table.append([task, 'Level 3 Matching Score', round(results['level_3']['scorers'], 2)])
    sub_task_table_print = tabulate(sub_task_table, headers=['Subtask', 'Metric', 'Score'], tablefmt='grid')
    
    group_table = []
    for group, results in group_results_dict.items():
        for metric, value in results.items():
            group_table.append([group, metric, round(value, 2)])
    group_table_print = tabulate(group_table, headers=['Scenario', 'Metric', 'Score'], tablefmt='grid')
    
    analysis_table = []
    for level, results in analysis_results_dict.items():
        for error_type, count in results.items():
            analysis_table.append([level, error_type, count])
    analysis_table_print = tabulate(analysis_table, headers=['Level / Scenario', 'Metric', 'Count'], tablefmt='grid')

    return sub_task_table_print, group_table_print, analysis_table_print