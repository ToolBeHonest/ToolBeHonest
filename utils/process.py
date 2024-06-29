import re
from tqdm import tqdm
from utils.evaluation_func import _evaluate_detecting, _evaluate_planning, _evaluate_diagnosing, \
                            _evaluate_planning_analysis, _evaluate_diagnosing_analysis
from utils.extract_func import _extract_subgoal_plantool_del_lastfinish, remove_finish_if_last


def convert_hf_data(items):
    tasks_dict = {}
    for item in items:
        subtask = item["subtask"]
        if subtask not in tasks_dict:
            tasks_dict[subtask] = []
        tasks_dict[subtask].append(item)
    return tasks_dict

def _process_task_infer(prompt_type, task_item, generation_func, args):
    task_query = task_item["task"]
    tool_list_str = task_item["tools"]
    unsolvable_task_query = task_item["unsolvable_task"]
    unsolvable_tool_list_str = task_item["unsolvable_tools"]
    
    if prompt_type == "level_1":
        prompt = f"{args.detecting_prompt}\n\n<task>\n{task_query}\n</task>\n<provided_tools>\n{tool_list_str}\n</provided_tools>"
        response = generation_func(prompt)

        unsolvable_prompt = f"{args.detecting_prompt}\n\n<task>\n{unsolvable_task_query}\n</task>\n\n<provided_tools>\n{unsolvable_tool_list_str}\n</provided_tools>"
        unsolvable_response = generation_func(unsolvable_prompt, args.api_url)

        return {"solvable": response, "unsolvable": unsolvable_response}

    elif prompt_type == "level_2":
        prompt = f"{args.planning_prompt}\n\n<task>\n{task_query}\n</task>\n<provided_tools>\n{tool_list_str}\n</provided_tools>"
        response = generation_func(prompt)
        
        unsolvable_prompt = f"{args.planning_prompt}\n\n<task>\n{unsolvable_task_query}\n</task>\n<provided_tools>\n{unsolvable_tool_list_str}\n</provided_tools>"
        unsolvable_response = generation_func(unsolvable_prompt, args.api_url)

        return {"solvable": response, "unsolvable": unsolvable_response}

    elif prompt_type == "level_3":
        prompt = f"{args.diagnosing_prompt}\n\n<task>\n{task_query}\n</task>\n<provided_tools>\n{tool_list_str}\n</provided_tools>"
        response = generation_func(prompt)

        unsolvable_prompt = f"{args.diagnosing_prompt}\n\n<task>\n{unsolvable_task_query}\n</task>\n<provided_tools>\n{unsolvable_tool_list_str}\n</provided_tools>"
        unsolvable_response = generation_func(unsolvable_prompt, args.api_url)

        return {"solvable": response, "unsolvable": unsolvable_response}

def _process_task_eval(prompt_type, task_item, task, args, 
                    result_task):
    answer_pattern = args.answer_pattern
    tool_list_str = task_item["tools"]
    
    provided_tool_dict = {"UnsolvableQuery": "Used to determine that the Query or Subgoal can not be completed"}
    pattern = re.compile(r'\d+\.\s*([^\:：]+)[:：]\s*(.*)')
    matches = pattern.findall(tool_list_str)
    
    for match in matches:
        tool_name, tool_desc = match
        provided_tool_dict[tool_name] = tool_desc
    
    if prompt_type == "level_1":
        response = result_task["solvable"]

        unsolvable_response = result_task["unsolvable"]

        match = re.search(answer_pattern, response)
        unsolvable_match = re.search(answer_pattern, unsolvable_response)
        if match:
            solvable_metric = _evaluate_detecting(match.group(1).strip(), "solvable")
        else:
            solvable_metric = _evaluate_detecting("", "solvable")
            
        if unsolvable_match:
            unsolvable_metric = _evaluate_detecting(unsolvable_match.group(1).strip(), "unsolvable")
        else:
            unsolvable_metric = _evaluate_detecting("", "unsolvable")
        
        return {"solvable": solvable_metric, "unsolvable": unsolvable_metric}

    elif prompt_type == "level_2":
        response = result_task["solvable"]

        unsolvable_response = result_task["unsolvable"]
        
        match = re.search(answer_pattern, response, re.DOTALL)
        unsolvable_match = re.search(answer_pattern, unsolvable_response, re.DOTALL)
        if match:
            solvable_tools = match.group(1).strip().split("\n")
            solvable_metric = _evaluate_planning(
                remove_finish_if_last(solvable_tools), 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.calculate_type
            )
        else:
            solvable_metric = _evaluate_planning(
                [""], 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.calculate_type
            )
            
        if unsolvable_match:
            unsolvable_tools = unsolvable_match.group(1).strip().split("\n")
            unsolvable_metric, unsolvable_condition = _evaluate_planning_analysis(
                remove_finish_if_last(unsolvable_tools), 
                remove_finish_if_last(task_item["planning_tools_unsolvable"]),
                [k for k, _ in provided_tool_dict.items()],
                args.calculate_type
            )
        else:
            unsolvable_metric, unsolvable_condition = _evaluate_planning_analysis(
                [""], 
                remove_finish_if_last(task_item["planning_tools_unsolvable"]),
                [k for k, _ in provided_tool_dict.items()],
                args.calculate_type
            )

        return {"solvable": {"progress_rate": solvable_metric}, \
            "unsolvable": {"progress_rate": unsolvable_metric, "condition": unsolvable_condition}}

    elif prompt_type == "level_3":
        response = result_task["solvable"]

        unsolvable_response = result_task["unsolvable"]

        match = re.search(answer_pattern, response, re.DOTALL)
        unsolvable_match = re.search(answer_pattern, unsolvable_response, re.DOTALL)
        
        if match:
            solvable_progress_rate, solvable_scorers = _evaluate_diagnosing(
                _extract_subgoal_plantool_del_lastfinish(match.group(1).strip()),
                remove_finish_if_last(task_item["planning_tools"]),
                provided_tool_dict, 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.emb_model, args.tools_embedding, task, args, args.calculate_type
            )
        else:
            solvable_progress_rate, solvable_scorers = _evaluate_diagnosing(
                _extract_subgoal_plantool_del_lastfinish(""),
                remove_finish_if_last(task_item["planning_tools"]),
                provided_tool_dict, 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.emb_model, args.tools_embedding, task, args, args.calculate_type
            )
            
        if unsolvable_match:
            unsolvable_progress_rate, unsolvable_scorers, condition = _evaluate_diagnosing_analysis(
                _extract_subgoal_plantool_del_lastfinish(unsolvable_match.group(1).strip()),
                remove_finish_if_last(task_item["planning_tools_unsolvable"]),
                provided_tool_dict, 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.emb_model, args.tools_embedding, task, args, args.calculate_type
            )
        else:
            unsolvable_progress_rate, unsolvable_scorers, condition = _evaluate_diagnosing_analysis(
                _extract_subgoal_plantool_del_lastfinish(""),
                remove_finish_if_last(task_item["planning_tools_unsolvable"]),
                provided_tool_dict, 
                remove_finish_if_last(task_item["planning_tools"]),
                [k for k, _ in provided_tool_dict.items()],
                args.emb_model, args.tools_embedding, task, args, args.calculate_type
            )
            

        return {
            "solvable": {"progress_rate": solvable_progress_rate, "scorers": solvable_scorers},
            "unsolvable": {"progress_rate": unsolvable_progress_rate, "scorers": unsolvable_scorers, "condition": condition}
        }

def process_task(level, task_item, generation_func, task, args, result=None):
    if args.mode in ["infer", "recover"]:
        return _process_task_infer(level, task_item, generation_func, args)
    elif args.mode == "eval":
        return _process_task_eval(level, task_item, task, args, result)
    else:
        raise Exception("Invalid mode")

def process_all_tasks_infer(items, args, generation_func, results=None):
    results = results or {}
    
    for task, task_items in items.items():
        print(f"task: {task}, 数量: {len(task_items)}")
        results[task] = {}
        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):
            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]
            for level in tqdm(levels, desc="Processing levels", leave=False):
                results[task] = results.get(task, {})
                result_list = results[task].get(level, [None] * len(task_items))
                task_item_index = task_items.index(task_item)
                result = result_list[task_item_index] if task_item_index < len(result_list) else None
                
                level_result = process_task(level, task_item, generation_func, task, args, result)
                results[task][level] = results[task].get(level, []) + [level_result]

    return results

def process_all_tasks_recover(items, args, generation_func, results):
    for task, task_items in items.items():
        print(f"task: {task}, 数量: {len(task_items)}")
        results[task] = results.get(task, {})
        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):
            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]
            for level in tqdm(levels, desc="Processing levels", leave=False):
                result_list = results[task].get(level, [None] * len(task_items))
                task_item_index = task_items.index(task_item)
                if task_item_index < len(result_list):
                    result = result_list[task_item_index]
                else:
                    result = None
                
                # Check if result is an empty string and needs to be regenerated
                if isinstance(result, dict):
                    needs_update = any(value == "" for value in result.values())
                else:
                    needs_update = result == ""

                if needs_update:
                    print(f"Updating result for: {task}, item_id: {task_item_index}, level: {level}")
                    level_result = process_task(level, task_item, generation_func, task, args, result)
                    result_list[task_item_index] = level_result
                    results[task][level] = result_list
                else:
                    print(f"Skipping for task: {task}, item_id: {task_item_index}, level: {level}")

    return results

def process_all_tasks_eval(items, args, results):
    metrics = {}
    
    for task, task_items in items.items():
        print(f"task: {task}, 数量: {len(task_items)}")
        metrics[task] = {}
        for task_item in tqdm(task_items, desc=f"Processing {task} items with {args.model_name_save}"):
            levels = [lvl for lvl in ["level_1", "level_2", "level_3"] if getattr(args, lvl)]
            for level in tqdm(levels, desc="Processing levels", leave=False):
                metrics[task] = metrics.get(task, {})
                result_list = results[task].get(level, [])
                task_item_index = task_items.index(task_item)
                if task_item_index < len(result_list):
                    result = result_list[task_item_index]
                else:
                    print(f"Result not found for task item {task_item} at index {task_item_index}")
                    raise Exception("Result not found in eval mode")
                
                level_result = process_task(level, task_item, None, task, args, result)
                metrics[task][level] = metrics[task].get(level, []) + [level_result]

    return metrics
