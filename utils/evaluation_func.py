import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type):
    if calculate_type == "hard":
        match_number = 0

        for i in range(len(golden_planning_tool_list)):
            if i < len(pred_planning_tool_list):
                if pred_planning_tool_list[i] == golden_planning_tool_list[i]:
                    match_number += 1
                else:
                    break
            else:
                break

        progress_rate = match_number / len(golden_planning_tool_list)
        return progress_rate
    
    elif calculate_type == "soft":
        match_count = 0
        g_index, p_index = 0, 0
        
        while g_index < len(golden_planning_tool_list) and p_index < len(pred_planning_tool_list):
            if golden_planning_tool_list[g_index] == pred_planning_tool_list[p_index]:
                g_index += 1
                match_count += 1
            p_index += 1

        progress_rate = match_count / len(golden_planning_tool_list)
        return progress_rate

    else:
        raise Exception("calculate_type must be hard or soft")

def calculate_scorers(pred_subgoal_texts, pred_planning_tool_list, 
                    golden_planning_tool_list, 
                    origin_provided_subgoal_dict, solvable_planning_tool_list, 
                    emb_model, tools_embedding, task, args):
    scorers = []
    
    for i in range(len(golden_planning_tool_list)):
        if golden_planning_tool_list[i] == "UnsolvableQuery":
            if i < len(pred_planning_tool_list):
                if golden_planning_tool_list[i] == pred_planning_tool_list[i]:
                    task_emb = tools_embedding[task]
                    pattern = r"Subgoal\s*\d+[:：]\s*(.*)"
                    pred_subgoal_text_pure = re.sub(pattern, r"\1", pred_subgoal_texts[i])
                    
                    if args.embedding_model == "minilm":
                        pred_embedding = emb_model.encode(pred_subgoal_text_pure)
                    elif args.embedding_model == "gemini":
                        pred_embedding = emb_model(pred_subgoal_text_pure)["embedding"]
                    similarities = cosine_similarity([pred_embedding], task_emb["embeddings"])[0]
                    best_match_index = np.argmax(similarities)
                    best_task_name = task_emb["name"][best_match_index]
                    
                    if best_task_name == solvable_planning_tool_list[i]:
                        scorers.append(1.0)
                    else:
                        true_deleted_tool_desc = re.sub(pattern, r"\1", origin_provided_subgoal_dict[solvable_planning_tool_list[i]])
                        
                        if args.embedding_model == "minilm":
                            true_deleted_tool_emb = emb_model.encode(true_deleted_tool_desc)
                        elif args.embedding_model == "gemini":
                            true_deleted_tool_emb = emb_model(true_deleted_tool_desc)["embedding"]
                            
                        scorers.append(float(
                            cosine_similarity([pred_embedding], [true_deleted_tool_emb])[0][0]))
                else:
                    scorers.append(0.0)
            else:
                scorers.append(0.0)

    return scorers

def _evaluate_detecting(pred_solvability, golden_solvability):
    return 0 if pred_solvability.lower() != golden_solvability.lower() else 1

def _evaluate_planning(
        pred_planning_tool_list, 
        golden_planning_tool_list, 
        provided_tool_list,
        calculate_type = "hard"
    ):
    
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        return 0.0

    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        return 0.0
    
    progress_rate = calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    
    return progress_rate

def _evaluate_planning_analysis(
        pred_planning_tool_list, 
        golden_planning_tool_list, 
        provided_tool_list,
        calculate_type = "hard"
    ):
    condition = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        return 0.0, "non_existent_tools"

    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        return 0.0, "solvability_hallu"
    
    progress_rate = calculate_progress_rate(pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    
    if condition is None:
        if not all(tool in golden_planning_tool_list for tool in pred_planning_tool_list):
            condition = "wrong_tools"
        else:
            if progress_rate == 1.0:
                condition = "correct"
            else:
                if golden_planning_tool_list.index("UnsolvableQuery") != pred_planning_tool_list.index("UnsolvableQuery"):
                    condition = "wrong_unsolvable_index"
                else:
                    condition = "wrong_reasoning"
                    
    return progress_rate, condition

def _evaluate_diagnosing(
        pred_planning_tuple, 
        golden_planning_tool_list, 
        origin_provided_subgoal_dict, solvable_planning_tool_list,
        provided_tool_list,
        emb_model, tools_embedding, task, args, calculate_type = "hard"
    ):
    pred_subgoal_texts, pred_planning_tool_list = pred_planning_tuple
    
    progress_rate = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        progress_rate = 0.0
    
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        unsolvable_count = golden_planning_tool_list.count("UnsolvableQuery")
        return 0.0, [0.0] * unsolvable_count

    if progress_rate is None:
        progress_rate = calculate_progress_rate(
            pred_planning_tool_list, golden_planning_tool_list, calculate_type)

    if "UnsolvableQuery" in golden_planning_tool_list:
        scorers = calculate_scorers(pred_subgoal_texts, pred_planning_tool_list, 
                                    golden_planning_tool_list, 
                                    origin_provided_subgoal_dict, solvable_planning_tool_list, 
                                    emb_model, tools_embedding, task, args)
    else:
        scorers = ""
        
    return progress_rate, scorers

def _evaluate_diagnosing_analysis(
        pred_planning_tuple, 
        golden_planning_tool_list, 
        origin_provided_subgoal_dict, solvable_planning_tool_list,
        provided_tool_list,
        emb_model, tools_embedding, task, args, calculate_type = "hard"
    ):
    pred_subgoal_texts, pred_planning_tool_list = pred_planning_tuple
    
    progress_rate = None
    condition = None
    if not all(tool in provided_tool_list for tool in pred_planning_tool_list):
        progress_rate = 0.0
        condition = "non_existent_tools"
    
    if "UnsolvableQuery" in golden_planning_tool_list and "UnsolvableQuery" not in pred_planning_tool_list:
        unsolvable_count = golden_planning_tool_list.count("UnsolvableQuery")
        return 0.0, [0.0] * unsolvable_count, "solvability_hallu"

    if progress_rate is None:
        progress_rate = calculate_progress_rate(
            pred_planning_tool_list, golden_planning_tool_list, calculate_type)
    
    if "UnsolvableQuery" in golden_planning_tool_list:
        scorers = calculate_scorers(pred_subgoal_texts, pred_planning_tool_list, 
                                    golden_planning_tool_list, 
                                    origin_provided_subgoal_dict, solvable_planning_tool_list, 
                                    emb_model, tools_embedding, task, args)
    else:
        scorers = ""
        
    if condition is None:
        if not all(tool in golden_planning_tool_list for tool in pred_planning_tool_list):
            condition = "wrong_tools"
        else:
            if progress_rate == 1.0:
                condition = "correct"
            else:
                if golden_planning_tool_list.index("UnsolvableQuery") != pred_planning_tool_list.index("UnsolvableQuery"):
                    condition = "wrong_unsolvable_index"
                else:
                    condition = "wrong_reasoning"

    return progress_rate, scorers, condition

