import re

def remove_finish_if_last(tools):
    if tools and tools[-1].lower() == 'finish':
        return tools[:-1]
    return tools

def _extract_subgoal_plantool_del_lastfinish(output):
    pattern = r"(Subgoal \d+[:：]\s*.*?)(?= Planned tool[:：]) Planned tool[:：]\s*([\w\s\+\-]+)(?=\n|$)"
    matches = re.findall(pattern, output)

    subgoal_texts, planned_tools = [], []
    for match in matches:
        subgoal_text, planned_tool = match
        subgoal_texts.append(subgoal_text.strip())
        planned_tools.append(planned_tool.strip())

    if planned_tools and planned_tools[-1].lower() == 'finish':
        planned_tools = planned_tools[:-1]
        subgoal_texts = subgoal_texts[:-1]
    
    return subgoal_texts, planned_tools