
_detecting_zh = "根据 <task> 中的任务描述，判断当前任务在 <provided_tools> 提供的工具下是否可以被解决。注意，你需要认真确认 <task> 中的需求、限制，以及 <provided_tools> 中的工具能力描述，并且你无法使用不在 <provided_tools> 中提供的工具。\n你需要使用 <answer> 和 </answer> 包裹你的答案。\n如果任务可以被解决，输出 '<answer>solvable</answer>'。如果任务不可被解决，输出 '<answer>unsolvable</answer>'。除此之外，不要输出任何多余内容。"
_detecting_en = "Determines whether the current task can be solved with the tools provided in <provided_tools>, based on the task description in <task>. Note that you need to carefully review the requirements, limitations of the <task>, and capability descriptions of the tools in <provided_tools>, and that you can't use tools that aren't provided in <provided_tools>. \nYou need to wrap your answer in <answer> and </answer>. \n If the task can be solved, output '<answer>solvable</answer>'. If the task is not solvable, output '<answer>unsolvable</answer>'. Other than that, do not output anything extra."

_planning_zh = "根据 <task> 中的任务描述以及 <provided_tools> 中提供的可用工具进行工具使用的规划。请按顺序列出完成任务时需要使用的工具名，使用 <answer> 和 </answer> 包裹并使用换行符\"\n\"隔开每个工具，例如：\n<answer>Tool1\nTool2\n...\nTooln</answer>\n如果任务的某一步在 <provided_tools> 提供的工具列表中没有可以解决的工具，请在对应步骤使用UnsolvableQuery工具，然后继续列出剩余步骤进行规划，例如：\n<answer>Tool1\n...\nUnsolvableQuery\n...\nTooln</answer>\n如果 <task> 中存在对使用工具数量的限制，在达到限制数量之后的下一步调用UnsolvableQuery来终止任务，例如限制工具数量为t个以内时：\n<answer>Tool1\n...\nToolt\nUnsolvableQuery</answer>\n\n在包裹的答案中，除了工具名之外，不要输出任何多余内容，也不需要对输出进行任何解释。\n你无法使用不在 <provided_tools> 中提供的工具。\n确保每一步骤的工具名都清晰且独立。仅输出你认为最正确的一个答案。"
_planning_en = "Plan your tool usage based on the task description in <task> and the available tools provided in <provided_tools>. List the names of the tools you need to use to complete the task in order, wrapping them in <answer> and </answer> and separating each tool with a line break \“\n\”, e.g.: \n<answer>Tool1\nTool2\n... \nTooln</answer>\nIf a step of the task does not have a tool that can be solved in the list of tools provided by <provided_tools>, use the UnsolvableQuery tool at the corresponding step, and then continue the planning for the remaining steps, e.g.: \n<answer>Tool1\n... \nUnsolvableQuery\n... \nTooln</answer>\nIf there is a limit to the number of tools that can be used in <task>, call UnsolvableQuery to terminate the task on the next step after the limit is reached, e.g., to limit the number of tools to t or less: \n<answer>Tool1\n... \nToolt\nUnsolvableQuery</answer>\n\nIn a wrapped answer, do not output any redundancy other than the name of the tool, and do not interpret the output in any way. \nYou cannot use tools that are not provided in <provided_tools>. \nEnsure that the tool names are clear and separated for each step. \nOutput only the one answer you think is most correct."

_diagnosing_zh = "请根据 <task> 中的任务描述和 <provided_tool> 中的工具拆分任务并制定一个工具使用的任务规划，要求：\n1. 根据可用工具列表提供的工具进行每一步的任务规划，可能包括多个步骤(t>=1)，每个步骤对应一个子目标和一个工具使用。\n2. 各子目标之间需有逻辑关系，确保子目标的完成推进整体任务进展。\n3. 无法使用不在 <provided_tool> 中的工具\n4. 使用 <answer> 和 </answer> 包裹整个答案\n任务规划的格式请参考如下示例：\n<answer>Subgoal 1: [描述] Planned tool: [工具名]\nSubgoal 2: [描述] Planned tool: [工具名]\n...\nSubgoal t: [描述] Planned tool: [工具名]</answer>\n在[描述]部分阐述子目标和需求之间的关系，在[工具名]的部分提供工具名，不要在[工具名]的部分进行任何的描述或解释。\n\n如果某个子目标因缺乏合适工具无法完成，请描述缺少工具的能力和需要解决的需求，并使用对应不可解决的工具UnsolvableQuery。然后，假设该需求已被完成，并继续规划后续步骤，例如：\n<answer>Subgoal 1: [描述] Planned tool: [工具名]\nSubgoal 2: [对缺少工具的功能描述] Planned tool: UnsovlableQuery\n...\nSubgoal t: [描述] Planned tool: [工具名]</answer>\n如果 <task> 中存在对使用工具数量的限制，在达到限制数量之后的下一步调用UnsolvableQuery来终止任务。例如限制工具数量为t个以内时：\n<answer>Subgoal 1: [description] Planned tool: [tool name]\n...\nSubgoal t: [description] Planned tool: [tool name]\nSubgoal t+1: [Task requires number of tools within t] Planned tool: UnsovlableQuery</answer>\n\n现在，我们开始规划当前任务，使用 <answer> 来标记任务开始，并使用 </answer> 结束任务。"
_diagnosing_en = "Divide the task and develop a task plan for tool usage based on the task description in <task> and the tools in <provided_tool>. Requirements:\n1. Each step of the task planning based on the tools provided in the list of available tools may consist of multiple steps (t>=1), with each step corresponding to a sub-objective and a tool usage. \n2. Sub-objectives need to be logically related to each other to ensure that the completion of sub-objectives advances the overall task progress. \n3. Tools that are not in <provided_tool> cannot be used \n4. Use <answer> and </answer> to wrap the entire answer \nThe format for task planning can be seen in the following example: \n<answer>Subgoal 1: [description] Planned tool: [tool name] \nSubgoal 2: [description ] Planned tool: [tool name]\n... \nSubgoal t: [description] Planned tool: [tool name]</answer>\nState the relationship between the subgoal and the requirement in the [description] section, provide the tool name in the [tool name] section, and don't provide any description or explanation in the [tool name] section. \nIf there is a limit to the number of tools that can be used in a <task>, the next step after the limit is reached calls UnsolvableQuery to terminate the task. For example, to limit the number of tools to t or less: \n<answer>Subgoal 1: [description] Planned tool: [tool name]\n...\nSubgoal t: [description] Planned tool: [tool name]\nSubgoal t+1: [Task requires number of tools within t] Planned tool: UnsovlableQuery</answer>\n\nNow, let's start scheduling the current task, using <answer> to mark the start of the task and </answer> mark the end of the task."
