import spacy

# 加载英语模型（需要先安装spaCy和英语模型）
nlp = spacy.load("en_core_web_sm")

def AutoCheckGrammar(plan):
    grammar_score = 10
    for statement in plan:
        doc = nlp(statement)
        # 检查句子完整性和复杂的语法规则
        if len(list(doc.sents)) != 1 or not doc[0].is_sent_start or doc[-1].text != '.':
            grammar_score -= 1
    return max(0, grammar_score)

def CheckCompliance(plan):
    # 假设存在一个详细的合规性规则集
    compliance_score = 10
    for statement in plan:
        if "unallowed action" in statement:  # 示例规则
            compliance_score -= 1
    return max(0, compliance_score)

def CheckPrePostConditions(plan):
    condition_score = 10
    fridge_status = 'closed'
    for statement in plan:
        if statement.lower() == "open the refrigerator":
            fridge_status = 'open'
        elif statement.lower() == "close the refrigerator":
            fridge_status = 'closed'
        elif "take something from the refrigerator" in statement.lower():
            if fridge_status != 'open':
                condition_score -= 1
    return max(0, condition_score)

# 示例使用
plan = ["Open the refrigerator", "Take something from the refrigerator", "Close the refrigerator"]
print("Grammar Score:", AutoCheckGrammar(plan))
print("Compliance Score:", CheckCompliance(plan))
print("Pre/Post Condition Score:", CheckPrePostConditions(plan))

def calculate_executability(grammar_score, compliance_score, condition_score):
    # 转换为百分比
    grammar_percentage = (grammar_score / 10) * 100
    compliance_percentage = (compliance_score / 10) * 100
    condition_percentage = (condition_score / 10) * 100

    # 权重可能根据实际需求调整
    weights = {'grammar': 0.3, 'compliance': 0.3, 'condition': 0.4}

    # 加权求和
    weighted_average = (grammar_percentage * weights['grammar'] +
                        compliance_percentage * weights['compliance'] +
                        condition_percentage * weights['condition'])

    return weighted_average

# 使用之前定义的函数计算每个得分
grammar_score = AutoCheckGrammar(plan)
compliance_score = CheckCompliance(plan)
condition_score = CheckPrePostConditions(plan)

# 计算最终的执行性得分
executability_score = calculate_executability(grammar_score, compliance_score, condition_score)
print(f"Executability Score: {executability_score}%")


'''
AutoCheckGrammar，检查句子的语法规则：
    是否含有动词，动词是否在句首，并且是否有直接宾语
'''

# import spacy
# nlp = spacy.load("en_core_web_sm")
# def check_task_grammar(task):
#     doc = nlp(task)
#     has_verb = any([token.pos_ == "VERB" for token in doc])
#     starts_with_verb = doc[0].pos_ == "VERB"
#     has_direct_object = any([token.dep_ == "dobj" for token in doc])
#     return has_verb and starts_with_verb and has_direct_object
# task = "Open the refrigerator"
# print("Grammar Check:", check_task_grammar(task))



'''
CheckCompliance: 检查计划中的每个步骤是否符合特定的规则或标准
    比如每个生成的动作是否来自Action List['MoveTo', 'Grasp', 'PickUp', 'PutDown']
'''
# def CheckCompliance(task_plan, action_list):
#     compliance_score = 10
#     max_penalty_per_action = 2  # 每个不合规动作的最大扣分
#
#     for step in task_plan:
#         # 提取每个步骤的第一个词
#         first_word = step.split()[0]
#         if first_word not in action_list:
#             compliance_score -= max_penalty_per_action
#             compliance_score = max(0, compliance_score)  # 确保分数不会变成负数
#
#     return compliance_score
#
# # 示例使用
# action_list = ['MoveTo', 'Grasp', 'PickUp', 'PutDown']
# task_plan = [
#     'MoveTo the right side of the table',
#     'Grasp the cup from the parallel',
#     'PickUp the cup stably',
#     'MoveTo the dining table',
#     'PutDown the cup stably'
# ]
#
# compliance_score = CheckCompliance(task_plan, action_list)
# print(f"Compliance Score: {compliance_score}")



'''
CheckPrePostConditions: 检查各步骤间逻辑关系
    检查任务的每一步是否符合逻辑，例如先“打开冰箱”再“取出食物”。
'''
# def CheckPrePostConditions(task_plan, pre_post_conditions):
#     compliance_score = 10
#     max_penalty_per_violation = 2  # 每个逻辑错误的最大扣分
#
#     # 检查每个条件
#     for condition in pre_post_conditions:
#         pre_condition, post_condition = condition
#
#         if pre_condition in task_plan and post_condition in task_plan:
#             pre_index = task_plan.index(pre_condition)
#             post_index = task_plan.index(post_condition)
#
#             # 如果先决条件出现在后续条件之后，则扣分
#             if pre_index > post_index:
#                 compliance_score -= max_penalty_per_violation
#                 compliance_score = max(0, compliance_score)  # 确保分数不会变成负数
#
#     return compliance_score
#
# # 示例使用
# task_plan = [
#     'MoveTo the refrigerator',
#     'Open the refrigerator',
#     'Take out the food',
#     'Close the refrigerator'
# ]
#
# # 定义先决条件和后续条件的规则
# pre_post_conditions = [
#     ('Open the refrigerator', 'Take out the food'),  # 必须先打开冰箱，然后才能取出食物
#     ('Take out the food', 'Close the refrigerator')  # 必须在取出食物后才能关闭冰箱
# ]
#
# compliance_score = CheckPrePostConditions(task_plan, pre_post_conditions)
# print(f"Compliance Score: {compliance_score}")

