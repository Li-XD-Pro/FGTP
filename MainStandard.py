import torch
from PIL import Image
from torchvision.ops import box_convert
import clip
import json
import numpy as np
import argparse
import openai
from ultralytics import YOLO
from moduls import get_args_parser, ReITR, get_noun_verb, task_attribute,state_detect,yolov8_detect,gpt

if __name__ =='__main__':
    # 0、OpenAI api_key
    openai.api_key = ""

    # 1、输入任务
    task = "Please take the cup to the diningtable."

    # 2、获取任务中的关键动词和名词，比如take和cup
    verb, noun = get_noun_verb(task)

    # 3、状态检测
    img_name = 'image/img4.jpg'
    model_yolov8 = YOLO('ckpt/best.pt')
    results_yolov8 = model_yolov8(img_name)  # results list
    x1, y1, x2, y2, success_flag =state_detect(results_yolov8=results_yolov8, task=task, noun=noun) # 对检测到的物品状态进行判断，选择状态正常的完成任务

    # 4、生成输入图像的场景图
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser(img_path=img_name,resume='ckpt/ReITR.pth')])
    args = parser.parse_args()
    scene_result = ReITR(args)

    # 5、删除场景图三元组描述中的重复部分，每种三元组描述只保留一个
    # 使用split()方法将字符串分割成列表
    scene_list = scene_result.split(', ')
    # 使用set()来删除重复项
    unique_scene_list = list(set(scene_list))
    # 将列表重新组合成一个字符串
    unique_scene_result = ', '.join(unique_scene_list)
    # 打印去重后的结果
    # print('Final Scene Graph: ',unique_scene_result)

    # 6、从场景图中筛选出被操作物品的位置信息
    scene_items = unique_scene_result.split(', ')
    filtered_items = [item for item in scene_items if item.startswith(noun[0])]
    filtered_scene_graph = ', '.join(filtered_items)
    filtered_scene_graph_0 = filtered_scene_graph.split(', ')
    # print('filtered_scene_graph: ',filtered_scene_graph_0[0])

    # 7、从输入图像中将Bounding Box的物体抠出来
    img = Image.open(img_name)
    object_region = img.crop([x1, y1, x2, y2])
    object_region = object_region.convert("RGB")
    object_region.save("result/img_cropped.jpg")

    # 8、将Bounding Box转换为语言描述：左右，表示被检测物体在整个场景中的位置
    scene_height = img.size[0]
    scene_width = img.size[1]
    #   计算矩形框的中心点坐标
    item_center_x = (x1 + x2) / 2
    item_center_y = (y1 + y2) / 2
    #   左侧和右侧
    if x1 < scene_width / 2:
        object_2D_location = "left"
    else:
        object_2D_location = "right"
    #   输出位置关系
    # print(f"目标物品的2D位置：{object_2D_location}")

    noun_matching = ''.join(noun)
    # print(noun_matching)

    # 9、物体名称进知识库做匹配，获得该物品可能具有的属性
    #   读取原始JSON文件
    with open('ontology.json', 'r') as original_file:
        data = json.load(original_file)
    #   初始化一个用于存储匹配的 answers 的列表
    matching_answers = []
    for item in data:
        questions = item.get("object", "").strip()
        answers = item.get("attribute", "").strip()
        if noun_matching == questions:############ 用noun去知识库做匹配
            matching_answers.append(answers)
    # print('Original attributes: ', matching_answers)

    # 10、使用CLIP对Crop出来的图像与匹配出来的属性进行相似度计算，选出物品真正具有的属性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    image = preprocess(object_region).unsqueeze(0).to(device)
    text = clip.tokenize(matching_answers).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # print("Label probs:", probs)
    #   将label_probs转换为一维数组
    label_probs = probs.ravel()
    #   找到最大值和其索引
    max_value = np.max(label_probs)
    max_index = np.argmax(label_probs)
    #   筛选出和最大值的差值在10倍以内的值及其索引
    filtered_values = []
    filtered_indices = []
    for i, value in enumerate(label_probs):
        if abs(value - max_value) < 10 * value:
            filtered_values.append(value)
            filtered_indices.append(i)

    clip_attribute = [matching_answers[i] for i in filtered_indices]
    clip_attribute = [value for value in clip_attribute if not value.startswith('is in')]
    # print('Clip attribute: ',clip_attribute)

    # 11、任务和属性匹配
    new_attributes = [item for item in matching_answers if not item.startswith('is')]
    new_attributes = [item for item in new_attributes if not item.startswith('has Color')]     # 删除颜色和位置属性
    bert_attribute_input = ','.join(new_attributes)
    # print(bert_attribute_input)
    bert_attribute = task_attribute(task_text=task,attribute_text=bert_attribute_input)

    object_attribute = clip_attribute + bert_attribute
    unique_object_attribute = list(set(object_attribute))
    unique_object_attribute = [item for item in unique_object_attribute if not item.startswith('is')]

    # print('unique_object_attribute: ', unique_object_attribute)

    print('**************************************************************  任务规划  **************************************************************')
    print('The_high-level_instruction: ', task)
    print('The_2D_location: ', object_2D_location)
    print('The_environment_scene_graph: ', filtered_scene_graph_0[0])
    print('The_object_attribute: ', str(unique_object_attribute))

    grasp_limitation =[]
    if 'has Shape boxshaped' in unique_object_attribute:
        grasp_limitation = 'Grasp from the center'
    if 'has Shape cylinder' in unique_object_attribute:
        grasp_limitation = 'Grasp from the parallel'
    if 'has Shape irregularshape' in unique_object_attribute:
        grasp_limitation = 'Grasp from multiple clip points'
    if 'has Shape sphere' in unique_object_attribute:
        grasp_limitation = 'Grasp in uniform force'
    if 'has Shape round' in unique_object_attribute:
        grasp_limitation = 'Grasp from the axis'
    print('The_grasp_limitation: ', grasp_limitation)

    put_limitation =[]
    if 'has Material metal' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    if 'has Material porcelain' in unique_object_attribute:
        put_limitation = 'PickUP gently'
    if 'has Material clothmade' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    if 'has Material glass' in unique_object_attribute:
        put_limitation = 'PickUP gently'
    if 'has Material plastic' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    if 'has Material wood' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    if 'has Material stainlesssteel' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    if 'has Material papermade' in unique_object_attribute:
        put_limitation = 'PickUP stably'
    print('The_put_limitation: ', put_limitation)
    #
    # 12、LLM
    prompt_task1 = "You are a task planner for service robot. Your aim is to break down high-level task into subtasks for the robot to execute. " \
             "You can only use actions from the action list ['MoveTo', 'Grasp', 'PickUp', 'PutDown']." \
             "The meaning of the actions are:" \
             "MoveTo: move to a target location, including an object or a room." \
             "Grasp: grasp an object." \
             "PickUp: pick up an object after grasping it." \
             "PutDown: put down an object after grasping it." \
             "The information I give you includes five variables:The_high-level_instruction, The_2D_location, The_environment_scene_graph, The_grasp_limitation, The_put_limitation." \
             "The meaning of the variables are:"\
             "The_high-level_instruction: including the operating object and destination location."\
             "The_2D_location: the location of the operating object, including left or right."\
             "The_environment_scene_graph: the location of the operating object, also the destination of the first subtask."\
             "The_grasp_limitation: the limitation of the the action 'Grasp'." \
             "The_put_limitation: the limitation of the action 'PickUP' and 'PutDown'." \
             "Example: The high-level instruction: Please take me the apple to the sofa. The object_2D_location: left. The_environment_scene_graph: apple on desk. The_grasp_limitation: Grasp from the parallel. The_put_limitation: PickUP gently. According to The scene_result, the apple in on the 'desk', and the object_2D_location 'left' so 1.MoveTo the left side of the desk.According to The_grasp_limitation, 2.Grasp the apple from the parallel. According to The_put_limitation, 3.PickUp the apple gently.According to the high-level instruction, the destination is the sofa, so 4.MoveTo the sofa. According to The_put_limitation, 5.PutDown the apple gently. The subtask: 1.MoveTo the left side of the desk. 2.Grasp the apple from the parallel. 3.PickUp the apple gently. 4.MoveTo the sofa. 5.PutDown the apple gently." \
             "You need to break down the task based on the following information, The_high-level_instruction: " + task + ". The_2D_location: " + object_2D_location + ". The_environment_scene_graph: "+ filtered_scene_graph_0[0] + ". The_grasp_limitation: " + str(grasp_limitation) + ". The_put_limitation:" + str(put_limitation) + \
             "Note: Please fully understand the meaning of the example. You just need to output subtasks just like I do, don't explain anything."

    Planning_result = gpt(prompt_task1)
    print('Planning_result:\n', Planning_result)

    # 13、将规划的结果写入result/planning.txt
    file_path = 'result/StandardPlanning.txt'
    # Writing to the file
    with open(file_path, 'w') as file:
        file.write(Planning_result)
    print("任务规划结果已写入: " + file_path)
