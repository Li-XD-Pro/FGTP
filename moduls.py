from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
from PIL import Image
import argparse
import torchvision.transforms as T
from models import build_model
from transformers import BertTokenizer, BertModel
import spacy
import openai
import torch.nn.functional as F

def get_args_parser(img_path, resume):
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default=img_path,
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default=resume, help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser



def ReITR(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(
        -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[
              :topk]
    keep_queries = keep_queries[indices]

    scene_results = []
    # 获取关系三元组
    for idx in keep_queries:
        subject_class = CLASSES[probas_sub[idx].argmax()]
        relationship_class = REL_CLASSES[probas[idx].argmax()]
        object_class = CLASSES[probas_obj[idx].argmax()]
        # print(f"Subject: {subject_class}, Relationship: {relationship_class}, Object: {object_class}")
        scene_result = f"{subject_class} {relationship_class} {object_class}"
        scene_results.append(scene_result)
    scene_result = ", ".join(scene_results)
    # print("Scene Result:", scene_result)

    return scene_result



def get_noun_verb(task_text):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(task_text)
    verbs = []
    nouns = []
    for token in doc:
        if "VB" in token.tag_:
            verbs.append(token.text)
        elif "NN" in token.tag_:
            nouns.append(token.text)

    new_sentence = " ".join(verbs + [nouns[0]])
    # print(new_sentence)
    return verbs, [nouns[0]]



def get_groundingdino(img, categories, ckpt):
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", ckpt)

    TEXT_PROMPT = categories
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # print('type of TEXT_PROMPT: ', type(TEXT_PROMPT))
    image_source, image = load_image(img)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # print('boxes:', boxes)
    # print('logits:', logits)
    # print('phrases:', phrases)

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("result/img_detected.jpg", annotated_frame)

    return boxes, logits, phrases

def task_attribute(task_text,attribute_text):
    # 分割变量 attribute_text
    elements_a = attribute_text.split(',')

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 预处理变量 task_text
    inputs_b = tokenizer(task_text, return_tensors='pt')
    with torch.no_grad():
        outputs_b = model(**inputs_b)
    embeddings_b = outputs_b.last_hidden_state[:, 0, :]

    # 计算每个元素与变量 task_text 的相似度
    cosine_similarities = []
    for element in elements_a:
        inputs_a = tokenizer(element, return_tensors='pt')
        with torch.no_grad():
            outputs_a = model(**inputs_a)
        embeddings_a = outputs_a.last_hidden_state[:, 0, :]
        cosine_similarity = F.cosine_similarity(embeddings_a, embeddings_b)
        cosine_similarities.append(cosine_similarity.item())

    # 找到最高相似度值
    max_similarity = max(cosine_similarities)

    # 筛选出相似度接近最高值的元素（相差小于0.05）
    close_to_max_elements = [element for element, similarity in zip(elements_a, cosine_similarities) if
                             max_similarity - similarity < 0.05]
    return close_to_max_elements

def gpt(prompt):
    response = openai.ChatCompletion.create(
        model= 'gpt-3.5-turbo',
        messages=[
            {'role': 'user','content': prompt}
        ],
        temperature=0,
    )
    # print(response.choices[0].message.content)
    answer = response.choices[0].message.content
    return answer

def yolov8_detect(results_yolov8):
    target_boxes = []
    for r in results_yolov8:
        n = len(r.boxes.cls)
        print("The number of detected objects: ", n)
        for i in range(n):
            if (r.boxes.cls[i] == 1):
                target_boxes = r.boxes.xyxy[i]
            if (r.boxes.cls[i] == 2):
                target_boxes = r.boxes.xyxy[i]
            if (r.boxes.cls[i] == 7):
                target_boxes = r.boxes.xyxy[i]
    print("target_boxes: ", target_boxes)
    target_boxes = target_boxes.cpu().numpy()
    x1 = target_boxes[0]
    y1 = target_boxes[1]
    x2 = target_boxes[2]
    y2 = target_boxes[3]
    return x1, y1, x2, y2

'''
    功能：检测当前物品的状态是否适合完成任务 
    思路：先使用YOLOv8检测当前物品的状态，检测到的状态存储在obj_class中，然后计算task与句子"The [noun] is [obj_class]"之间的相似度similarities_values
         同时，在opposite_class中存储obj_class的相反状态，并计算task与句子"The [noun] is [opposite_class]"之间的相似度opposite_similarities_values
         比较similarities_values与opposite_similarities_values的大小，若前者大，则当前物品的状态适合完成任务，返回success_flag = 1；若后者大，则当前物品的状态不适合完成任务，返回success_flag = 2
'''
def state_detect(results_yolov8, task, noun):
    for r in results_yolov8:
        n = len(r.boxes.cls)
        print("The number of detected objects: ", n)
        obj_class = [[] for _ in range(n)] # 定义n维空列表
        target_boxes = [[] for _ in range(n)]
        for i in range(n):
            if(r.boxes.cls[i] == 0):
                obj_class[i] = 'closed'
                target_boxes[i] = r.boxes.xyxy[i]
            if(r.boxes.cls[i] == 1):
                obj_class[i] = 'open'
                target_boxes[i] = r.boxes.xyxy[i]
            if(r.boxes.cls[i] == 2):
                obj_class[i] = 'empty'
                target_boxes[i] = r.boxes.xyxy[i]
            if(r.boxes.cls[i] == 3):
                obj_class[i] = 'occupied'
                target_boxes[i] = r.boxes.xyxy[i]
            if(r.boxes.cls[i] == 7):
                obj_class[i] = 'unconnected'
                target_boxes[i] = r.boxes.xyxy[i]
            if(r.boxes.cls[i] == 5):
                obj_class[i] = 'connected'
                target_boxes[i] = r.boxes.xyxy[i]

    print('obj_class:', obj_class)
    opposite_class = []
    if(obj_class == ['closed']):
        opposite_class = ['open']
    if (obj_class == ['open']):
        opposite_class = ['closed']
    if(obj_class == ['closed']):
        opposite_class = ['open']
    if (obj_class == ['empty']):
        opposite_class = ['occupied']
    if (obj_class == ['occupied']):
        opposite_class = ['empty']
    if(obj_class == ['unconnected']):
        opposite_class = ['connected']
    if (obj_class == ['connected']):
        opposite_class = ['unconnected']

    # Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to calculate cosine similarity for tensors
    def cosine_similarity_tensors(tensor1, tensor2):
        return F.cosine_similarity(tensor1, tensor2, dim=1)

    # Generating sentences
    sentences = [f"The {noun} is {state}" for state in obj_class]
    opposite_sentences = [f"The {noun} is {state}" for state in opposite_class]

    # Tokenizing and encoding sentences and task
    encoded_sentences = [tokenizer(sentence, return_tensors='pt') for sentence in sentences]
    encoded_opposite_sentences = [tokenizer(opposite_sentences, return_tensors='pt') for opposite_sentences in opposite_sentences]
    encoded_task = tokenizer(task, return_tensors='pt')

    # Processing with BERT model
    with torch.no_grad():
        outputs_sentences = [model(**encoded_sentence) for encoded_sentence in encoded_sentences]
        outputs_opposite_sentences = [model(**encoded_opposite_sentences) for encoded_opposite_sentences in encoded_opposite_sentences]
        output_task = model(**encoded_task)

    # Extracting the embeddings (CLS token)
    embeddings_sentences = [output.last_hidden_state[:, 0, :] for output in outputs_sentences]
    embeddings_opposite_sentences = [output.last_hidden_state[:, 0, :] for output in outputs_opposite_sentences]
    embedding_task = output_task.last_hidden_state[:, 0, :]

    # Calculating cosine similarities
    similarities = [cosine_similarity_tensors(embedding_task, embedding_sentence) for embedding_sentence in
                    embeddings_sentences]

    opposite_similarities = [cosine_similarity_tensors(embedding_task, embedding_opposite_sentences) for embedding_opposite_sentences in
                    embeddings_opposite_sentences]

    # Convert tensor to value and find the index of the highest similarity
    similarities_values = [similarity.item() for similarity in similarities]
    opposite_similarities_values = [opposite_similarities.item() for opposite_similarities in opposite_similarities]

    print('similarities_values: ', similarities_values)
    print('opposite_similarities_values: ', opposite_similarities_values)

    success_flag = 0
    if(similarities_values > opposite_similarities_values):
        success_flag = 1
    else:
        success_flag = 2
    target_boxes = target_boxes[0].cpu().numpy()
    x1 = target_boxes[0]
    y1 = target_boxes[1]
    x2 = target_boxes[2]
    y2 = target_boxes[3]
    return x1, y1, x2, y2, success_flag
