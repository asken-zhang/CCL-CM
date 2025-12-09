import clip
import torch

cosine_sim = torch.nn.CosineSimilarity()

class CLIP():
    def __init__(self, device, model="ViT-B/32", eval=True) -> None:
        self.model, self.preprocess = clip.load(model, device=device)
        if eval:
            self.model.eval()

        self.MEAN = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
        self.STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        self.device = device

    def text_tokens(self, prompts_list):
        return clip.tokenize(prompts_list)

    def image_embeds(self, images, normalize=True):
        if normalize:
            return self.model.encode_image(
                (images.to(self.device) - self.MEAN[None, :, None, None]) / self.STD[None, :, None, None]
            )
        else:
            return self.model.encode_image(images.to(self.device))

    def text_embeds(self, text_tokens, with_grad=False):
        if with_grad:
            return self.model.encode_text(text_tokens.to(self.device))
        else:
            return self.model.encode_text(text_tokens.to(self.device)).detach()

def sliding_window(image, window_size, step_size):
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            yield (row, col, image[row:row + window_size[0], col:col + window_size[1]])


def overlapping_area(detection_1, detection_2, show = False):
    '''
    计算两个检测区域覆盖大小，detection：(x, y, pred_prob, width, height, area)
    '''
    # Calculate the x-y co-ordinates of the
    # rectangles
    # detection_1的 top left 和 bottom right
    x1_tl = detection_1[0]
    y1_tl = detection_1[1]
    x1_br = detection_1[0] + detection_1[3]
    y1_br = detection_1[1] + detection_1[4]

    # detection_2的 top left 和 bottom right
    x2_tl = detection_2[0]
    y2_tl = detection_2[1]
    x2_br = detection_2[0] + detection_2[3]
    y2_br = detection_2[1] + detection_2[4]

    # 计算重叠区域
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_1[4]
    area_2 = detection_2[3] * detection_2[4]

    # 1. 重叠比例计算1
    # total_area = area_1 + area_2 - overlap_area
    # return overlap_area / float(total_area)

    # 2.重叠比例计算2
    area = area_1
    if area_1 < area_2:
        area = area_2
    return float(overlap_area / area)


def nms(detections, threshold=0.1):
    '''
    抑制策略：
    1. 最大的置信值先行
    2. 最大的面积先行
    非极大值抑制减少重叠区域, detection:(x, y, pred_prob, width, height)
    '''
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    # 根据预测值大小排序预测结果
    detections = sorted(detections, key=lambda detections: detections[2].any(), reverse=True)
    # print((detections[0][5], detections[-1][5]))
    # Unique detections will be appended to this list
    # 非极大值抑制后的检测区域
    new_detections=[]
    # Append the first detection
    # 默认第一个区域置信度最高是正确检测区域
    new_detections.append(detections[0])
    # Remove the detection from the original list
    # 去除以检测为正确的区域
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    print(len(detections))

    for index, detection in enumerate(detections):
        if len(new_detections) >= 20:
            break
        overlapping_small = True
        # 重叠区域过大，则删除该区域，同时结束检测，过小则继续检测
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                overlapping_small = False
                break
        # 整个循环中重叠区域都小那么增加
        if overlapping_small:
            new_detections.append(detection)
    return new_detections

def cosine_avg(features, targets):
    return cosine_sim(features, targets)