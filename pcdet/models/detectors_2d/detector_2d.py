import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
import torch
import numpy as np
from math import *
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import time
def get_millisecond():
    return int(time.time() * 1000)


class MMDetCustomModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        image_detector_config_file = model_cfg.IMAGE_DETECTOR_CONFIG_FILE
        image_detector_weight_file = model_cfg.IMAGE_DETECTOR_WEIGHTS_FILE
        self.use_gt = model_cfg.get("USE_GT", False)
        if(self.use_gt):
            self.val_activated = model_cfg.VAL_ACTIVATED
            self.lower_score = model_cfg.LOWER_SCORE
            self.higher_score = model_cfg.HIGHER_SCORE
            self.propotion = model_cfg.PROPOTION
        deploy_cfg = model_cfg.get("DEPLOY_CFG", None)
        if deploy_cfg is not None:
            self.speed_up = True
            detector_2d_cfg = model_cfg.IMAGE_DETECTOR_CONFIG_FILE
            device = 'cuda'
            backend_model = model_cfg.BACKEND_MODEL
            deploy_cfg, detector_2d_cfg = load_config(deploy_cfg, detector_2d_cfg)
            self.task_processor = build_task_processor(detector_2d_cfg, deploy_cfg, device)
            self.model = self.task_processor.build_backend_model(backend_model)
            self.input_shape = get_input_shape(deploy_cfg)
        else:
            self.image_model = init_detector(image_detector_config_file, image_detector_weight_file, device='cuda:0')
            self.speed_up = False
        # self.total_time = 0
        # self.total_cnt = 0
        
        
    def forward(self, data_dict):
        #* 图像
        images = data_dict['images']
        #! 保存图片备用
        # import cv2
        # img = images[0]
        # for x1, y1, x2, y2 in data_dict["gt_boxes2d"][0]:
        #     cv2.rectangle(img, (floor(x1), floor(y1)), (ceil(x2), ceil(y2)), color = (0, 0, 255))
        # cv2.imwrite("image.png", img)
        if(self.use_gt):
            if(self.training or self.val_activated):
                gt_boxes2d_num = data_dict["gt_boxes2d"].shape[1]
                gt_boxes2d_scores = ((self.higher_score-self.lower_score)*torch.rand(1, gt_boxes2d_num, 1)+self.lower_score).cuda()
                random_indices = torch.randperm(gt_boxes2d_num)
                selected_indices = random_indices[:int(gt_boxes2d_num * self.propotion)]
                gt_boxes2d = data_dict["gt_boxes2d"][:, selected_indices, :]
                gt_boxes2d_scores = gt_boxes2d_scores[:, selected_indices, :]
                gt_boxes2d = torch.cat([gt_boxes2d, gt_boxes2d_scores], dim=-1)
        images_results = []
        for image in images:
            if self.speed_up:
                # start_time = get_millisecond()
                model_inputs, _ = self.task_processor.create_input(image, self.input_shape)
                with torch.no_grad():
                    image_result = self.model.test_step(model_inputs)[0]
                # end_time = get_millisecond()
                # self.total_time += end_time-start_time
                # self.total_cnt+=1
                # print("inner detector2d inference time: {}".format(end_time-start_time))
            else:
                image_result = inference_detector(self.image_model, image)

            image_result_bboxes = image_result.pred_instances.bboxes
            image_result_scores = image_result.pred_instances.scores.unsqueeze(-1)
            image_result_labels = image_result.pred_instances.labels
            valid_mask = (image_result_labels==0)
            image_result_bboxes = image_result_bboxes[valid_mask]
            image_result_scores = image_result_scores[valid_mask]
            cur_image_result = torch.cat([image_result_bboxes, image_result_scores], dim=-1).unsqueeze(0)
            images_results.append(cur_image_result)
        boxes2d_by_detector = torch.cat(images_results,dim=0)
        if self.use_gt:
            if(self.training or self.val_activated):
                if boxes2d_by_detector.shape[1] + gt_boxes2d_num > 100:
                    drop_num = boxes2d_by_detector.shape[1] + gt_boxes2d_num-100
                    boxes2d_by_detector = boxes2d_by_detector[:, :-drop_num, :]
                    boxes2d_by_detector = torch.cat([boxes2d_by_detector, gt_boxes2d], dim=1)
                else:
                    boxes2d_by_detector = torch.cat([boxes2d_by_detector, gt_boxes2d], dim=1)
        #! 保存图片备用
        # import cv2
        # img = images[0]
        # for x1, y1, x2, y2, _ in boxes2d_by_detector[0]:
        #     cv2.rectangle(img, (floor(x1), floor(y1)), (ceil(x2), ceil(y2)), color = (0, 0, 255))
        # cv2.imwrite("image.png", img)
        # boxes2d_by_detector = torch.cat([boxes2d'by_detector, gt_boxes2d], dim=1)
        #* 填充到100个
        boxes2d_by_detector_num = boxes2d_by_detector.shape[1]
        added_num = 100-boxes2d_by_detector_num
        added_tensor = torch.zeros([1, added_num, 5], device=0)
        boxes2d_by_detector = boxes2d_by_detector.cuda()
        boxes2d_by_detector = torch.cat([boxes2d_by_detector, added_tensor], dim=1)
        data_dict["results_2d"] = boxes2d_by_detector
        return data_dict