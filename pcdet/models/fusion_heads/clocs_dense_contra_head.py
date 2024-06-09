import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes3d_lidar_to_kitti_camera, lidar_boxes_to_image_kitti_torch_cuda
from ...ops.clocs.clocs_utils import compute_clocs_iou_sparse
from ..dense_heads.anchor_head_template import AnchorHeadTemplate
from ...utils import loss_utils
import torch.nn.functional as F
import torch
import numba
import copy
@numba.jit(nopython=True,parallel=True)
def compute_clocs_iou_dense(boxes, query_boxes, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index, max_num):
    N = boxes.shape[0] #70400
    K = query_boxes.shape[0] #30
    ind=0
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1
                    if(ind >= max_num):
                        return overlaps, tensor_index, ind
                        

                elif k==K-1:
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n,0]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1
                    if(ind >= max_num):
                        return overlaps, tensor_index, ind
            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n,0]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n
                ind = ind+1
                if(ind >= max_num):
                    return overlaps, tensor_index, ind
    return overlaps, tensor_index, ind
class ClocsDenseContraHead(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.model_cfg = model_cfg
        
        num_filters = self.model_cfg.NUM_FILTERS
        num_filters = [input_channels] + num_filters
        
        self.fuse = []
        for i in range(1, len(num_filters)-1):
            self.fuse.append(nn.Conv2d(num_filters[i-1], num_filters[i], 1))
            self.fuse.append(nn.ReLU())
        self.fuse.append(nn.Conv2d(num_filters[-2], num_filters[-1], 1))
        self.fuse = nn.Sequential(*self.fuse)
        
        self.image_extractor = []
        img_num_filters = self.model_cfg.IMAGE_NUM_FILTERS
        img_num_filters = [self.model_cfg.IMAGE_INPUT_CHANNELS] + img_num_filters
        for i in range(1, len(img_num_filters)-1):
            self.image_extractor.append(nn.Conv2d(img_num_filters[i-1], img_num_filters[i], 1))
            self.image_extractor.append(nn.ReLU())
        self.image_extractor.append(nn.Conv2d(img_num_filters[-2], img_num_filters[-1], 1))
        # self.image_extractor.append(nn.ReLU())
        self.image_extractor = nn.Sequential(*self.image_extractor)
            
        self.lidar_extractor = []
        lidar_num_filters = self.model_cfg.LIDAR_NUM_FILTERS
        lidar_num_filters = [self.model_cfg.LIDAR_INPUT_CHANNELS] + lidar_num_filters
        for i in range(1, len(lidar_num_filters)-1):
            self.lidar_extractor.append(nn.Conv2d(lidar_num_filters[i-1], lidar_num_filters[i], 1))
            self.lidar_extractor.append(nn.ReLU())
        self.lidar_extractor.append(nn.Conv2d(lidar_num_filters[-2], lidar_num_filters[-1], 1))
        # self.lidar_extractor.append(nn.ReLU())
        self.lidar_extractor = nn.Sequential(*self.lidar_extractor)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        
        # self.maxpool_dim = self.model_cfg.MAXPOOL_DIM
        # self.maxpool = nn.MaxPool2d([self.model_cfg.MAXPOOL_DIM,1],1)
    def get_logits(self, image_features, lidar_features):
        # 计算image_features @ text_features.T相似度矩阵
        logits_per_image = self.logit_scale * image_features @ lidar_features.T
        logits_per_lidar = self.logit_scale * lidar_features @ image_features.T
        return logits_per_image, logits_per_lidar

    def forward(self, data_dict):
        preds = data_dict['batch_box_preds']
        #* 转成0~1之间
        pred_3d_scores = torch.sigmoid(data_dict['batch_cls_preds'])
        #! 计算包围框中心点到lidar的距离, clocs是计算包围框底部中心点到lidar的距离
        dis_to_lidar = torch.norm(preds[:, :, :2],p=2,dim=2,keepdim=True)/82.0
        
        calib_V2C_T = data_dict['calib_matrix_V2C_T']
        calib_R0_T = data_dict['calib_matrix_R0_T']
        calib_P2_T = data_dict['calib_matrix_P2_T']
        image_shape = data_dict['image_shape']
        img_height = image_shape[:, 0]
        img_width = image_shape[:, 1]
        
        #* 计算lidar坐标系下的包围框投影到图像上的坐标系
        box_preds_on_image = lidar_boxes_to_image_kitti_torch_cuda(preds, 
                                                                   calib_P2_T, 
                                                                   calib_R0_T,
                                                                   calib_V2C_T, 
                                                                   img_height, 
                                                                   img_width)

        boxes2d_by_detector = data_dict['results_2d'].detach().cpu().numpy()
        box_preds_on_image = box_preds_on_image.detach().cpu().numpy()
        pred_3d_scores = pred_3d_scores.detach().cpu().numpy()
        dis_to_lidar = dis_to_lidar.detach().cpu().numpy()
        cls_pred_list = []
        for box_2d_preds, box_2d_detector, scores_3d, dist_to_lidar_single in zip(box_preds_on_image, 
                                                                                    boxes2d_by_detector,
                                                                                    pred_3d_scores,
                                                                                    dis_to_lidar):
            scores_2d = box_2d_detector[:, 4:5]
            box_2d_detector = box_2d_detector[:, :4]
            boxes_3d_num = box_2d_preds.shape[0]
            boxes_2d_num = box_2d_detector.shape[0]
            overlap = torch.zeros((boxes_3d_num, boxes_2d_num, 4), dtype = box_2d_detector.dtype, device = box_2d_detector.device)-1
            ious = compute_clocs_iou_sparse(box_2d_preds.contiguous(),
                                    box_2d_detector.contiguous(),
                                    scores_3d,
                                    scores_2d.contiguous(),
                                    dist_to_lidar_single,
                                    overlap)

            ious = ious.permute(2, 0, 1).view(1, 4, boxes_3d_num, boxes_2d_num)
            lidar_data = torch.cat([preds[0], scores_3d], dim=-1).permute(0, 1).reshape(1, -1, preds.shape[1], 1)
            image_data = boxes2d_by_detector[0].permute(0, 1).reshape(1, -1, 1, boxes2d_by_detector.shape[1])
            lidar_features = self.lidar_extractor(lidar_data).squeeze(0)
            image_features = self.image_extractor(image_data).squeeze(0)
            lidar_features = lidar_features.squeeze().transpose(0, 1).unsqueeze(1)
            image_features = image_features.squeeze().transpose(0, 1).unsqueeze(0)
            lidar_num = lidar_features.shape[0]
            image_num = image_features.shape[1]
            feature_dim = image_features.shape[-1]
            lidar_feature_expanded = lidar_features.expand(lidar_num, image_num, feature_dim)
            image_feature_expanded = image_features.expand(lidar_num, image_num, feature_dim) 
            sim_feature = F.cosine_similarity(lidar_feature_expanded, image_feature_expanded, -1)[None, None, :, :]
            input_features = torch.cat([ious, sim_feature], dim=1)
            res = self.fuse(input_features)
                
            output = torch.amax(res, dim = -1)
            output = output.squeeze().reshape(1,-1,1)
            cls_pred_list.append(output)
        cls_preds = torch.cat(cls_pred_list, dim=0)
        
        # lidar_data = torch.cat([preds, pred_3d_scores], dim=-1).permute(0, 2, 1).reshape(1, -1, 1, preds.shape[1])
        # image_data = boxes2d_by_detector.permute(0, 2, 1).permute(0, 2, 1).reshape(1, -1, 1, boxes2d_by_detector.shape[1])
        # lidar_features = self.lidar_extractor(lidar_data)
        # image_features = self.image_extractor(image_data)
        self.forward_ret_dict['lidar_features_expanded'] = lidar_feature_expanded
        self.forward_ret_dict['image_features_expanded'] = image_feature_expanded
        
        #*获取label
        lidar_label = ious[:, 0:1, :, :]>self.model_cfg.CONTRA_MATCH_IOU
        self.forward_ret_dict['contrastive_label'] = lidar_label
        
        self.forward_ret_dict['cls_preds'] = cls_preds
        gt_boxes = data_dict['gt_boxes'].detach().cpu().numpy()
        gt_boxes_classes = np.expand_dims(gt_boxes[:, :, -1], axis=-1)
        gt_boxes_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(gt_boxes[0], data_dict['calib'][0]), axis=0)
        gt_boxes_in_camera = np.concatenate([gt_boxes_in_camera, gt_boxes_classes], axis=-1)
        preds_in_lidar = copy.deepcopy(preds)
        #* 在相机坐标系下的预测框
        preds_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(preds[0].detach().cpu().numpy(), data_dict['calib'][0]), axis=0)

        if self.training:
            targets_dict = self.assign_targets(
                preds=preds_in_camera,
                gt_boxes=gt_boxes_in_camera
            )
            self.forward_ret_dict.update(targets_dict)
            
        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = cls_preds
            data_dict['batch_box_preds'] = preds_in_lidar
            data_dict['cls_preds_normalized'] = False

        return data_dict
    
    def get_loss(self): 
        tb_dict = {}
        cls_loss, tb_dict1 = self.get_cls_layer_loss()
        contra_loss, tb_dict2 = self.get_contra_loss()
        
        fusion_loss = cls_loss+contra_loss
        tb_dict['fusion_loss'] = fusion_loss.item()
        tb_dict.update(tb_dict1)
        tb_dict.update(tb_dict2)
        return fusion_loss, tb_dict
    
    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        self.add_module(
            'contra_loss_func',
            loss_utils.CosineContrastiveLoss()
        )
    
    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
    
    def get_contra_loss(self):
        lidar_feature_expanded = self.forward_ret_dict['lidar_features_expanded']
        image_feature_expanded = self.forward_ret_dict['image_features_expanded']
        contra_label = self.forward_ret_dict['contrastive_label'].squeeze().int()
        contra_loss = self.contra_loss_func(lidar_feature_expanded, image_feature_expanded, contra_label)
        contra_loss = contra_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['contra_weight']
        tb_dict = {
            'contra_loss': contra_loss
        }
        return contra_loss, tb_dict
    
    def assign_targets(self, preds, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            [preds.reshape(*self.anchors[0].shape)], gt_boxes
        )
        return targets_dict
