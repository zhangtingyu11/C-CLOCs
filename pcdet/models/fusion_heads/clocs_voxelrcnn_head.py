import numpy as np
import torch.nn as nn
from ...utils.box_utils import boxes3d_lidar_to_kitti_camera, lidar_boxes_to_image_kitti_torch_cuda
from ...ops.clocs.clocs_utils import compute_clocs_iou_sparse
from ...ops.clocs.clocs_utils import cos_similarity
from ..dense_heads.anchor_head_template import AnchorHeadTemplate
import torch
import numba
import copy
import torch.nn.functional as F
from ...utils import loss_utils
import torch
from abc import ABCMeta, abstractmethod

from ..dense_heads.target_assigner.atss_target_assigner import ATSSTargetAssigner
from ..dense_heads.target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner, PredAxisAlignedTargetAssigner
def _sigmoid_cross_entropy_with_logits(logits, labels):
  loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits) # this is the original
  loss += torch.log1p(torch.exp(-torch.abs(logits)))
  loss_mask = (loss < 10000)
  loss_mask = loss_mask.type(torch.FloatTensor).cuda()
  loss = loss*loss_mask
  return loss

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  dense = torch.zeros(size).fill_(default_value)
  dense[indices] = indices_value

  return dense

class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    if ignore_nan_targets:
      target_tensor = torch.where(torch.isnan(target_tensor),
                                prediction_tensor,
                                target_tensor)
    return self._compute_loss(prediction_tensor, target_tensor, **params)

  @abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass

class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25):
    """Constructor.

    Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
      all_zero_negative: bool. if True, will treat all zero as background.
        else, will treat first label as background. only affect alpha.
    """
    self._alpha = alpha
    self._gamma = gamma

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    weights = weights.unsqueeze(2)
    if class_indices is not None:
      weights *= indices_to_dense_vector(class_indices,
            prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
    per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = torch.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = torch.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                              (1 - target_tensor) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor*per_entry_cross_ent)
    return focal_cross_entropy_loss * weights

class ResnetBlock(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(ResnetBlock, self).__init__()
    self.in_planes = in_planes
    self.out_planes = out_planes
    self.main_conv1 = nn.Conv2d(in_planes, out_planes, 1)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.main_conv2 = nn.Conv2d(out_planes, out_planes, 1)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    if(in_planes != out_planes):
      self.side_conv = nn.Conv2d(in_planes, out_planes, 1)
      self.side_bn = nn.BatchNorm2d(out_planes)
    self.relu3 = nn.ReLU(inplace=True)
  def forward(self, x):
    out = self.relu1(self.bn1(self.main_conv1(x)))
    out = self.bn2(self.main_conv2(out))
    if(self.in_planes == self.out_planes):
      ind_x = x
    else:
      ind_x = self.side_bn(self.side_conv(x))
    return self.relu3(ind_x+out)


class ClocsVoxelRCNNHead(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.model_cfg = model_cfg
        self.predict_boxes_when_training = model_cfg.get("PREDICT_BOXES_WHEN_TRAINING", False)
        
        num_filters = self.model_cfg.NUM_FILTERS
        num_filters = [input_channels] + num_filters
        
        self.fuse = []
        for i in range(1, len(num_filters)-1):
            if self.model_cfg.get("USE_RES_BLOCK", False):
              self.fuse.append(ResnetBlock(num_filters[i-1], num_filters[i]))
            else:
              self.fuse.append(nn.Conv2d(num_filters[i-1], num_filters[i], 1))
              self.fuse.append(nn.ReLU())
            
        #* 因为希望最后的结果不是全部大于0的, 所以没有加ReLU
        self.fuse.append(nn.Conv2d(num_filters[-2], num_filters[-1], 1))
        self.fuse = nn.Sequential(*self.fuse)
        if self.model_cfg.get("USE_CONTRA", False):
          #* 图像特征的提取器
          self.image_extractor = []
          img_num_filters = self.model_cfg.IMAGE_NUM_FILTERS
          img_num_filters = [self.model_cfg.IMAGE_INPUT_CHANNELS] + img_num_filters
          for i in range(1, len(img_num_filters)-1):
              self.image_extractor.append(nn.Conv2d(img_num_filters[i-1], img_num_filters[i], 1))
              self.image_extractor.append(nn.ReLU())
          self.image_extractor.append(nn.Conv2d(img_num_filters[-2], img_num_filters[-1], 1))
          self.image_extractor = nn.Sequential(*self.image_extractor)
          
          #* LiDAR特征提取器
          self.lidar_extractor = []
          lidar_num_filters = self.model_cfg.LIDAR_NUM_FILTERS
          lidar_num_filters = [self.model_cfg.LIDAR_INPUT_CHANNELS] + lidar_num_filters
          for i in range(1, len(lidar_num_filters)-1):
              self.lidar_extractor.append(nn.Conv2d(lidar_num_filters[i-1], lidar_num_filters[i], 1))
              self.lidar_extractor.append(nn.ReLU())
          self.lidar_extractor.append(nn.Conv2d(lidar_num_filters[-2], lidar_num_filters[-1], 1))
          self.lidar_extractor = nn.Sequential(*self.lidar_extractor)

    def forward(self, data_dict):
        #* 读取3D目标检测的结果
        preds = data_dict['batch_box_preds']
        #* 将3D目标检测器的预测结果存进去， 后续计算loss需要使用
        self.forward_ret_dict['preds_3d'] = data_dict['batch_cls_preds']
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
        #* 2D目标检测的结果
        boxes2d_by_detector = data_dict['results_2d']
        #* 2D目标检测的个数
        boxes_2d_num = boxes2d_by_detector.shape[1]
        cls_pred_list = []
        if self.model_cfg.USE_CONTRA:
          lidar_feature_expanded_list = []
          image_feature_expanded_list = []
        for box_2d_preds, box_2d_detector, scores_3d, dist_to_lidar_single, image_shape, lidar_pred in zip(box_preds_on_image, 
                                                                                    boxes2d_by_detector,
                                                                                    pred_3d_scores,
                                                                                    dis_to_lidar,
                                                                                    data_dict["image_shape"], 
                                                                                    preds):
            scores_2d = box_2d_detector[:, 4:5]
            box_2d_pos = box_2d_detector[:, :4]
            boxes_3d_num = box_2d_preds.shape[0]
            overlap = torch.zeros((boxes_3d_num, boxes_2d_num, 4), dtype = box_2d_detector.dtype, device = box_2d_detector.device)-1
            #* 用[iou, 3D目标检测分数, 2D目标检测分数, 到LiDAR的距离来填充]
            ious = compute_clocs_iou_sparse(box_2d_preds.contiguous(),
                                    box_2d_pos.contiguous(),
                                    scores_3d,
                                    scores_2d.contiguous(),
                                    dist_to_lidar_single,
                                    overlap)

            ious = ious.permute(2, 0, 1).view(1, 4, boxes_3d_num, boxes_2d_num)
            non_empty_mask = (ious[0, 0, :, :]!=-10).nonzero()
            # if non_empty_mask.shape[0]==0:
            #   self.cal_loss_flag = False
            # else:
            #   self.cal_loss_flag = True
            
            iou_condition = ious[0][0, :, :] > self.model_cfg.IOU_THRESH
            #* 找到每个3D物体中满足条件的2D物体的最大置信度
            max_confidences, _ = torch.max(torch.where(iou_condition, ious[0, 2, :, :], torch.zeros_like(ious[0, 2, :, :])), dim=1)
            self.forward_ret_dict['3d_max_confidence'] = max_confidences
            input_features = ious

            if self.model_cfg.USE_CONTRA:
              image_height, image_width = image_shape
              box_2d_detector_normalized = box_2d_detector.clone()
              box_2d_detector_normalized[:, 0]/=image_width
              box_2d_detector_normalized[:, 2]/=image_width
              box_2d_detector_normalized[:, 1]/=image_height
              box_2d_detector_normalized[:, 3]/=image_height
              lidar_pred_normalized = lidar_pred.clone()
              lidar_pred_normalized[:, 0]/=70.4
              lidar_pred_normalized[:, 1]/=80
              lidar_pred_normalized[:, 2]/=4
              lidar_data = torch.cat([lidar_pred_normalized, scores_3d], dim=-1).permute(0, 1).reshape(1, -1, preds.shape[1], 1)
              image_data = box_2d_detector_normalized.permute(0, 1).reshape(1, -1, 1, boxes2d_by_detector.shape[1])
              lidar_features = self.lidar_extractor(lidar_data).squeeze(0)
              image_features = self.image_extractor(image_data).squeeze(0)
              lidar_features = lidar_features.squeeze().transpose(0, 1).unsqueeze(1).contiguous()
              image_features = image_features.squeeze().transpose(0, 1).unsqueeze(0).contiguous()
              lidar_num = lidar_features.shape[0]
              image_num = image_features.shape[1]
              feature_dim = image_features.shape[-1]
              lidar_feature_expanded = lidar_features.expand(lidar_num, image_num, feature_dim)
              image_feature_expanded = image_features.expand(lidar_num, image_num, feature_dim) 

              lidar_feature_expanded_list.append(lidar_feature_expanded.view(1, *lidar_feature_expanded.shape))
              image_feature_expanded_list.append(image_feature_expanded.view(1, *image_feature_expanded.shape))

              sim_feature = torch.zeros(lidar_num, image_num, device = 0)
              cos_similarity(lidar_features, image_features, sim_feature)
              sim_feature = sim_feature[None, None, :, :]
              # sim_feature = F.cosine_similarity(lidar_feature_expanded, image_feature_expanded, -1)[None, None, :, :]
              input_features = torch.cat([input_features, sim_feature], dim=1)

            # if (self.cal_loss_flag):
            non_empty_input_features = (input_features[:, :, non_empty_mask[:, 0], non_empty_mask[:, 1]]).unsqueeze(2)
            new_res = self.fuse(non_empty_input_features)
            res = torch.zeros((1, 1, boxes_3d_num, boxes_2d_num), dtype=torch.float32, device=0)-100
            res[:, :, non_empty_mask[:, 0], non_empty_mask[:, 1]] = new_res.squeeze(2)
            # else:
            # res = torch.zeros((1, 1, boxes_3d_num, boxes_2d_num)).type(torch.float32).cuda()-100

            #* 相当于maxpooling
            output = torch.amax(res, dim = -1)
            output = output.squeeze().view(1,-1,1)
            cls_pred_list.append(output)
        cls_preds = torch.cat(cls_pred_list, dim=0)
        if self.model_cfg.USE_CONTRA:
          self.forward_ret_dict['lidar_features_expanded'] = torch.cat(lidar_feature_expanded_list, dim=0)
          self.forward_ret_dict['image_features_expanded'] = torch.cat(image_feature_expanded_list, dim=0)

          #* iou大于阈值的位置
          iou_mask = ious[:, 0:1, :, :] > self.model_cfg.CONTRA_MATCH_IOU

          # #* 根据置信度找到每行最大值的索引
          # iou = ious[0, 0, :, :]
          # max_iou_indices3d = torch.argmax(iou, dim=1)
          
          # lidar_label3d = torch.zeros(boxes_3d_num, boxes_2d_num, dtype=torch.bool)

          # #* 将每行中IOU大于阈值且置信度最大的位置置为1
          # row_indices3d = torch.arange(boxes_3d_num)
          # lidar_label3d[row_indices3d, max_iou_indices3d] = True
          # lidar_label3d = lidar_label3d.view(1, 1, boxes_3d_num, boxes_2d_num)
          self.forward_ret_dict['contrastive_label'] = iou_mask

        self.forward_ret_dict['cls_preds'] = cls_preds

        if self.training:
            gt_boxes = data_dict['gt_boxes'].detach().cpu().numpy()
            gt_boxes_classes = np.expand_dims(gt_boxes[:, :, -1], axis=-1)
            gt_boxes_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(gt_boxes[0], data_dict['calib'][0]), axis=0)
            gt_boxes_in_camera = np.concatenate([gt_boxes_in_camera, gt_boxes_classes], axis=-1)
            #* 在相机坐标系下的预测框
            preds_in_camera = np.expand_dims(boxes3d_lidar_to_kitti_camera(preds[0].detach().cpu().numpy(), data_dict['calib'][0]), axis=0)
            targets_dict = self.assign_targets(
                preds=preds_in_camera,
                gt_boxes=gt_boxes_in_camera
            )
            self.forward_ret_dict.update(targets_dict)
            
        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = cls_preds
            data_dict['batch_box_preds'] = preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
    
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
 
    def get_loss(self):
        tb_dict = {}
        cls_loss, tb_dict1 = self.get_cls_layer_loss()
        if self.model_cfg.USE_CONTRA:
          contra_loss, tb_dict2 = self.get_contra_loss()
          fusion_loss = cls_loss+contra_loss
          tb_dict.update(tb_dict2)
        else:
          fusion_loss = cls_loss
        tb_dict['fusion_loss'] = fusion_loss.item()
        tb_dict.update(tb_dict1)
        return fusion_loss, tb_dict
    
    def build_losses(self, losses_cfg):
        if self.model_cfg.USE_CONTRA:
          self.add_module(
              'contra_loss_func',
              loss_utils.CosineContrastiveLoss()
          )

    def get_cls_layer_loss(self):
        #* clocs预测的分数, sigmoid前
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        cls_preds = self.forward_ret_dict['cls_preds']
        if self.model_cfg.USE_LA:
          #* 每个3D物体和与其IOU大于阈值的2D物体的最大2D置信度
          max3d_confidence = self.forward_ret_dict['3d_max_confidence']
          #* 3D目标检测器的分数取出来求sigmoid
          sig_preds = (torch.sigmoid(self.forward_ret_dict['preds_3d'])).view(1, -1)
          #* clocs预测的负样本
          clocs_neg = (max3d_confidence<self.model_cfg.CLOCS_NEG_IOU_THRESH).view(1, -1)
          #* 3D目标检测器预测出来的负样本
          neg3d = sig_preds<self.model_cfg.CLOCS_NEG_IOU_THRESH
          #* 标签中的正样本
          positives = box_cls_labels > 0
          #* 如果两个都很低， 但是标签是正样本， 说明这个要变成-1
          box_cls_labels[clocs_neg & neg3d & positives]=-1
          
          #* 3D目标检测器预测出来的正样本
          pos3d = sig_preds>self.model_cfg.CLOCS_POS_IOU_THRESH
          #* clocs预测出来的正样本
          clocs_pos = (max3d_confidence>self.model_cfg.CLOCS_POS_IOU_THRESH).view(1, -1)
          #* 标签中的负样本
          negatives = box_cls_labels==0
          #* 如果预测出来都觉得是正样本， 但是其实是负样本， 也要设置成-1
          box_cls_labels[pos3d & clocs_pos & negatives]=-1
        
        #* 只关心标签大于等于0的, 大于0是正样本, =0是负样本
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        
        #* 构建one-hot编码
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        #* 使用focal loss
        focal_loss = SigmoidFocalClassificationLoss()

        #* 负样本的权重是1
        negative_cls_weights = negatives.type(torch.float32) * 1.0
        #* 正样本的权重也是1
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        #* 正样本的个数
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        
        #* 权重需要除正样本的个数
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        #* 用focal loss计算损失
        cls_losses = focal_loss._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]
        
        batch_size = int(cls_preds.shape[0])
        #* 最后的损失需要除以batch size
        cls_losses = cls_losses.sum()/batch_size

        cls_loss = cls_losses * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
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
            [preds.reshape([1, 1, 1, 1, 100, 7])], gt_boxes
        )
        return targets_dict
    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'PredAxisAlignedTargetAssigner':
            target_assigner = PredAxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner