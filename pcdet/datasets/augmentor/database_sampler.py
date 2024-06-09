import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
import random
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common
import cv2
import yaml

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

# def draw_rectangle(origin_image, new_right, new_left, new_bottom, new_top, color = (0, 0, 255), alpha = 0.7):
#     create = np.zeros((new_bottom-new_top, new_right-new_left, 3), dtype=np.uint8)
#     create[:, :, 0] = color[0]
#     create[:, :, 1] = color[1]
#     create[:, :, 2] = color[2]
#     img_add = cv2.addWeighted(origin_image[new_top:new_bottom, new_left:new_right, 0:3], alpha ,create, 1-alpha, 0)
#     origin_image[new_top:new_bottom, new_left:new_right, 0:3] = img_add
#     cv2.rectangle(origin_image, (new_left, new_top), (new_right, new_bottom), color = color, thickness=2)
#     return origin_image
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
    
class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            if not db_info_path.exists():
                assert len(sampler_cfg.DB_INFO_PATH) == 1
                sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']

            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            # if self.logger is not None:
            #     self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                # if self.logger is not None:
                #     self.logger.info('Database filter by min points %s: %d => %d' %
                #                      (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        #* 确保每次每次能把所有的样本全部采到
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int_)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int_)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int_)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int_)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float_)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int_)
        for _order in paste_order:
            _box2d = boxes2d[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = crop_feat[_order]
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] > 0).astype(np.int_)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1

        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict

    def collect_image_crops_kitti(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int_)

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        # raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int_)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int_)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int_)

        # img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255
        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]]
        

        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_kitti(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iou2d1 = box_utils.pairwise_iou(sampled_boxes2d, existed_boxes2d).cpu().numpy()
        iou2d2 = box_utils.pairwise_iou(sampled_boxes2d, sampled_boxes2d).cpu().numpy()
        iou2d2[range(sampled_boxes2d.shape[0]), range(sampled_boxes2d.shape[0])] = 0
        iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

        ret_valid_mask = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                         (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                         (valid_mask))

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None

        if self.img_aug_type == 'kitti':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_kitti(data_dict, sampled_boxes, valid_mask)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'kitti':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int_)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int_)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        if self.img_aug_type == 'kitti':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_kitti(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'kitti':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int_)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        else:
            raise NotImplementedError
        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            # data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                #* 确保添加的物体个数+原有的物体个数不会超过限制的阈值
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                if self.img_aug_type is not None:
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask)
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)

                valid_mask = valid_mask.nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')
        return data_dict


class DataFusionSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        assert self.img_aug_type == "clocs"
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)

        cfg_path = self.sampler_cfg["CFG_PATH"]
        self.cfg = self.cfg_from_yaml_file(cfg_path)
        #* LIDAR线数
        lidar_lines = self.cfg["LIDAR_LINES"]
        #* 每个angle块占据多少角度
        self.angle_interval = 360/lidar_lines
        #* 以竖直向前为0度, 角度范围为-self.angle_limit ~ self.angle_limit
        self.angle_limit = self.cfg["ANGLE_LIMIT"]
        #* 要分多少个角度块
        self.angle_num = int(self.angle_limit*2/self.angle_interval)
        
        #* 距离范围是0~self.range_limit
        self.range_limit = self.cfg["RANGE_LIMIT"]
        #* 要分多少个距离块
        self.range_num = self.cfg["RANGE_NUM"]
        #* 每个range块占据多少距离
        self.range_interval = self.range_limit/self.range_num
        
        self.iof_thresh = self.cfg["IOF_THRESH"]
        
        self.height_limit = self.cfg["HEIGHT_LIMIT"]
        self.start_angle = 90-self.angle_limit
        
        self.logger = logger
        self.use_shared_memory = False
        

        db_info_path = sampler_cfg.DB_INFO_PATH[0]
        db_info_path = self.root_path.resolve() / db_info_path
        with open(str(db_info_path), 'rb') as f:
            infos = pickle.load(f)
            self.db_infos = infos

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        # self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None
        self.gt_database_data_key = None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            infos = self.db_infos[class_name]
            m, n = len(infos), len(infos[0])
            pointer = [[0] * n for _ in range(m)]
            indices = [[0] * n for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    indices[i][j] = np.arange(len(infos[i][j]))
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': pointer,
                'indices': indices
            }

    @staticmethod
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.safe_load(f)
        return config
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        for key, dinfos in db_infos.items():
            m, n = len(dinfos), len(dinfos[0])
            for i in range(m):
                for j in range(n):
                    filter_infos = []
                    pre_len = len(dinfos[i][j])
                    for info in dinfos[i][j]:
                        if info['difficulty'] not in removed_difficulty:
                            filter_infos.append(info)
                    db_infos[key][i][j] = filter_infos
                    # if self.logger is not None:
                    #     self.logger.info('Database filter by difficulty %s: %d => %d' % (key+"{}, {}".format(i, j), pre_len, len(db_infos[key][i][j])))
        return db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                infos = db_infos[name]
                m, n = len(infos), len(infos[0])
                for i in range(m):
                    for j in range(n):
                        filtered_infos = []
                        pre_len = len(infos[i][j])
                        for info in infos[i][j]:
                            if info['num_points_in_gt'] >= min_num:
                                filtered_infos.append(info)
                        db_infos[name][i][j] = filtered_infos
                        # if self.logger is not None:
                        #     self.logger.info('Database filter by min points %s: %d => %d' %
                        #                     (name+"{},{}".format(i, j), pre_len, len(filtered_infos)))
        return db_infos
    
    def filter_lidar_points(self, points=None, calib=None, road_planes=None):
        lidar_points = points
        lidar_calib = calib
        lidar_planes = road_planes
        a, b, c, d = lidar_planes
        points_rect = lidar_calib.lidar_to_rect(lidar_points[:, :3])
        points_rect_height = (-d - a * points_rect[:, 0] - c * points_rect[:, 2]) / b
        valid_mask = points_rect_height >= points_rect[:, 1]+self.height_limit
        points_np = lidar_points[valid_mask]
        return points_np
    
    # def sample_with_fixed_number(self, class_name, sample_group):
    #     """
    #     Args:
    #         class_name:
    #         sample_group:
    #     Returns:

    #     """
    #     sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
    #     #* 确保每次每次能把所有的样本全部采到
    #     if pointer >= len(self.db_infos[class_name]):
    #         indices = np.random.permutation(len(self.db_infos[class_name]))
    #         pointer = 0

    #     sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
    #     pointer += sample_num
    #     sample_group['pointer'] = pointer
    #     sample_group['indices'] = indices
    #     return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int_)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int_)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int_)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int_)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float_)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int_)
        cropped_images = [0]*len(paste_order)
        # flag = False
        for _order in paste_order:
            _box2d = boxes2d[_order]
            # if _box2d[2]-_box2d[0] > 200:
            #     flag = True
            # if flag:
            #     cv2.imwrite("original_image.png", image)
            # if flag:
            #     cv2.imwrite("crop_feat.png", crop_feat[_order])
            added_image = cv2.resize(crop_feat[_order], (_box2d[2]-_box2d[0], _box2d[3]-_box2d[1]), interpolation=cv2.INTER_LINEAR)
            # if flag:
                # cv2.imwrite("resized_crop_feat.png", added_image)
            paste_image = copy.deepcopy((image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]]))
            if(added_image.shape[2] == 4):
                p_mask = added_image[:, :, 3]>0
            else:
                p_mask = added_image[:, :, 2]>=0
            paste_image[p_mask] = added_image[p_mask][:, :3]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = paste_image
            copy_image = cv2.GaussianBlur(image, (3, 3), 0)
            # if flag:
                # cv2.imwrite("blur_image.png", copy_image)
            #* 原图剪切下来的物体
            cropped_image = image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2], :]
            if _order >= gt_number:
                #* 模糊后的图剪切下来的物体
                added_image = copy_image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2], :]
                #* 原图的mask部分设置成模糊后的图的mask部分
                cropped_image[p_mask] = added_image[p_mask]
                # tranparent = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 1), dtype = np.uint8)
                # tranparent[p_mask] = 255
                # concat_image = np.concatenate([cropped_image, tranparent], axis = -1)
                # if flag:
                    # cv2.imwrite("cropped_blur_image.png", concat_image)
            cropped_images[_order] = cropped_image
            # image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2], :] = cropped_image
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]][p_mask] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]][p_mask] > 0).astype(np.int_)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]][p_mask] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1
        for _order in paste_order:
            _box2d = boxes2d[_order]
            cropped_image = cropped_images[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2], :] = cropped_image
            # image = draw_rectangle(image, _box2d[2], _box2d[0], _box2d[3], _box2d[1])
            # if flag:
                # cv2.imwrite("final_image.png", image)
                # exit(0)
        
        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict

    def collect_image_crops_clocs(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int_)

        # copy crops from images
        # img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        # raw_image = io.imread(img_path)
        # raw_image = raw_image.astype(np.float32)
        # raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int_)
        # new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        # raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int_)
        # raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        # raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        # if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
        #     new_center = new_box.reshape(2,2).mean(0)
        #     new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
        #     new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int_)

        # img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255
        img_crop2d = io.imread(str(self.root_path)+'/image_gt_database_train/'+info['path'])
        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_clocs(self, data_dict, sampled_boxes, existed_boxes):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iof = 0
        valid_flag = []
        for idx, sampled_box2d in enumerate(sampled_boxes2d):
            for obj in existed_boxes2d:
                #* 计算2D包围框的iou
                inter = self.compute_intersection(sampled_box2d, obj)
                obj_area = self.compute_area(obj)
                add_obj_area = self.compute_area(sampled_box2d)
                #* 更新iof
                if(inter/obj_area > iof):
                    iof = inter/obj_area
                if(inter/add_obj_area > iof):
                    iof = inter/add_obj_area
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[idx:idx+1, 0:7], existed_boxes[:, 0:7])
            if iof < self.iof_thresh and iou.max() == 0:
                valid_flag.append(True)
                existed_boxes2d = np.concatenate([existed_boxes2d, np.expand_dims(sampled_box2d, axis=0)], axis=0)
                existed_boxes = np.concatenate([existed_boxes, sampled_boxes[idx:idx+1, 0:7]], axis=0)
            else:
                valid_flag.append(False)
                
        valid_flag = np.array(valid_flag, dtype=np.bool_)

        ret_valid_mask = valid_flag

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask


    def compute_intersection(self, boxes_a, boxes_b):
        tlx, tly, brx, bry = boxes_a
        iw = (min(brx, boxes_b[2]) -
                max(tlx, boxes_b[0]))
        if iw > 0:
            #* 2D包围框和标签之间的高度的重叠的长度
            ih = (min(bry, boxes_b[3]) -
                    max(tly, boxes_b[1]))
            if ih > 0:
                return iw * ih
        return 0
    
    @staticmethod
    def compute_area(box):
        obj_tlx, obj_tly, obj_brx, obj_bry = box
        obj_area = (obj_brx - obj_tlx) * (obj_bry - obj_tly)
        return obj_area

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, existed_boxes):
        mv_height = None

        if self.img_aug_type == 'clocs':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_clocs(data_dict, sampled_boxes, existed_boxes)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'clocs':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int_)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int_)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        if self.img_aug_type == 'clocs':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_clocs(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'clocs':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int_)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        else:
            raise NotImplementedError
        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            # data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                pre, _ = info['path'].split('.')
                
                file_path = str(self.root_path) +'/gt_database/'+pre+'.bin'

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict
    
    def get_label_data(self, filename = None):
        with open(filename, 'r') as f:
            lines = f.readlines()
            objects = [Object3d(line) for line in lines]
            return objects
        
    def get_calib_data(self, filename=None):
        """获取标定数据

        Args:
            filename (_type_, optional): 标定文件. Defaults to None.

        Returns:
            _type_: 获取到的标定类
        """
        calib = calibration_kitti.Calibration(filename)
        return calib
    
    def get_plane_data(self, filename = None):
        with open(filename, 'r') as f:
            lines = f.readlines()
        plane_data = lines[-1].strip().split()
        plane_data = list(map(float, plane_data))
        return plane_data
    
    def get_lidar_data(self, filename=None):
        """读取lidar的bin数据

        Args:
            filename (_type_, optional): 如果是None, 就读取self.bin_file. Defaults to None.

        Returns:
            _type_: [N*4]的点云
        """
        lidar_points = np.fromfile(filename, dtype=np.float32)
        return lidar_points.reshape(-1, 4)    
        
    def sample_with_fixed_number(self, data_dict, class_name, points, occupied, angle_interval, start_angle, range_interval, sample_num):
        #* 计算非零的索引
        unoccupied_indexs = (occupied==0).nonzero()
        #* -> N * 2
        unoccupied_indexs = np.concatenate([np.expand_dims(unoccupied_indexs[0], axis=-1), np.expand_dims(unoccupied_indexs[1], axis=-1)], axis=-1)
        
        new_unoccupied_indexs = []
        for i, j in unoccupied_indexs:
            if self.db_infos[class_name][i][j]:
                new_unoccupied_indexs.append([i, j])
        unoccupied_indexs = np.array(new_unoccupied_indexs)
            
        #! 均匀采样
        # decay_factor = 0.05
        probabilities = np.array([1 for i in range(unoccupied_indexs.shape[0])])
        probabilities = probabilities/probabilities.sum()
        
        #* 采样可以放置的位置
        sample_num = min(sample_num, unoccupied_indexs.shape[0])
        indexs = np.random.choice(range(unoccupied_indexs.shape[0]), size=sample_num, replace=False, p=probabilities)
        choosen_indexs = unoccupied_indexs[indexs]
        #* 按照从远到近排序
        choosen_indexs = sorted(choosen_indexs, key=lambda x: x[1], reverse=True)

        database = self.db_infos[class_name]
        sampled_dict = []
        for angle_idx, range_idx in choosen_indexs:
            sampled_object = {}
            random_idx = random.randint(0, len(database[angle_idx][range_idx])-1)
            cur_filename = database[angle_idx][range_idx][random_idx]["image_path"]
            
            pre, _ = cur_filename.split('.')
            frame_id, cls_type, idx = pre.split('_')
            
            #* 读取这个样本的标签
            label_file = '../data/kitti/training/label_2/' + frame_id + '.txt'
            added_labels = self.get_label_data(label_file)
            
            #* 读取这个样本的标定文件
            calib_file = '../data/kitti/training/calib/' + frame_id + '.txt'
            calib = self.get_calib_data(calib_file)
            
            added_object = added_labels[int(idx)]
            added_object = np.array([*(added_object.loc), added_object.l, added_object.h, added_object.w, added_object.ry])
            added_object = np.expand_dims(added_object, axis=0)
            #* 转到lidar坐标系
            added_object = box_utils.boxes3d_kitti_camera_to_lidar(added_object, calib)
            
            added_object = added_object[0]
            sampled_object['box3d_lidar'] = added_object
            sampled_object['path'] = cur_filename
            sampled_object['num_points_in_gt'] = database[angle_idx][range_idx][random_idx]["num_points_in_gt"]
            sampled_object['image_idx'] = frame_id
            sampled_object['name'] = class_name
            sampled_dict.append(sampled_object)
        return sampled_dict

        
    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        calib = data_dict['calib']
        points = data_dict['points']
        road_plane = data_dict['road_plane']
        points = self.filter_lidar_points(points, calib, road_plane)  
              
        angle_limit = self.angle_limit
        

        #* 起始的角度和终止的角度
        start_angle, end_angle = 90-angle_limit, 90+angle_limit

        #* 计算每个角度区间大小
        angle_interval = self.angle_interval
        
        #* 计算每个距离区间的大小
        range_interval = self.range_interval
        
        #* occupy数组, 默认都是0表示未占用, 大小为 angle_num * range_num
        occupied = np.zeros((self.angle_num, self.range_num), dtype=np.uint8)
        for point in points:
            #* 计算距离lidar的距离, 并判断距离索引
            dis = np.sqrt(point[0] * point[0] + point[1] * point[1])
            #* 计算距离索引
            range_idx = int(dis/self.range_interval)
            #* 超过所有就不考虑
            if(range_idx < 0 or range_idx >= self.range_num): 
                continue
            
            #* 计算角度值，并判断角度索引
            degree = np.arctan2(point[0], -point[1]) * 180 / np.pi
            #! 会出现int(-0.5)=0的情况
            if degree < self.start_angle:
                continue
            #* angle_idx逆时针递增
            angle_idx = int((degree-self.start_angle)/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num): 
                continue
            occupied[angle_idx][range_idx] = 1
        
        for angle_idx in range(self.angle_num):
            vis_flag =False
            for range_idx in range(self.range_num):
                if(occupied[angle_idx][range_idx]):
                    vis_flag = True
                elif(vis_flag):
                    occupied[angle_idx][range_idx] = 2
              
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                #* 确保添加的物体个数+原有的物体个数不会超过限制的阈值
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(data_dict, class_name, points, occupied, angle_interval, start_angle, range_interval, int(self.sample_class_num[class_name]))

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'

                # iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                # iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                # iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                # iou1 = iou1 if iou1.shape[1] > 0 else iou2
                # valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                if self.img_aug_type is not None:
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, existed_boxes.copy())
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)

                valid_mask = valid_mask.nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')
        return data_dict
