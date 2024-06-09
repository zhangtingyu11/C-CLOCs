from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import yaml
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from math import *
from pathlib import Path
import logging
import pickle
from pcdet.utils.object3d_kitti import get_objects_from_label
from pcdet.utils import calibration_kitti, box_utils
import cvbase as cvb
import pycocotools.mask as maskUtils

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pcdet.datasets.nuscenes import nuscenes_utils
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from typing import *
from shapely.geometry import MultiPoint, box

class Segment_Ground_Truth_Nuscenes:
    def __init__(self, cfg_file) -> None:
        #* 初始化logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Generate Mask")
        
        #* 从配置文件中读取数据
        self.cfg = self.cfg_from_yaml_file(cfg_file)
        
        #* 数据集的划分, 如果找不到SPLIT, 则默认为"train"
        self.split = self.cfg['SPLIT']
        #* 必须要在这三个中间选
        assert self.split in ['train', 'val', 'trainval']
        
        #* 数据集的根目录
        data_root = self.cfg['DATA_ROOT']
        #* 将其转化为Path对象
        self.data_root = Path(data_root)
        
        #* 要提取的camera类型
        self.cam_types = self.cfg["CAM_TYPES"]
        
        #* 存储KITTI数据库的文件, 由OpenPCDet生成
        info_file = self.cfg['INFO_PATH']
        #* 要存储的数据的哈希表
        #* key: 文件名字, 形如000000_Pedestrian_0, 用于记录当前这个物体的唯一标识
        #* val: 一个字典,目前只有一个key, 
        #*      'num_points_in_gt': 当前这个物体中包含多少个点
        self.db_dict = {}
        
        #* nuscenes版本号
        version = self.cfg["VERSION"]
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        with open(info_file, 'rb') as f:
            self.infos = pickle.load(f)
            for idx, info in enumerate(self.infos):
                sample_idx = idx
                gt_names = info["gt_names"]
                num_lidar_pts = info["num_lidar_pts"]
                for i, gt_name in enumerate(gt_names):
                    #* 以{sample_idx}_{gt_name}_{在样本中的编号}进行存储
                    filename = "{}_{}_{}".format(sample_idx, gt_name, i)
                    self.db_dict[filename] = {}
                    self.db_dict[filename]['num_points_in_gt'] = num_lidar_pts[i]
        
        #*当前选择的类别类型
        self.choosen_class = self.cfg['CHOOSEN_CLASSES']
        
        #* 数据来源
        self.data_source = self.cfg["DATA_SOURCE"]

        #* SAM模型的权重文件
        sam_model_type = self.cfg["SAM_MODEL_TYPE"]
        sam_weight_file = self.cfg['SAM_WEIGHT_FILE']
        #* 初始化SAM模型
        sam = sam_model_registry[sam_model_type](checkpoint=sam_weight_file)
        #* 将sam模型放到第0块显卡上
        sam.to(device=0)
        #* 初始化mask_generator用于后续生成实例分割的结果
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            min_mask_region_area = 1000)
        
        #* LIDAR线数
        lidar_lines = self.cfg["LIDAR_LINES"]
        #* 每个angle块占据多少角度
        self.angle_interval = 360/lidar_lines
        #* 以竖直向前为0度, 角度范围为-self.angle_limit ~ self.angle_limit
        self.angle_limit = self.cfg["ANGLE_LIMIT"]
        #* 要分多少个角度块
        self.angle_num = int(self.angle_limit*2/self.angle_interval)
        
        #* 距离范围是-self.range_limit~self.range_limit
        self.range_limit = self.cfg["RANGE_LIMIT"]
        #* 要分多少个距离块
        self.range_num = self.cfg["RANGE_NUM"]
        #* 每个range块占据多少距离
        self.range_interval = self.range_limit/self.range_num

        
    @staticmethod
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.safe_load(f)
        return config
        
    def read_image(self, image_path):
        self.image = cv2.imread(str(image_path))
    
    def generate_mask(self):
        #* 根据实例分割的生成器生成掩码
        masks = self.mask_generator.generate(self.image)
        return masks
    
    def plot_img(self):
        plt.figure(figsize=(20,20))
        plt.imshow(self.image)
    
    def show_anns(self, anns):
        plt.figure(figsize=(20,20))
        plt.imshow(self.image)

        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        img = (img*255).astype(np.uint8)
        img = cv2.addWeighted(self.image, 1-0.35, img[:, :, :3], 0.35, 0)
        return img
            
    def add_box(self, img, box2d):
        tlx, tly, brx, bry = box2d
        centerx = (tlx+brx)/2
        centery = (tly+bry)/2

        img=cv2.rectangle(img, (floor(tlx), floor(tly)), (ceil(brx), ceil(bry)), (0, 255, 0), 2)
        cv2.circle(img, (round(centerx), round(centery)), radius = 4, color=(0, 0, 255), thickness=-1)
        return img

    def add_boxes(self, boxes):
        for box in boxes:
            self.add_box(box)
    
    def show(self):
        plt.axis('off')
        plt.show() 

    def save_img(self, img, frame_id, idx, gt_name, cam_type,
                 data_root=None, split=None):
        if split is None:
            split = self.split
        if data_root is None:
            data_root = self.data_root
        if(self.data_source == "SAM"):
            save_dir_name = "image_gt_database_" + str(split) +"_"+cam_type
        save_dir = data_root / save_dir_name
        
        image_name_list = [str(frame_id), gt_name, str(idx)]
        #* 如果没有这个目录就创建这个目录
        if(not os.path.exists(str(save_dir))):
            os.makedirs(save_dir)
        
        #* 保存的地址
        save_address = data_root / save_dir_name / ('_'.join(image_name_list) + '.png')
        #* 将图片转化为BGRA
        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        #* 保存图片
        cv2.imwrite(str(save_address), result)

    def custom_save_img(self, img, image_name, image_type = 'png', root = None):
        image_fullname = '.'.join([image_name, image_type])
        if root is not None:
            save_address = '/'.join([root, image_fullname])
        else:
            save_address = image_fullname
        cv2.imwrite(save_address, img)
        
    def post_process_coords(self, 
        corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
    ) -> Union[Tuple[float, float, float, float], None]:
        """Get the intersection of the convex hull of the reprojected bbox corners
        and the image canvas, return None if no intersection.

        Args:
            corner_coords (list[int]): Corner coordinates of reprojected
                bounding box.
            imsize (tuple[int]): Size of the image canvas.

        Return:
            tuple [float]: Intersection of the convex hull of the 2D box
                corners and the image canvas.
        """
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
        
    def save_single_scene(self, frame_id, masks, 
                          camera_intrinsic, pose_rec, cs_rec, 
                          gt_boxes_tokens, gt_names, 
                          cam_type, flag, angle_idxs, range_idxs):
        """保存单个样本

        Args:
            frame_id (_type_): 当前样本的frame_id
            masks (_type_): SAM生成的掩码
            camera_intrinsic (_type_): 相机内参
            pose_rec (_type_): 姿态记录
            cs_rec (_type_): 标定的传感器数据
            gt_boxes_tokens (_type_): gt包围框的token
            gt_names (_type_): gt包围框的类别
            cam_type (_type_): 相机的类别
            flag (_type_): 用于记录每个物体是不是可用的
            angle_idxs (_type_): 用于记录每个物体所落在的角度的索引
            range_idxs (_type_): 用于记录每个物体所落在的范围的索引
        """
        #* masks是一个列表, 列表中的每个元素是一个字典, 表示这个物体的分割结果
        #* 其中'segmentation'存储的是尺寸为H*W的bool值矩阵, 如果为True, 表示这个像素的位置是当前分割的物体, 否则不是
        img_height, img_width= masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]

        #* 将分割图合并到一个图里面, 不同的instance用不同的数字标注
        cnt = 1
        masked_img = np.zeros((img_height, img_width), dtype = np.int32)
        for ann in masks:
            seg = ann['segmentation']
            masked_img[seg] = cnt
            cnt+=1
            
        for idx, gt_boxes_token in enumerate(gt_boxes_tokens):
            #* 根据包围框的token获取当前包围框
            box = self.nusc.get_box(gt_boxes_token)
            #* 将物体转到ego坐标系
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)
            #* 将物体转到LiDAR坐标系
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)
            x, y, _ = box.center
            
            #* 当前物体的类别
            gt_name = gt_names[idx]
            #* 当前物体的标注信息
            ann_rec = self.nusc.get('sample_annotation', gt_boxes_token)    
            #* 当前物体的可见度
            visibility = ann_rec["visibility_token"]
            #* 需要满足可见度在80%~100%
            if(visibility!='4'):
                continue
            
            #* 计算角度, 计算的是和水平向左的水平线的夹角, 顺时针为正(0~360)
            degree = 180 - np.arctan2(x, y) * 180 / np.pi
            #* 转成和竖直向下的线的夹角
            degree = (degree+90)%360
            #* 从左往右序号分别是0, 1, 2 ...
            angle_idx = int(degree/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num):
                continue
            
            #* 计算到相机的距离
            dis = np.sqrt(x*x + y*y)
            range_idx = int(dis/(self.range_interval))
            if(range_idx < 0 or range_idx >= self.range_num):
                continue
            
            #* 包围框的3D角点
            corners_3d = box.corners()
            #* 判断点在不在传感器的前面
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            #* 选取在传感器前面的点
            corners_3d = corners_3d[:, in_front]

            #* 将3D包围框的角点投影到图像上
            corner_coords = view_points(corners_3d, camera_intrinsic,
                                        True).T[:, :2].tolist()
            
            #* 只保留落在图像上的点
            final_coords = self.post_process_coords(corner_coords)

            if final_coords is None:
                continue
            else:
                top_left_y, top_left_x, bottom_right_y, bottom_right_x = final_coords
            
            #* 计算2D包围框的中间像素点坐标(像素点是整数)
            center_x = int((top_left_x+bottom_right_x)/2)
            center_y = int((top_left_y+bottom_right_y)/2)
            
            #* 创建一个H * W * 4的透明图像, 作为最后裁剪前的图片(后面会赋值)
            final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            
            #* masked_img==masked_img[center_x][center_y]会获取一个H * W的bool数组
            #* 转成unit8方便后面乘255
            mask = (masked_img==masked_img[center_x][center_y]).astype(np.uint8)
            mask *= 255
            
            #* 创建一个kernel
            k = np.ones((5, 5), np.uint8)
            #* 重复20轮, 进行闭运算
            close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20).astype(np.bool_)
            
            #* 将close_mask的部分赋值成原图
            final_mask[close_mask, :3]= self.image[close_mask]
            #* 将close_mask部分的透明度设置成不透明
            final_mask[close_mask, 3] = 255
            
            #* 判断这个分割的范围是否合理
            res = self.judge_valid(final_mask, final_coords)
            if(res):
                top_left_x = floor(top_left_x)
                top_left_y = floor(top_left_y)
                bottom_right_x = ceil(bottom_right_x)
                bottom_right_y = ceil(bottom_right_y)
                self.save_img(final_mask[top_left_x:bottom_right_x+1, top_left_y:bottom_right_y+1], 
                            frame_id, 
                            idx, 
                            gt_name,
                            cam_type)
                flag[idx]=True
                range_idxs[idx] = range_idx
                angle_idxs[idx] = angle_idx
        self.logger.info('Frame id {} {} Finished'.format(frame_id, cam_type))
                
    def generate_database_images(self):
        #* 创建database, 是一个字典
        #* key: 类别
        #* value: 一个三维列表[self.angle_num * self.range_num * 0]
        self.database = {}
        for c in self.choosen_class:
            self.database[c] = [[[] for _ in range(self.range_num)] for _ in range(self.angle_num)]
        for frame_id, info in enumerate(self.infos):
            #* 获取当前帧的gt框的token
            gt_boxes_tokens = info["gt_boxes_token"]
            #* 获取gt框的类别名
            gt_names = info["gt_names"]
            #* 获取当前帧的token
            sample_token = info["token"]
            #* 获取当前帧的数据
            sample_rec = self.nusc.get("sample", sample_token)
            #* 记录每个gt框能否加进数据库中
            flag = np.array([False] * len(gt_boxes_tokens), dtype=np.bool_)
            #* 记录每个gt框的角度索引
            angle_idxs = [-1] * len(gt_boxes_tokens)
            #* 记录每个gt框的距离索引
            range_idxs = [-1] * len(gt_boxes_tokens)
            for cam_type in self.cam_types:
                #* 获取当前相机传感器的token
                cam_token = sample_rec["data"][cam_type]
                #* 获取当前传感器的样本数据
                sd_rec = self.nusc.get('sample_data', cam_token)
                #* 获取当前传感器的标定数据
                cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                #* 获取当前传感器的姿态数据
                pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
                #* 获取当前传感器的相机内参
                camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
                #* 当前相机图像路径
                image_path = self.data_root / sd_rec["filename"]
                self.read_image(image_path)
                masks = self.generate_mask()
                self.save_single_scene(frame_id, masks, 
                                       camera_intrinsic, pose_rec, 
                                       cs_rec, gt_boxes_tokens, gt_names,
                                       cam_type, flag, angle_idxs, range_idxs)
            for i, gt_name in enumerate(gt_names):
                #* 如果当前样本可以加入数据库中
                if flag[i]:
                    filename = '_'.join([str(frame_id), gt_name, str(i)])
                    added_info = {}
                    added_info['image_path'] = filename
                    added_info['num_points_in_gt'] = self.db_dict[filename]["num_points_in_gt"]
                    angle_idx = angle_idxs[i]
                    range_idx = range_idxs[i]
                    self.database[gt_name][angle_idx][range_idx].append(added_info)
                
        if(self.data_source == "SAM"):
            save_address = self.data_root / 'image_database_train.pkl'
            with open(str(save_address), 'wb') as f:
                pickle.dump(self.database, f)
            
    def judge_valid(self, image, label, iou = 0.75):
        #* 判断掩码的位置, findContours的输入是个二值图像
        conv_image = (image[:, :, 3:4]>0).astype(np.uint8)
        conv_image *= 255
        
        #* 寻找轮廓
        #* cv2.RETR_EXTERNAL 表示只检索最外层的轮廓，而不检索任何嵌套轮廓。如果有多个对象，这将返回每个对象的最外层轮廓。
        #* cv2.CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线段并仅保留它们的端点。例如，直立矩形轮廓由 4 个点编码。
        contours, _ = cv2.findContours(conv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #* 找到每个轮廓的最小矩形边框
        bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
        
        #* 矩形边框的最大面积
        max_area = 0
        #* 最大的矩形边框的坐标[左上角x, 左上角y, 宽, 高]
        max_rect = None

        #* 遍历所有轮廓的最小矩形框
        for rect in bounding_rects:
            #* 左上角x, 左上角y, 宽, 高
            x, y, w, h = rect
            #* 矩形框的面积
            area = w * h

            if area > max_area:
                max_area = area
                max_rect = rect
        x, y, w, h = max_rect
        #* 最大矩形框的左上角x, 左上角y, 右下角x, 右下角y
        tlx, tly, brx, bry = x, y, x+w, y+h
        
        return self.compute_iou([tlx, tly, brx, bry], label) >= iou
    
    def compute_iou(self, box_1, box_2):
        tlx, tly, brx, bry = box_1
        qbox_area = (brx-tlx)*(bry-tly)
        #* 2D包围框和标签之间的宽度的重叠的长度
        iw = (min(brx, box_2[2]) -
                max(tlx, box_2[0]))
        ua = 0
        if iw > 0:
            #* 2D包围框和标签之间的高度的重叠的长度
            ih = (min(bry, box_2[3]) -
                    max(tly, box_2[1]))
            if ih > 0:
                #* 并集面积 = 标签的2D框的面积+最大矩形框的面积-交集面积
                ua = (
                    (box_2[2] - box_2[0]) *
                    (box_2[3] - box_2[1]) + qbox_area - iw * ih)    
                return (iw * ih / ua)
        return 0
    
    def show_mask_with_black_and_white(self, box2d, masks, mode = None):
        top_left_y, top_left_x, bottom_right_y, bottom_right_x = box2d
        centerx = int((top_left_x+bottom_right_x)/2)
        centery = int((top_left_y+bottom_right_y)/2)
        
        img_height, img_width= masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1],
        
        cnt = 1
        masked_img = np.zeros((img_height, img_width), dtype = np.int32)
        for ann in masks:
            seg = ann['segmentation']
            masked_img[seg] = cnt
            cnt+=1
        
        if(mode == "generate_origin_black_and_white_image"):
            #! 生成初始的黑白的mask
            mask = (masked_img==masked_img[centerx][centery]).astype(np.uint8)
            black_white_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            black_white_mask *= 255
            plt.imshow(black_white_mask)
            self.custom_save_img(black_white_mask, 'black_white_original_mask')
        elif(mode == 'generate_origin_transparent_image'):
            #! 生成初始的mask后的透明图片
            final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            mask = (masked_img==masked_img[centerx][centery])
            final_mask[mask, :3]= self.image[mask]
            final_mask[mask, 3] = 255
            plt.imshow(final_mask)
            self.custom_save_img(final_mask, 'original_transparent_mask')
        elif(mode == "generate_black_and_white_image_after_close"):
            #! 经过闭运算后的黑白mask
            mask = (masked_img==masked_img[centerx][centery]).astype(np.uint8)
            mask *= 255
            mask.astype(np.uint8)
            k = np.ones((5, 5), np.uint8)
            close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20)
            black_white_close = np.repeat(close[:, :, np.newaxis], 3, axis=2)
            plt.imshow(black_white_close)
            self.custom_save_img(black_white_close, 'black_white_close_mask')
        
            #* 显示label和轮廓的包围框
            left, right, top, bottom = img_width-1, 0, img_height-1, 0
            for x in range(img_width):
                for y in range(img_height):
                    if(black_white_close[y][x][0] > 0):
                        left = min(left, x)
                        right = max(right, x)
                        top = min(top, y)
                        bottom = max(bottom, y)
            
            #* 画2D包围框, 因为opencv坐标系和定义的坐标系不一样
            cv2.rectangle(black_white_close, 
                        (floor(top_left_y), floor(top_left_x)), (ceil(bottom_right_y), ceil(bottom_right_x)),
                        (0, 255, 0), 
                        2)
            cv2.rectangle(black_white_close, 
                    (floor(left), floor(top)), (ceil(right), ceil(bottom)),
                    (0, 0, 255), 
                    2)
            self.custom_save_img(black_white_close, 'gt_and_pred', transparent=False)
            #* 计算iou
            tlx, tly, brx, bry = left, top, right, bottom
            qbox_area = (brx - tlx) * (bry - tly)
            iw = (min(brx, box2d[2]) -
                    max(tlx, box2d[0]))
            ua = 0
            ih = 0
            if iw > 0:
                ih = (min(bry, box2d[3]) -
                        max(tly, box2d[1]))
                if ih > 0:
                    ua = (
                        (box2d[2] - box2d[0]) *
                        (box2d[3] - box2d[1]) + qbox_area - iw * ih)    
            print("iou = {}".format(iw * ih / ua))
        elif(mode == "generate_transparent_image_after_close"):
            #! 闭运算后的mask后的透明图片
            final_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)
            mask = (masked_img==masked_img[centerx][centery]).astype(np.uint8)
            mask *= 255
            mask.astype(np.uint8)
            k = np.ones((5, 5), np.uint8)
            close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=20)
            close = close.astype(np.bool_)
            final_mask[close, :3]= self.image[close]
            final_mask[close, 3] = 255
            plt.imshow(final_mask)
            self.custom_save_img(final_mask, 'close_mask')
        
if __name__ == '__main__':
    sgtk = Segment_Ground_Truth_Nuscenes("tools/cfgs/dataset_configs/database_generate_nuscenes.yaml")
    #! 34
    # sgtk.read_image(34)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((46.17, 196.15, 328.40, 286.09), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (46.17, 196.15, 328.40, 286.09))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 3
    # sgtk.read_image(3)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((614.24, 181.78, 727.31, 284.77), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (614.24, 181.78, 727.31, 284.77))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 29
    # sgtk.read_image(29)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((652.31, 174.94, 690.16, 204.97), masks)
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (652.31, 174.94, 690.16, 204.97))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    # sgtk.show()
    
    sgtk.generate_database_images()

