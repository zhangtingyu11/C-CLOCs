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


def make_json_dict(imgs, anns):
	imgs_dict = {}
	anns_dict = {}
	for ann in anns:
		image_id = ann["image_id"]
		if not image_id in anns_dict:
			anns_dict[image_id] = []
			anns_dict[image_id].append(ann)
		else:
			anns_dict[image_id].append(ann)
	
	for img in imgs:
		image_id = img['id']
		imgs_dict[image_id] = img['file_name']

	return imgs_dict, anns_dict

class Segment_Ground_Truth_KITTI:
    def __init__(self, cfg_file) -> None:
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
        
        #* 存储当前数据集分割的frame_id的文件
        imagesets_path = self.data_root / 'ImageSets' / (self.split + '.txt')
        with open(imagesets_path, 'r') as f:
            lines = f.readlines()
        self.sample_idxs = [line.strip() for line in lines]
        
        #*当前选择的类别类型
        self.choosen_class = self.cfg['CHOOSEN_CLASSES']
        
        #* 数据来源
        self.data_source = self.cfg["DATA_SOURCE"]
        #* KINS数据集文件路径
        kins_json_path = self.cfg["KINS_JSON_PATH"]
        anns = cvb.load(kins_json_path)
        imgs_info = anns['images']
        anns_info = anns["annotations"]
        self.kins_iou_thresh = self.cfg["KINS_IOU_THRESH"]

        imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
        self.kins_database = {}
        for img_id in anns_dict.keys():
            img_name = imgs_dict[img_id]
            frame_id, _ = img_name.split('.')
            anns = anns_dict[img_id]
            self.kins_database[frame_id] = anns
        
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
            min_mask_region_area = 1000)
        
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
        
        #* 存储KITTI数据库的文件, 由OpenPCDet生成
        dbfile = self.cfg['OPENPCDET_INFO_PATH']
        #* 要存储的数据的哈希表
        #* key: 文件名字, 形如000000_Pedestrian_0, 用于记录当前这个物体的唯一标识
        #* val: 一个字典,目前只有一个key, 
        #*      'num_points_in_gt': 当前这个物体中包含多少个点
        self.db_dict = {}
        with open(dbfile, 'rb') as f:
            #* 加载文件, info是个字典
            #* key: 类别名称, value:一个列表, 列表中的每个元素都是个字典, 存储某个ground truth的信息
            info = pickle.load(f)
            #* 遍历字典中的每个值(values就是一个列表)
            for values in info.values():
                #* 遍历列表
                for value in values:
                    #* value['path']的值形如'gt_database/000000_Pedestrian_0.bin', 用split函数对/作为分隔符取后半部分
                    filename = value['path'].split('/')[-1]
                    #* 再对000000_Pedestrian_0.bin用.作为分隔符取前半部分
                    filename = filename.split('.')[0]
                    self.db_dict[filename] = {}
                    self.db_dict[filename]['num_points_in_gt'] = value['num_points_in_gt']

        
    @staticmethod
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.safe_load(f)
        return config
        
    def read_image(self, frame_id):
        #* 加上.png后缀
        image_name = str(frame_id).zfill(6)+'.png'
        #* 设置图像的路径
        if self.split in ['train', 'val', 'trainval']:
            image_path = self.data_root / 'training' / 'image_2' /  image_name
        else:
            image_path = self.data_root / 'testing' / 'image_2' /  image_name
        #* 读取图像
        self.image = cv2.imread(str(image_path))
        
    def read_calib(self, frame_id):
        calib_name = frame_id+'.txt'
        calib_path = self.data_root / 'training' / 'calib' /  calib_name
        self.calib = calibration_kitti.Calibration(calib_path)
    
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

        img=cv2.rectangle(img, (floor(tlx), floor(tly)), (ceil(brx), ceil(bry)), (0, 255, 0), 3)
        cv2.circle(img, (round(centerx), round(centery)), radius = 4, color=(0, 0, 255), thickness=-1)
        return img

    def add_boxes(self, boxes):
        for box in boxes:
            self.add_box(box)
    
    def show(self):
        plt.axis('off')
        plt.show() 

    def save_img(self, img, frame_id, idx, obj, data_root=None, split=None, save_to_database=True, angle_idx=None, range_idx=None):
        if split is None:
            split = self.split
        if data_root is None:
            data_root = self.data_root
        if(self.data_source == "SAM"):
            save_dir_name = "image_gt_database_" + str(split) 
        elif(self.data_source == "KINS"):
            save_dir_name = "image_gt_database_" + str(split) + "KINS"
        save_dir = data_root / save_dir_name
        
        image_name_list = [frame_id, obj.cls_type, str(idx)]
        #* 如果没有这个目录就创建这个目录
        if(not os.path.exists(str(save_dir))):
            os.makedirs(save_dir)
        
        #* 保存的地址
        save_address = data_root / save_dir_name / ('_'.join(image_name_list) + '.png')
        #* 将图片转化为BGRA
        result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        #* 保存图片
        cv2.imwrite(str(save_address), result)
        #* 将数据填写到数据库文件中
        if(save_to_database and angle_idx is not None and range_idx is not None):
            added_info = {}
            added_info['image_path'] = '_'.join(image_name_list) + '.png'
            added_info['difficulty'] = obj.level
            added_info['num_points_in_gt'] = self.db_dict['_'.join(image_name_list)]["num_points_in_gt"]
            self.database[obj.cls_type][angle_idx][range_idx].append(added_info)
    
    def custom_save_img(self, img, image_name, image_type = 'png', root = None):
        image_fullname = '.'.join([image_name, image_type])
        if root is not None:
            save_address = '/'.join([root, image_fullname])
        else:
            save_address = image_fullname
        cv2.imwrite(save_address, img)
        
    def save_single_scene(self, frame_id, masks):
        assert self.split in ['train', 'val', 'trainval']
        #* 根据frame_id填写标注文件路径
        label_txt = self.data_root / 'training' / 'label_2' / (frame_id+'.txt')
        #* 根据标注文件获取GT信息
        objects = get_objects_from_label(str(label_txt))
        
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
        
        #* 遍历列表中的索引和值
        for idx, obj in enumerate(objects):
            #* 需要要求物体无遮挡, 无截断, 并且类别属于所属的类别
            if(obj.occlusion>0 or obj.cls_type not in self.choosen_class or obj.truncation > 0):
                continue
            
            #* 标签中2D包围框左上角和右下角点的x坐标
            #! 图像坐标系在此定义为竖直向下为x轴, 水平向右为y轴
            #! KITTI box2d存储的是(左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标)
            #! KITTI的坐标系是水平向右为x轴, 竖直向下为y轴, 和之前定义的正好相反, 因此需要把x, y对调
            top_left_y, top_left_x, bottom_right_y, bottom_right_x = obj.box2d
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
            res = self.judge_valid(final_mask, obj.box2d)
            if(res):
                #* 相机坐标系下包围框的x, y, z
                x_in_camera, y_in_camera, z_in_camera = obj.loc
                #* 构成相机坐标系下的3D包围框
                box_3d_in_camera = np.array([[x_in_camera, y_in_camera, z_in_camera,
                                        obj.l, obj.w, obj.h, obj.ry]])
                box_3d_in_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box_3d_in_camera, self.calib)
                x, y, z, dx, dy, dz, heading = box_3d_in_lidar[0]
                
                #* 计算角度, 计算的是和水平向右的水平线的夹角, 顺时针为正
                degree = 180 - np.arctan2(x, y) * 180 / np.pi
                #* 从左往右序号分别是0, 1, 2 ...
                angle_idx = int((degree-(90-self.angle_limit))/self.angle_interval)
                if(angle_idx < 0 or angle_idx >= self.angle_num):
                    continue
                
                #* 计算到相机的距离
                dis = np.sqrt(x*x + y*y)
                range_idx = int(dis/(self.range_interval))
                if(range_idx < 0 or range_idx >= self.range_num):
                    continue
                top_left_x = floor(top_left_x)
                top_left_y = floor(top_left_y)
                bottom_right_x = ceil(bottom_right_x)
                bottom_right_y = ceil(bottom_right_y)
                self.save_img(final_mask[top_left_x:bottom_right_x+1, top_left_y:bottom_right_y+1], 
                              frame_id, 
                              idx, 
                              obj, 
                              angle_idx = angle_idx, 
                              range_idx = range_idx)
        self.logger.info('Frame id {} Finished'.format(frame_id))
    
    def save_single_scene_kins(self, frame_id, anns):
        assert self.split in ['train', 'val', 'trainval']
        label_txt = self.data_root / 'training' / 'label_2' / (frame_id+'.txt')
        #* 根据标注文件获取GT信息
        objects = get_objects_from_label(str(label_txt))
        
        image_height, image_width, _ = self.image.shape
        final_mask = np.zeros((image_height, image_width, 4), dtype=np.uint8)
        
        #* 遍历列表中的索引和值
        for idx, obj in enumerate(objects):
            #* 需要要求物体无遮挡, 无截断, 并且类别属于所属的类别
            if(obj.occlusion>0 or obj.cls_type not in self.choosen_class or obj.truncation > 0):
                continue
            
            #* 标签中2D包围框左上角和右下角点的x坐标
            #! 图像坐标系在此定义为竖直向下为x轴, 水平向右为y轴
            #! KITTI box2d存储的是(左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标)
            #! KITTI的坐标系是水平向右为x轴, 竖直向下为y轴, 和之前定义的正好相反, 因此需要把x, y对调
            top_left_y, top_left_x, bottom_right_y, bottom_right_x = obj.box2d
            max_iou = 0
            for ann in anns:
                tlx, tly, width, height = ann["bbox"]
                iou = self.compute_iou([tlx, tly, tlx+width, tly+height], obj.box2d)
                if(iou < max_iou):
                    continue
                max_iou = iou
                amodal_rle = maskUtils.frPyObjects(ann['segmentation'], image_height, image_width)
                amodal_ann_mask = maskUtils.decode(amodal_rle)[:, :, 0].astype(np.bool_)
            if(max_iou >= self.kins_iou_thresh):
                final_mask[amodal_ann_mask, :3] = self.image[amodal_ann_mask]
                final_mask[amodal_ann_mask, 3] = 255
                
                #* 相机坐标系下包围框的x, y, z
                x_in_camera, y_in_camera, z_in_camera = obj.loc
                #* 构成相机坐标系下的3D包围框
                box_3d_in_camera = np.array([[x_in_camera, y_in_camera, z_in_camera,
                                        obj.l, obj.w, obj.h, obj.ry]])
                box_3d_in_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box_3d_in_camera, self.calib)
                x, y, z, dx, dy, dz, heading = box_3d_in_lidar[0]
                
                #* 计算角度, 计算的是和水平向右的水平线的夹角, 顺时针为正
                degree = 180 - np.arctan2(x, y) * 180 / np.pi
                #* 从左往右序号分别是0, 1, 2 ...
                angle_idx = int((degree-(90-self.angle_limit))/self.angle_interval)
                if(angle_idx < 0 or angle_idx >= self.angle_num):
                    continue
                
                #* 计算到相机的距离
                dis = np.sqrt(x*x + y*y)
                range_idx = int(dis/(self.range_interval))
                if(range_idx < 0 or range_idx >= self.range_num):
                    continue
                top_left_x = floor(top_left_x)
                top_left_y = floor(top_left_y)
                bottom_right_x = ceil(bottom_right_x)
                bottom_right_y = ceil(bottom_right_y)
                self.save_img(final_mask[top_left_x:bottom_right_x+1, top_left_y:bottom_right_y+1], 
                              frame_id, 
                              idx, 
                              obj, 
                              angle_idx = angle_idx, 
                              range_idx = range_idx)
        self.logger.info('Frame id {} Finished'.format(frame_id))
                
                
    def generate_database_images(self):
        #* 创建database, 是一个字典
        #* key: 类别
        #* value: 一个三维列表[self.angle_num * self.range_num * 0]
        self.database = {
            'Car':[ [ [] for _ in range(self.range_num) ] for _ in range(self.angle_num) ],
            'Pedestrian': [ [ [] for _ in range(self.range_num) ] for _ in range(self.angle_num) ], 
            'Cyclist': [ [ [] for _ in range(self.range_num) ] for _ in range(self.angle_num) ]
        }
        for frame_id in self.sample_idxs:
            self.read_image(frame_id)
            self.read_calib(frame_id)
            if(self.data_source == "SAM"):
                masks = self.generate_mask()
                self.save_single_scene(frame_id, masks)
            elif(self.data_source == "KINS"):
                if(frame_id not in self.kins_database):
                    continue
                annos = self.kins_database[frame_id]
                self.save_single_scene_kins(frame_id, annos)
                
        if(self.data_source == "SAM"):
            save_address = self.data_root / 'image_database_train.pkl'
            with open(str(save_address), 'wb') as f:
                pickle.dump(self.database, f)
        elif(self.data_source == "KINS"):
            save_address = self.data_root / 'image_database_train_kins.pkl'
            with open(str(save_address), 'wb') as f:
                pickle.dump(self.database, f)
            
    def judge_valid(self, image, label, iou = 0.7):
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
                        3)
            cv2.rectangle(black_white_close, 
                    (floor(left), floor(top)), (ceil(right), ceil(bottom)),
                    (0, 0, 255), 
                    3)
            self.custom_save_img(black_white_close, 'gt_and_pred')
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
    sgtk = Segment_Ground_Truth_KITTI("tools/cfgs/dataset_configs/database_generate_kitti.yaml")
    # #! 34
    # sgtk.read_image(34)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((46.17, 196.15, 328.40, 286.09), masks, "generate_black_and_white_image_after_close")
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (46.17, 196.15, 328.40, 286.09))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 3
    # sgtk.read_image(3)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((614.24, 181.78, 727.31, 284.77), masks, "generate_black_and_white_image_after_close")
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (614.24, 181.78, 727.31, 284.77))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    #! 29
    # sgtk.read_image(29)
    # masks = sgtk.generate_mask()
    # sgtk.show_mask_with_black_and_white((652.31, 174.94, 690.16, 204.97), masks, "generate_black_and_white_image_after_close")
    # img = sgtk.show_anns(masks)
    # img = sgtk.add_box(img, (652.31, 174.94, 690.16, 204.97))
    # sgtk.custom_save_img(img, 'sam_mask')
    
    # sgtk.show()
    
    sgtk.generate_database_images()

