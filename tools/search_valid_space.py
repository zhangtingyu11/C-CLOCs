import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import pickle
from pcdet.utils.box_utils import *
from pcdet.utils.object3d_kitti import get_objects_from_label
from pcdet.utils import calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes
from pathlib import Path
import cv2
import yaml
from math import *
from copy import *
import os

class SearchValidSpace:
    def __init__(self, cfg_file) -> None:
        #* 从配置文件中读取数据
        self.cfg = self.cfg_from_yaml_file(cfg_file)
        
        #* 创建画布
        fig_size = self.cfg["FIG_SIZE"]
        self.fig = plt.figure(figsize = fig_size)
        self.ax = self.fig.gca()
        
        #* 画图的中心点, 默认以(0, 0)为中心点
        self.center = self.cfg["CENTER"]
        
        #* 三种区域的颜色        
        self.direct_occupied_color = self.cfg['DIRECT_OCCUPIED_COLOR']
        self.indirect_occupied_color = self.cfg['INDIRECT_OCCUPIED_COLOR']
        self.free_space_color = self.cfg['FREE_SAPCE_COLOR']
        
        #* 数据集根目录
        data_root = self.cfg["DATA_ROOT"] 
        self.data_root = Path(data_root)
        
        self.choosen_frame_id = self.cfg["CHOOSEN_FRAME_ID"]
        
        self.gt_sampling_num = self.cfg["GT_SAMPLING_NUM"]
        
        self.split = self.cfg["SPLIT"]
        if(self.split in ["train", "val", "trainval"]):
            self.split_dir = "training"
        
        #* 生成各个文件的路径
        self.bin_file = self.data_root / self.split_dir / 'velodyne' / (self.choosen_frame_id+'.bin')
        self.calib_file = self.data_root / self.split_dir / 'calib' / (self.choosen_frame_id+'.txt')
        self.database_file = self.data_root / 'image_database_train.pkl'
        self.label_file = self.data_root / self.split_dir / 'label_2' / (self.choosen_frame_id+'.txt')
        self.image_file = self.data_root / self.split_dir / 'image_2' / (self.choosen_frame_id+'.png')
        self.plane_file = self.data_root / self.split_dir / 'planes' / (self.choosen_frame_id+'.txt')
        
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
        
        self.gt_sampling_num = self.cfg["GT_SAMPLING_NUM"]
        
        self.height_limit = self.cfg["HEIGHT_LIMIT"]

    @staticmethod
    def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.safe_load(f)
        return config
    
    def get_lidar_data(self, filename=None):
        """读取lidar的bin数据

        Args:
            filename (_type_, optional): 如果是None, 就读取self.bin_file. Defaults to None.

        Returns:
            _type_: [N*4]的点云
        """
        bin_file = self.bin_file if filename is None else filename
        bin_file = str(bin_file)
        lidar_points = np.fromfile(bin_file, dtype=np.float32)
        return lidar_points.reshape(-1, 4)    
    
    def get_database_data(self, filename=None):
        """读取数据库数据
        数据库是一个列表, 存了处于不同angle范围的物体, 索引是逆时针递增。

        Returns:
            _type_: _description_
        """
        database_file = filename if filename is not None else self.database_file
        database_file = str(database_file)
        with open(database_file, 'rb') as f:
            database = pickle.load(f)
        return database
    
    def get_calib_data(self, filename=None):
        """获取标定数据

        Args:
            filename (_type_, optional): 标定文件. Defaults to None.

        Returns:
            _type_: 获取到的标定类
        """
        calib_file = self.calib_file if filename is None else filename
        calib_file = str(calib_file)
        calib = calibration_kitti.Calibration(calib_file)
        return calib
        
    def get_image_data(self, filename = None):
        image_file = self.image_file if filename is None else filename
        image_file = str(image_file)
        return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    
    def get_plane_data(self, filename = None):
        plane_file = self.plane_file if filename is None else filename
        plane_file = str(plane_file)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        plane_data = lines[-1].strip().split()
        plane_data = list(map(float, plane_data))
        return plane_data
    
    def get_label_data(self, filename = None):
        label_file = self.label_file if filename is None else filename
        label_file = str(label_file)
        return get_objects_from_label(label_file)

    def filter_lidar_points(self, points=None, calib=None, road_planes=None):
        lidar_points = self.get_lidar_data() if points is None else points
        lidar_calib = self.get_calib_data() if calib is None else points
        lidar_planes = self.get_plane_data() if road_planes is None else road_planes
        a, b, c, d = lidar_planes
        points_rect = lidar_calib.lidar_to_rect(lidar_points[:, :3])
        #* 求得每个LiDAR点所处的地面高度
        points_rect_height = (-d - a * points_rect[:, 0] - c * points_rect[:, 2]) / b
        #* 必须要高于地面高度才会保留
        valid_mask = points_rect_height > points_rect[:, 1]+self.height_limit
        points_np = lidar_points[valid_mask]
        valid_mask1 = []
        for point in points_np:
            #* 计算距离lidar的距离, 并判断距离索引
            dis = np.sqrt(point[0] * point[0] + point[1] * point[1])
            #* 计算距离索引
            range_idx = int(dis/self.range_interval)
            #* 超过所有就不考虑
            if(range_idx < 0 or range_idx >= self.range_num): 
                valid_mask1.append(False)
                continue
            
            #* 计算角度值，并判断角度索引
            degree = np.arctan2(point[0], -point[1]) * 180 / np.pi
            if(degree < 45):
                valid_mask1.append(False)
                continue  
            #* angle_idx逆时针递增
            angle_idx = int((degree-self.start_angle)/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num): 
                valid_mask1.append(False)
                continue
            valid_mask1.append(True)
        valid_mask1 = np.array(valid_mask1, dtype=np.bool_)
        return points_np[valid_mask1]

    def plot_lidar(self, points):
        self.ax.scatter(-points[:, 1], points[:, 0], s=0.1, c='#000000')

    def show(self):
        plt.xlim(-60, 60)
        plt.ylim(0, 82)
        
        #* angle text
        for idx in range(self.angle_num):
            cur_angle = self.start_angle + idx * self.angle_interval
            text_x = np.cos(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            text_y = np.sin(np.deg2rad(180-cur_angle)) * (self.range_limit + 3)
            self.ax.text(text_x, text_y, str(round(180-cur_angle, 2)), fontsize=20, 
                         color='black', ha='center', va='center', rotation=90-cur_angle)
        text_x = np.cos(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        text_y = np.sin(np.deg2rad(90-self.angle_limit)) * (self.range_limit + 3)
        self.ax.text(text_x, text_y, str(round(90-self.angle_limit, 2)), fontsize=20, 
                     color='black', ha='center', va='center', rotation=-self.angle_limit)
        
        #* range_text
        x_points, y_points = [], []
        range_split = 5
        range_interval = self.range_limit/range_split
        for idx in range(range_split):
            cur_range = idx*range_interval
            text_x = np.cos(np.deg2rad(90+self.angle_limit)) * (cur_range)-2
            text_y = np.sin(np.deg2rad(90+self.angle_limit)) * (cur_range)-1
            x_points.append(text_x+2)
            y_points.append(text_y+1)
            self.ax.text(text_x, text_y, str(round(cur_range, 2)), fontsize=20, 
                         color='black', ha='center', va='center')
        
        text_x = np.cos(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-2
        text_y = np.sin(np.deg2rad(90+self.angle_limit)) * (self.range_limit)-1
        x_points.append(text_x+2)
        y_points.append(text_y+1)
        self.ax.text(text_x, text_y, str(round(self.range_limit, 2)), fontsize=20, 
                     color='black', ha='center', va='center')
        
        self.ax.scatter(x_points, y_points)
        # 显示图形
        plt.axis('off')
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1], loc = 'lower right', fontsize=28)
        plt.savefig('augmented_lidar.png', bbox_inches='tight', dpi = self.fig.dpi, pad_inches=0.0)
        plt.show()
    
    def plot_valid_area(self):
        self.start_angle = 90-self.angle_limit
        points = self.filter_lidar_points()

        #* 起始的角度

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
            #* angle_idx逆时针递增
            angle_idx = int((degree-self.start_angle)/self.angle_interval)
            if(angle_idx < 0 or angle_idx >= self.angle_num): 
                continue
            occupied[angle_idx][range_idx] = 1
        vis = {
            'direct occupation': False, 
            'indirect occupation': False, 
            'free space': False
        }
        
        for angle_idx in range(self.angle_num):
            #* 是否出现已经被占用的红色区域
            vis_flag =False
            angle = angle_idx*self.angle_interval + self.start_angle
            for range_idx in range(self.range_num):
                rang = (range_idx+1)*self.range_interval
                if(occupied[angle_idx][range_idx]):
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                        width=self.range_interval, fill=True, color=self.direct_occupied_color, alpha=0.5,
                                        label = 'direct occupation' if not vis['direct occupation'] else None)
                    #* 这个角度已经出现了被占用的区域
                    vis_flag = True
                    vis['direct occupation'] = True
                elif(vis_flag):
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                        width=self.range_interval, fill=True, color=self.indirect_occupied_color, alpha=0.5, 
                                        label = 'indirect occupation' if not vis['indirect occupation'] else None)  
                    occupied[angle_idx][range_idx] = 2
                    vis['indirect occupation'] = True
                else:
                    wedge = patches.Wedge(self.center, rang, angle, angle+self.angle_interval,
                                width=self.range_interval, fill=True, color=self.free_space_color, alpha=0.5, 
                                label = 'free space' if not vis['free space'] else None)
                    vis['free space'] = True
                self.ax.add_patch(wedge)
        # self.plot_lidar(points)
        
        self.augment_display(points, occupied)
    
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
    
    def augment_display(self, points, occupied):
        #* 计算为0的索引
        unoccupied_indexs = (occupied==0).nonzero()
        #* -> N * 2
        unoccupied_indexs = np.concatenate([np.expand_dims(unoccupied_indexs[0], axis=-1), np.expand_dims(unoccupied_indexs[1], axis=-1)], axis=-1)
        
        # decay_factor = 0.05
        probabilities = np.array([1 for i in range(unoccupied_indexs.shape[0])])
        probabilities = probabilities/probabilities.sum()
        
        #* 采样可以放置的位置
        indexs = np.random.choice(range(unoccupied_indexs.shape[0]), size=self.gt_sampling_num, replace=False, p=probabilities)
        choosen_indexs = unoccupied_indexs[indexs]
        
        origin_labels = self.get_label_data()
        origin_boxes_camera_3d = np.array([[*(obj.loc), obj.l, obj.h, obj.w, obj.ry] for obj in origin_labels])
        origin_boxes_2d = np.array([[*obj.box2d] for obj in origin_labels])
            
        #* 获取当前帧的标注标定信息
        origin_calib = self.get_calib_data()
        #* 将3D包围框从相机坐标系转到LiDAR坐标系
        origin_boxes_lidar_3d = boxes3d_kitti_camera_to_lidar(origin_boxes_camera_3d, origin_calib)
        #* 初始的GT有多少个
        origin_label_box_num = origin_boxes_lidar_3d.shape[0]
        
        #* 获取原始图像
        origin_image = self.get_image_data()
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2BGRA)
        #* 直接粘贴包围框的图像
        origin_image_copy_direct = deepcopy(origin_image)
        #* 粘贴kins数据集的分割结果
        origin_image_copy_kins = deepcopy(origin_image)
        
        origin_height, origin_width, _ = origin_image.shape
        
        #* 对选取的索引排序，越远的越先采样
        choosen_indexs = sorted(choosen_indexs, key=lambda x: x[1], reverse=True)
        #* 获取数据库中的车辆信息
        database = self.get_database_data()['Car']
        
        #* 记录增加的物体的2D包围框
        added_boxes_coor2d = []
        point_vis = False
        for angle_idx, range_idx in choosen_indexs:
            choosen_database = database[angle_idx][range_idx]
            
            #* 从这个角度区间随机选一个, 如果这个位置没有样本就跳过
            if(len(choosen_database) == 0):
                continue
            
            #* 从当前位置的数据库里面随机选择一个
            random_idx = random.randint(0, len(choosen_database)-1)
            #* 获取选择的样本的图像路径
            added_image_path = choosen_database[random_idx]['image_path']
            # added_image_path= '006137_Car_0.png'
            
            filename_base, _ = added_image_path.split('.')
            frame_id, _, idx = filename_base.split('_')
            idx = int(idx)
            added_image_path_origin = self.data_root / "image_gt_database_train" / added_image_path
            added_image = self.get_image_data(added_image_path_origin)
            added_image_kins_path = self.data_root / "image_gt_database_trainKINS" / added_image_path
            assert os.path.exists(str(added_image_kins_path))
            added_image_kins = self.get_image_data(added_image_kins_path)
            
            #* 读取添加的这个样本的原始图片
            added_image_path_direct = self.data_root / self.split_dir / "image_2" / (frame_id+'.png')
            added_image_direct = self.get_image_data(added_image_path_direct)
            
            #* 读取这个样本的标签
            added_label_file = self.data_root / self.split_dir / "label_2" / (frame_id+'.txt')
            added_labels = self.get_label_data(added_label_file)
            
            #* 读取这个样本的标定文件
            added_calib_file = self.data_root / self.split_dir / "calib" / (frame_id+'.txt')
            added_calib = self.get_calib_data(added_calib_file)
            
            #* 找到添加的这个物体的标签
            added_object_label = added_labels[idx]
            #* 构建添加的这个物体在相机坐标系下的包围框 -> 1*7
            added_boxes_camera_3d_in_added = np.array([[*(added_object_label.loc), 
                                            added_object_label.l, added_object_label.h, 
                                            added_object_label.w, added_object_label.ry]])
            #* 直接添加的物体的2D包围框
            added_boxes2d_direct = added_object_label.box2d
            direct_tlx, direct_tly, direct_brx, direct_bry = added_boxes2d_direct
            direct_tlx = floor(direct_tlx)
            direct_tly = floor(direct_tly)
            direct_brx = ceil(direct_brx)
            direct_bry = ceil(direct_bry)
            added_image_direct = added_image_direct[direct_tly:direct_bry, direct_tlx:direct_brx]
            
            #* 转到lidar坐标系
            added_boxes_lidar_3d = boxes3d_kitti_camera_to_lidar(added_boxes_camera_3d_in_added, added_calib)
            added_plane_file = self.data_root / self.split_dir / "planes" / (frame_id+'.txt')
            a, b, c, d = self.get_plane_data(added_plane_file)
            center_cam = added_calib.lidar_to_rect(added_boxes_lidar_3d[:, 0:3])
            cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
            center_cam[:, 1] = cur_height_cam
            cur_lidar_height = added_calib.rect_to_lidar(center_cam)[:, 2]
            mv_height = added_boxes_lidar_3d[:, 2] - added_boxes_lidar_3d[:, 5] / 2 - cur_lidar_height
            added_boxes_lidar_3d[:, 2] -= mv_height  # lidar view
            
            
            boxes_corners = boxes_to_corners_3d(added_boxes_lidar_3d)

            
            #* 将点云增加到特定位置
            added_bin_file = self.data_root / "gt_database" / (filename_base + '.bin')
            added_points = self.get_lidar_data(added_bin_file)
            added_points[:, 0] += added_boxes_lidar_3d[0][0]
            added_points[:, 1] += added_boxes_lidar_3d[0][1]
            added_points[:, 2] += added_boxes_lidar_3d[0][2]

            
            #* 计算bev iou 和已经添加的框计算iof(交集/非添加物体的面积)
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(added_boxes_lidar_3d[:, 0:7], origin_boxes_lidar_3d[:, 0:7])
            iof = 0
            #* 增加的物体
            added_boxes_camera_3d_in_origin = boxes3d_lidar_to_kitti_camera(added_boxes_lidar_3d, origin_calib)
            added_box2d_in_origin = boxes3d_kitti_camera_to_imageboxes(added_boxes_camera_3d_in_origin, origin_calib)
            #* 坐标需要在图像范围内
            if(0<=added_box2d_in_origin[0][0]<origin_width and 0<=added_box2d_in_origin[0][2]<origin_width and 
               0<=added_box2d_in_origin[0][1]<origin_height and 0<=added_box2d_in_origin[0][3]<origin_height):
                pass
            else:
                continue
            for obj in origin_boxes_2d:
                #* 计算2D包围框的iof
                inter = self.compute_intersection(added_box2d_in_origin[0], obj)
                obj_area = self.compute_area(obj)
                add_obj_area = self.compute_area(added_box2d_in_origin[0])
                #* 更新iof
                if(inter/obj_area > iof):
                    iof = inter/obj_area
                if(inter/add_obj_area > iof):
                    iof = inter/add_obj_area
                  
            if(iou.max() == 0 and iof < self.iof_thresh):
                with open('box_corners.bin', 'w') as f:
                    boxes_corners.tofile(f)
                with open('added_lidar.bin', 'w') as f:
                    added_points[:, :3].tofile(f)
                print(added_image_path)
                # self.ax.scatter(-added_boxes_lidar_3d[0][1], added_boxes_lidar_3d[0][0], c='r', s=5,
                #             label = 'choosen position' if not point_vis else None)
                #* 只需要加一次标签就可以
                point_vis = True
                #* 添加包围框
                origin_boxes_lidar_3d = np.concatenate([origin_boxes_lidar_3d, added_boxes_lidar_3d], axis=0)
                origin_boxes_2d = np.concatenate([origin_boxes_2d, added_box2d_in_origin], axis=0)
                #* 添加对应的点云
                points = np.concatenate([added_points[:, :3], points[:, :3]], axis=0)
                #* 添加对应的图像
                left, top, right, bottom = added_box2d_in_origin[0]
                left = floor(left)
                top = floor(top)
                right = ceil(right)
                bottom = ceil(bottom)
                added_boxes_coor2d.append([left, top, right, bottom])
                #! 画图备用
                cv2.imwrite("before_resize.png", added_image_kins)
                #* 将添加的图片resize成这个尺寸
                added_image = cv2.resize(added_image, (right-left, bottom-top), interpolation=cv2.INTER_LINEAR)

                #* resize直接粘贴的图片
                added_image_direct = cv2.resize(added_image_direct, (right-left, bottom-top), interpolation=cv2.INTER_LINEAR)
                #* resize KINS的mask
                added_image_kins = cv2.resize(added_image_kins, (right-left, bottom-top), interpolation=cv2.INTER_LINEAR)
                #! 画图备用
                cv2.imwrite("after_resize.png", added_image_kins)
                
                origin_image = self.paste_image_with_mask(origin_image.copy(), added_image, left, top, right, bottom)
                origin_image_copy_kins = self.paste_image_with_mask(origin_image.copy(), added_image_kins, left, top, right, bottom)
                
                #* 直接粘贴的图片
                origin_image_copy_direct[top:bottom, left:right, :3] = added_image_direct
        
        for new_top, new_left, new_bottom, new_right in added_boxes_coor2d:
            origin_image = self.draw_rectangle(origin_image, new_right, new_left, new_bottom, new_top, color = (0, 0, 255))
            origin_image_copy_direct = self.draw_rectangle(origin_image_copy_direct, new_right, new_left, new_bottom, new_top, color = (0, 0, 255))
            origin_image_copy_kins = self.draw_rectangle(origin_image_copy_kins, new_right, new_left, new_bottom, new_top, color = (0, 0, 255))
            
            # cv2.rectangle(origin_image, (new_top, new_left), (new_bottom, new_right), color = (0, 0, 255), thickness=2)
            # cv2.rectangle(origin_image_copy_direct, (new_top, new_left), (new_bottom, new_right), color = (0, 0, 255), thickness=2)
            # cv2.rectangle(origin_image_copy_kins, (new_top, new_left), (new_bottom, new_right), color = (0, 0, 255), thickness=2)
        n = len(origin_labels)
        for idx, (new_top, new_left, new_bottom, new_right) in enumerate(origin_boxes_2d[:n]):
            # TODO 现在是只画车的, 还需要改改
            if(origin_labels[idx].cls_type == 'Car'):
                origin_image = self.draw_rectangle(origin_image, ceil(new_right), floor(new_left), ceil(new_bottom), floor(new_top), color = (0, 255, 0))
                origin_image_copy_direct = self.draw_rectangle(origin_image_copy_direct, ceil(new_right), floor(new_left), ceil(new_bottom), floor(new_top), color = (0, 255, 0))
                origin_image_copy_kins = self.draw_rectangle(origin_image_copy_kins, ceil(new_right), floor(new_left), ceil(new_bottom), floor(new_top), color = (0, 255, 0))
                
                # cv2.rectangle(origin_image, (floor(new_top), floor(new_left)), (ceil(new_bottom), ceil(new_right)), color = (0, 255, 0), thickness=2)
                # cv2.rectangle(origin_image_copy_direct, (floor(new_top), floor(new_left)), (ceil(new_bottom), ceil(new_right)), color = (0, 255, 0), thickness=2)
                # cv2.rectangle(origin_image_copy_kins, (floor(new_top), floor(new_left)), (ceil(new_bottom), ceil(new_right)), color = (0, 255, 0), thickness=2)
                
        cv2.imwrite('augmented_image.png', origin_image[:, :, :3])
        cv2.imwrite('augmented_image_direct.png', origin_image_copy_direct[:, :, :3])
        cv2.imwrite('augmented_image_kins.png', origin_image_copy_kins[:, :, :3])
        
        self.plot_lidar(points)
        self.add_boxes(origin_boxes_lidar_3d, origin_label_box_num)

    def draw_rectangle(self, origin_image, new_right, new_left, new_bottom, new_top, color = (0, 0, 255), alpha = 0.7):
        create = np.zeros((new_right-new_left, new_bottom-new_top, 3), dtype=np.uint8)
        create[:, :, 0] = color[0]
        create[:, :, 1] = color[1]
        create[:, :, 2] = color[2]
        img_add = cv2.addWeighted(origin_image[new_left:new_right, new_top:new_bottom, 0:3], alpha ,create, 1-alpha, 0)
        origin_image[new_left:new_right, new_top:new_bottom, 0:3] = img_add
        cv2.rectangle(origin_image, (new_top, new_left), (new_bottom, new_right), color = color, thickness=2)
        return origin_image
        
    
    def paste_image_with_mask(self, origin_image, added_image, left, top, right, bottom):
        cropped_image = origin_image[top:bottom, left:right, :]
        alpha_mask = added_image[:, :, 3]>0
        cropped_image[alpha_mask] = added_image[alpha_mask]
        origin_image[top:bottom, left:right, :] = cropped_image
        
        copy_image = deepcopy(origin_image)
        #! 画图备用
        cv2.imwrite("before_blur.png", copy_image)
        copy_image = cv2.GaussianBlur(copy_image, (3, 3), 0)
        #! 画图备用
        cv2.imwrite("after_blur.png", copy_image)
        cropped_image = origin_image[top:bottom, left:right, :]
        added_image = copy_image[top:bottom, left:right, :]
        #! 画图备用
        added_image[:, :, 3] = 0
        added_image[alpha_mask, 3] = 255
        cv2.imwrite("added_mask.png", added_image)
        cropped_image[alpha_mask] = added_image[alpha_mask]
        origin_image[top:bottom, left:right, :] = cropped_image
        return origin_image
        
    def add_boxes(self, boxes, label_box_num):
        vis_label = False
        vis_added = False
        for idx, (x, y, _, length, width, _, heading) in enumerate(boxes):
            #* 这是原来的包围框
            if(idx < label_box_num):
                rectangle = patches.Rectangle((-y-width/2,x-length/2), width, length, angle = np.rad2deg(heading), color = '#8F3C2E', alpha=0.7,
                                              rotation_point = 'center',
                                              label = 'original GT' if not vis_label else None)
                vis_label = True
            #* 这是添加的包围框
            else:
                rectangle = patches.Rectangle((-y-width/2,x-length/2), width, length, angle = np.rad2deg(heading), color = '#137BDB', alpha=0.7,
                                              rotation_point = 'center',
                                              label = 'added GT' if not vis_added else None)
                vis_added = True
            self.ax.add_patch(rectangle)

if __name__ == '__main__':
    svs = SearchValidSpace("/home/zty/Project/DeepLearning/OpenPCDet/tools/cfgs/dataset_configs/database_generate_kitti.yaml")
    svs.plot_valid_area()
    svs.show()
    # pngs = ["001322_Car_3.png", "000202_Car_0.png", "005797_Car_4.png", "006431_Car_3.png"]
    # for png in pngs:
    #     basename = png.split('.')[0]
    #     frame_id, typ, idx = basename.split('_')
    #     added_label_file = svs.data_root / svs.split_dir / "label_2" / (frame_id+'.txt')
    #     added_labels = svs.get_label_data(added_label_file)
    #     added_image_file = svs.data_root / svs.split_dir / "image_2" / (frame_id+'.png')
    #     added_image = svs.get_image_data(added_image_file)
    #     label = added_labels[int(idx)]
    #     direct_tlx, direct_tly, direct_brx, direct_bry = label.box2d
    #     direct_tlx = floor(direct_tlx)
    #     direct_tly = floor(direct_tly)
    #     direct_brx = ceil(direct_brx)
    #     direct_bry = ceil(direct_bry)
    #     added_image_direct = added_image[direct_tly:direct_bry, direct_tlx:direct_brx]
    #     cv2.imwrite(png, added_image_direct)
        
    