# modify_annotations_txt.py
# 将Car’，’Cyclist’，’Pedestrian’
# 将 ‘Van’, ‘Truck’, ‘Tram’ 合并到 ‘Car’ 类别中去
# 将 ‘Person_sitting’ 合并到 ‘Pedestrian’ 类别中去
# ‘Misc’ 和 ‘Dontcare’ 这两类直接忽略

import glob
import os
import shutil
import json
import cv2
from tqdm import tqdm

label_root = 'data/kitti/training/label_2/'   # 标签文件所在目录
coco_label_dir = 'data/coco/coco_label_2/'
coco_dir = 'data/coco/'

txt_list = glob.glob(label_root + '*.txt') # 存储Labels文件夹所有txt文件路径

# 查看类别集合
def show_category(txt_list):
    category_list= []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ') # 去掉前后多余的字符并把其分开
                    category_list.append(labeldata[0]) # 只要第一个字段，即类别
        except IOError as ioerr:
            print('File error:'+str(ioerr))
    print(set(category_list)) # 输出集合

def kitti2coco(label_dir, img_dir, output_dir, suffix):
    # Create COCO annotation structure
    coco = {}
    coco['images'] = []
    coco['annotations'] = []
    coco['categories'] = []

    # Add categories
    categories = [
        {'id': 0, 'name': 'Car'},
        {'id': 1, 'name': 'Pedestrian'},
        {'id': 2, 'name': 'Cyclist'}
    ]
    coco['categories'] = categories

    # Add images and annotations
    image_id = 0
    annotation_id = 0
    print("kitti转coco:")
    for file in tqdm(os.listdir(label_dir)):
        if file.endswith('.txt'):
            image_path = os.path.join(img_dir, file[:-4] + '.png')
            
            # 读取图片的高宽
            img_file = cv2.imread(image_path)
            img_height, img_width = img_file.shape[0],img_file.shape[1]
            
            image = {
                'id': image_id,
                'file_name': file[:-4] + '.png',
                'height': img_height, # KITTI image height
                'width': img_width # KITTI image width
            }
            coco['images'].append(image)

            with open(os.path.join(label_dir, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    category_id = 0 if line[0] == 'Car' else 1 if line[0] == 'Pedestrian' else 2
                    bbox = [float(coord) for coord in line[4:8]]

                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1


                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [x1, y1, bbox_width,bbox_height],
                        'area': bbox_height*bbox_width,
                        'iscrowd': 0
                    }
                    coco['annotations'].append(annotation)
                    annotation_id += 1

            image_id += 1

    # Write COCO annotation to file
    with open(os.path.join(output_dir, 'instances_'+ suffix + '2017' +'.json'), 'w') as f:
        json.dump(coco, f)

# 将多个字段合并成一行
def merge(line):
    each_line=''
    for i in range(len(line)):
        if i!= (len(line)-1):
            each_line=each_line+line[i]+' '
        else:
            each_line=each_line+line[i] # 最后一条字段后面不加空格
    each_line=each_line+'\n'
    return (each_line)


if __name__ == "__main__":
    print('before modify categories are:\n')
    show_category(txt_list)
    os.makedirs(coco_label_dir, exist_ok=True)
    print("转换标签")
    for item in tqdm(txt_list):
        new_txt=[]
        try:
            with open(item, 'r') as r_tdf:
                for each_line in r_tdf:
                    labeldata = each_line.strip().split(' ')
                    # if labeldata[0] in ['Truck','Van','Tram']: # 合并汽车类
                        # labeldata[0] = labeldata[0].replace(labeldata[0],'Car')
                    # if labeldata[0] == 'Person_sitting': # 合并行人类
                        # labeldata[0] = labeldata[0].replace(labeldata[0],'Pedestrian')
                    if labeldata[0] in ['DontCare', 'Misc', 'Truck','Van','Tram', 'Person_sitting']: # 忽略Dontcare类
                        continue
                    # if labeldata[0] == 'Misc': # 忽略Misc类
                    #     continue
                    new_txt.append(merge(labeldata)) # 重新写入新的txt文件
            sp = item.split('/')
            pre = '/'.join(sp[:-2])
            item = coco_dir + "coco_label_2/"+ sp[-1]
            with open(item,'w+') as w_tdf: # w+是打开原文件将内容删除，另写新内容进去
                for temp in new_txt:
                    w_tdf.write(temp)
        except IOError as ioerr:
            print('File error:'+str(ioerr))

    print('\nafter modify categories are:\n')
    new_txt_list = glob.glob(coco_label_dir + '*.txt')
    show_category(new_txt_list)
    
    # 指定数据集路径和训练/验证集路径
    data_dir = "data/kitti/"
    dest_dir = 'data/coco/'

    train_img_dir = os.path.join(dest_dir, 'train2017')
    train_label_dir = os.path.join(dest_dir, 'labels/train_labels')
    val_img_dir = os.path.join(dest_dir, 'val2017')
    val_label_dir = os.path.join(dest_dir, 'labels/val_labels')

    # 创建训练/验证集文件夹
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 获取该类别的所有图像
    img_path = os.path.join(data_dir + 'training/image_2')

    label_path = os.path.join(dest_dir + 'coco_label_2')

    train_images_file = "data/kitti/ImageSets/train.txt"
    val_images_file = "data/kitti/ImageSets/val.txt"
    with open(train_images_file, 'r') as f:
        train_images = f.readlines()
    with open(val_images_file, 'r') as f:
        val_images = f.readlines()

    print("复制训练集图片")
    for image_name in tqdm(train_images):
        src_path = os.path.join(img_path, image_name.strip()+".png")
        dst_path = os.path.join(train_img_dir, image_name.strip()+".png")
        shutil.copyfile(src_path, dst_path)

        src_label_path = os.path.join(label_path, image_name.strip() + '.txt')
        dst_label_path = os.path.join(train_label_dir, image_name.strip() + '.txt')
        shutil.copyfile(src_label_path, dst_label_path)

    print("复制验证集图片")
    for image_name in tqdm(val_images):
        src_path = os.path.join(img_path, image_name.strip()+".png")
        dst_path = os.path.join(val_img_dir, image_name.strip()+".png")
        shutil.copyfile(src_path, dst_path)

        src_label_path = os.path.join(label_path, image_name.strip() + '.txt')
        dst_label_path = os.path.join(val_label_dir, image_name.strip() + '.txt')
        shutil.copyfile(src_label_path, dst_label_path)
        

    print("数据集划分完成！" + "训练集图片数目: " + str(len(train_images)) + '验证集图片数目: '+ str(len(val_images)))
    
    data_root = 'data/coco/'

    # 输出路径
    outputs_path = os.path.join(data_root, 'annotations')
    os.makedirs(outputs_path, exist_ok=True)
    
    train_img_dir = os.path.join(data_root, 'train2017')    # 训练集图片的路径
    train_label_dir = os.path.join(data_root, 'labels/train_labels')    # 训练集label的路径
    kitti2coco(train_label_dir, train_img_dir, outputs_path, 'train')   # 转换训练集标注格式

    val_img_dir = os.path.join(data_root, 'val2017')    # 验证集图片的路径
    val_label_dir = os.path.join(data_root, 'labels/val_labels')    # 验证集label的路径
    kitti2coco(val_label_dir, val_img_dir, outputs_path, 'val') # 转换验证集标注格式

    print('格式转换完成！')