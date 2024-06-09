from bs4 import BeautifulSoup
import requests
from mmengine.config import Config
import os
import os.path as osp
START = "retinanet"

def find_and_modify_key(dictionary, target_key, new_value):
    """修改字典中的某个key为一个值

    Args:
        dictionary (_type_): 字典
        target_key (_type_): 目标key
        new_value (_type_): 目标值
    """
    for key, value in dictionary.items():
        if key == target_key:
            dictionary[key] = new_value
        elif isinstance(value, dict):
            find_and_modify_key(value, target_key, new_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    find_and_modify_key(item, target_key, new_value)

def modify_nuscenes(dictionary):
    dictionary.data_root = 'data/nuImages'
    dictionary.train_dataloader.dataset.data_root = dictionary.data_root
    dictionary.train_dataloader.dataset.ann_file = 'annotations/nuimages_v1.0-train.json'
    dictionary.train_dataloader.dataset.data_prefix.img = './'
    
    dictionary.val_dataloader.dataset.data_root = dictionary.data_root
    dictionary.val_dataloader.dataset.ann_file = 'annotations/nuimages_v1.0-val.json'
    dictionary.val_dataloader.dataset.data_prefix.img = './'
    
    dictionary.test_dataloader.dataset.data_root = dictionary.data_root
    dictionary.test_dataloader.dataset.ann_file = 'annotations/nuimages_v1.0-val.json'
    dictionary.test_dataloader.dataset.data_prefix.img = './'
    
    dictionary.val_evaluator.ann_file = 'data/nuImages/annotations/nuimages_v1.0-val.json'
    dictionary.test_evaluator.ann_file = 'data/nuImages/annotations/nuimages_v1.0-val.json'
    
def get_batch_size(dictionary):
    return dictionary['train_dataloader']['batch_size']

def get_lr(dictionary):
    return dictionary["optim_wrapper"]["optimizer"]["lr"]

def modify_classes(dictionary, classes_value):
    for key, value in dictionary.items():
        if key == "dataset":
            if("metainfo") in dictionary[key]:
                dictionary[key]["metainfo"]["classes"] = classes_value
            else:
                dictionary[key]["metainfo"] = {}
                dictionary[key]["metainfo"]["classes"] = classes_value
        elif isinstance(value, dict):
            modify_classes(value, classes_value)

header = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
while(True):
    try:
        response = requests.get("https://github.com/open-mmlab/mmdetection/tree/v3.0.0", headers = header)
        print("访问mmdetection成功")
        break
    except:
        print("访问mmdetection失败, 重试")
        continue
html = response.text
soup = BeautifulSoup(html, features="html.parser")

all_tables = soup.findAll("table")
model_table = all_tables[1]
detection_col = model_table.findAll("tr")[1]
detection_col_td = detection_col.findAll("td")[0]
detection_col_ul = detection_col_td.find("ul")
start_flag = False
for model_url in detection_col_ul.findAll("li"):
    href = model_url.find("a")
    link_url = href['href']
    if START not in link_url:
    # if not start_flag and START not in link_url:
        continue
    else:
        start_flag = True
    link_url = "https://github.com"+link_url
    while(True):
        try:
            response = requests.get(link_url, headers = header)
            print("访问{}成功".format(link_url))
            break
        except:
            print("访问{}失败，重试".format(link_url))
            continue
    html = response.text
    soup = BeautifulSoup(html, features="html.parser")
    all_tables = soup.findAll("table")
    for table in all_tables:
        for trs in table.findAll("tr"):
            config_file = None
            weight_file = None            
            for href in trs.findAll("a"):
                if(href.text == "config"):
                    config_file = href["href"]
                    sp = config_file.split('/')
                    config_file = '/'.join(sp[-3:])[:-2]
                elif(href.text == "model"):
                    weight_file = href["href"][2:-2]
            if(config_file is None or weight_file is None):
                continue
            config_file = config_file.split('/')
            config_file[1] = 'retinanet_nuscenes'
            config_file = '/'.join(config_file)
            if not os.path.exists("mmdetection/"+config_file):
                continue
            cfg = Config.fromfile("mmdetection/"+config_file)
            print(cfg.pretty_text)
            #! 修改类别
            # find_and_modify_key(cfg, "num_classes", 3)
            # modify_classes(cfg, ('Car', 'Pedestrian', 'Cyclist'))
            modify_nuscenes(cfg)
            find_and_modify_key(cfg, "num_classes", 10)
            modify_classes(cfg, ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier'))
            find_and_modify_key(cfg, "load_from", weight_file)
            batch_size = get_batch_size(cfg)
            batch_size*=8   #* 模拟8卡
            find_and_modify_key(cfg, "base_batch_size", batch_size)
            find_and_modify_key(cfg, "batch_size", batch_size)
            lr = get_lr(cfg)
            dump_filename = 'tools/image_data/'+config_file
            sp = dump_filename.split('/')
            dump_dir = '/'.join(sp[:-1])
            os.makedirs(dump_dir, exist_ok=True)
            cfg.dump('tools/image_data/'+config_file)
            return_code = 1
            while(return_code !=0 and batch_size!=0):
                return_code = os.system('cd mmdetection; python tools/train.py ../tools/image_data/'+config_file + \
                    " --work-dir " + "work_dirs_nuImages/" + osp.splitext(osp.basename(config_file))[0])
                if return_code==0:
                    break
                print("以batch_size{}训练{}失败".format(batch_size, config_file))
                batch_size//=2    
                lr/=2  
                find_and_modify_key(cfg, "base_batch_size", batch_size)
                find_and_modify_key(cfg, "batch_size", batch_size)
                find_and_modify_key(cfg, "lr", lr)
                dump_filename = 'tools/image_data/'+config_file
                sp = dump_filename.split('/')
                dump_dir = '/'.join(sp[:-1])
                os.makedirs(dump_dir, exist_ok=True)
                cfg.dump('tools/image_data/'+config_file)
            if(return_code == 0):
                print("以batch_size{}训练{}成功".format(batch_size, config_file))
            else:
                print("训练{}失败".format(config_file))

        