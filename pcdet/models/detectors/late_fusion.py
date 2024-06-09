from .detector3d_template import Detector3DTemplate
from ..fusion_heads import clocs_second_contra_head, clocs_second_head, clocs_dense_head, clocs_voxelrcnn_head
import time
import torch
from collections import *
fusion_heads = [clocs_second_head.ClocsSECONDHead, clocs_voxelrcnn_head.ClocsVoxelRCNNHead, clocs_second_contra_head.ClocsSecondContraHead,
                clocs_dense_head.ClocsDenseHead]

def get_millisecond():
    return int(round(time.time() * 1000))

class ClocsNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # self.total_time = defaultdict(float)
        # self.total_cnt = defaultdict(int)

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if all(not isinstance(cur_module, head_module) for head_module in fusion_heads):
                cur_module.eval()
                for param in cur_module.parameters():
                    param.requires_grad = False
            else:
                for param in cur_module.parameters():
                    param.requires_grad = True
            # start_time = get_millisecond()
            batch_dict = cur_module(batch_dict)
            # torch.cuda.synchronize()
            # end_time = get_millisecond()
            # self.total_cnt[cur_module.model_cfg["NAME"]]+=1
            # if(self.total_cnt[cur_module.model_cfg["NAME"]] >= 10):
            #     self.total_time[cur_module.model_cfg["NAME"]]+=end_time-start_time
                
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # start_time = get_millisecond()
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # torch.cuda.synchronize()
            # end_time = get_millisecond()
            # self.total_cnt["postprocessing"]+=1
            # if(self.total_cnt["postprocessing"] >= 10):
            #     self.total_time["postprocessing"]+=end_time-start_time
            
            # for key in self.total_time.keys():
            #     if(self.total_cnt[key]-9)>0:
            #         print("{} inference time: {}".format(key, self.total_time[key]/(self.total_cnt[key]-9)))
            
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_fusion, tb_dict = self.fusion_head.get_loss()
        tb_dict = {
            'loss_fusion': loss_fusion.item(),
            **tb_dict
        }

        loss = loss_fusion
        return loss, tb_dict, disp_dict
