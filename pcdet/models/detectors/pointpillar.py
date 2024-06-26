from .detector3d_template import Detector3DTemplate
import torch
import pickle

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if(self.model_cfg.get('SAVE_PKL', False)):
                for frame_id, pred, score in zip(batch_dict['frame_id'], batch_dict['batch_box_preds'], batch_dict['batch_cls_preds']):
                    if(batch_dict['cls_preds_normalized']):
                        result_3d = torch.cat([pred, score], dim=-1)
                    else:
                        result_3d = torch.cat([pred, torch.sigmoid(score)], dim=-1)
                    pkl_file = '../data/kitti/lidar_detector_data/'+ self.model_cfg.CONFIG_NAME + '/' + str(frame_id) + '.pkl'
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(result_3d.detach().cpu().numpy(), f)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
