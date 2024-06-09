from . import clocs_cuda
import torch
class CustomCosineSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lidar_features, camera_features):
        lidar_num = lidar_features.shape[0]
        camera_num = camera_features.shape[0]
        lidar_features = lidar_features.view(lidar_num, -1, lidar_features.shape[-1])
        camera_features = camera_features.view(-1, camera_num, camera_features.shape[-1])
        
        cos_sim = torch.zeros([lidar_num, camera_num], device = lidar_features.device, requires_grad = True)
        cos_similarity(lidar_features, camera_features, cos_sim)
        ctx.save_for_backward(lidar_features, camera_features, cos_sim)
        return cos_sim

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output -- [B, N]
        lidar_features, camera_features, cos_sim = ctx.saved_tensors
        lidar_features = lidar_features.view(lidar_features.shape[0], -1)
        camera_features = camera_features.view(camera_features.shape[1], -1)
        
        norm_lidar_features = torch.norm(lidar_features, dim=-1)
        norm_camera_features = torch.norm(camera_features, dim=-1)
        
        lidar_num = lidar_features.shape[0]
        camera_num = camera_features.shape[0]
        lidar_first_below = (norm_lidar_features.view(-1, 1) * norm_camera_features.view(1, -1)).unsqueeze(-1)
        camera_ext = camera_features.view(1, camera_num, -1).expand(lidar_num, camera_num, camera_features.shape[-1])
        lidar_second_below = norm_lidar_features.view(lidar_num, 1).expand(lidar_num, camera_num).unsqueeze(-1)
        lidar_ext = lidar_features.view(lidar_num, 1, -1).expand(lidar_num, camera_num, lidar_features.shape[-1])
        lidar_grad = camera_ext/lidar_first_below - cos_sim.view(lidar_num, camera_num, -1) * lidar_ext / lidar_second_below**2
        lidar_grad = torch.sum(lidar_grad, dim=1)
        
        camera_first_below = (norm_camera_features.view(-1, 1) * norm_lidar_features.view(1, -1)).unsqueeze(-1)
        lidar_ext = lidar_features.view(1, lidar_num, -1).expand(camera_num, lidar_num, lidar_features.shape[-1])
        camera_second_below = norm_camera_features.view(camera_num, 1).expand(camera_num, lidar_num).unsqueeze(-1)
        camera_ext = camera_features.view(camera_num, 1, -1).expand(camera_num, lidar_num, camera_features.shape[-1])
        camera_grad = lidar_ext/camera_first_below - cos_sim.T.view(camera_num, lidar_num, -1) * camera_ext / camera_second_below**2
        camera_grad = torch.sum(camera_grad, dim=1)
        return grad_output @ camera_grad, grad_output.T @ lidar_grad

def compute_clocs_iou_sparse(boxes_3d_projected, 
                            boxes_2d_projected, 
                            scores_3d, 
                            scores_2d, 
                            dis_to_lidar_3d, 
                            overlaps):
    clocs_cuda.clocs_comput_iou_sparse(boxes_3d_projected, 
                                    boxes_2d_projected, 
                                    scores_3d, 
                                    scores_2d, 
                                    dis_to_lidar_3d, 
                                    overlaps)
    return overlaps

def compute_clocs_iou_dense(boxes_3d_projected, 
                            boxes_2d_projected, 
                            scores_3d, 
                            scores_2d, 
                            dis_to_lidar_3d, 
                            max_num,
                            overlaps,
                            tensor_idx,
                            count):
    clocs_cuda.clocs_comput_iou_dense(boxes_3d_projected, 
                                    boxes_2d_projected, 
                                    scores_3d, 
                                    scores_2d, 
                                    dis_to_lidar_3d, 
                                    max_num,
                                    overlaps,
                                    tensor_idx, 
                                    count)

def cos_similarity(lidar_features, 
                    camera_features,
                    cos_sim 
):
    clocs_cuda.cos_similarity_gpu(lidar_features,
                                  camera_features,
                                  cos_sim)