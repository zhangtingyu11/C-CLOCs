#ifndef CLOCS_H
#define CLOCS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int clocs_compute_iou_gpu_sparse(at::Tensor boxes3d_projected, 
                                at::Tensor boxes_2d, 
                                at::Tensor scores_3d, 
                                at::Tensor scores_2d,  
                                at::Tensor dis_to_lidar_3d,
                                at::Tensor overlap);

int clocs_compute_iou_gpu_dense(at::Tensor boxes3d_projected, 
                                at::Tensor boxes_2d, 
                                at::Tensor scores_3d, 
                                at::Tensor scores_2d,
                                at::Tensor dis_to_lidar_3d,
                                at::Tensor max_num,
                                at::Tensor overlap,
                                at::Tensor tensor_idx,
                                at::Tensor count);

int cos_similarity_gpu(at::Tensor lidar_features, 
                                at::Tensor camera_features, 
                                at::Tensor cos_similarity);
#endif