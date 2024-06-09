/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "clocs.h"
#include <stdio.h>

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void clocscomputeiousparseLauncher(const int num_3d, 
                                  const int num_2d, 
                                  const float * boxes3d, 
                                  const float* boxes_2d, 
                                  const float * scores_3d, 
                                  const float * scores_2d, 
                                  const float* dis_to_lidar_3d,
                                  float * overlap);

void clocscomputeioudenseLauncher(const int num_3d, 
                                  const int num_2d, 
                                  const int * max_num,
                                  const float * boxes3d, 
                                  const float* boxes_2d, 
                                  const float * scores_3d, 
                                  const float * scores_2d, 
                                  const float* dis_to_lidar_3d,
                                  float * overlap,
                                  int * tensor_idx, 
                                  int * count);

void cossimilarityLauncher(const float * lidar_features, 
                                const float * camera_features, 
                                const int lidar_num, 
                                const int camera_num,
                                const int feature_dim,
                                float * cos_similarity);

int clocs_compute_iou_gpu_sparse(at::Tensor boxes3d_projected, 
                                at::Tensor boxes_2d, 
                                at::Tensor scores_3d, 
                                at::Tensor scores_2d, 
                                at::Tensor dis_to_lidar_3d,
                                at::Tensor overlap){

    CHECK_INPUT(boxes3d_projected);
    CHECK_INPUT(boxes_2d);
    CHECK_INPUT(scores_3d);
    CHECK_INPUT(scores_2d);
    CHECK_INPUT(dis_to_lidar_3d);
    CHECK_INPUT(overlap);

    const int num_3d = boxes3d_projected.size(0);
    const int num_2d = boxes_2d.size(0);


    const float * boxes3d_data = boxes3d_projected.data<float>();
    const float * boxes2d_data = boxes_2d.data<float>();
    float * scores_3d_data = scores_3d.data<float>();
    float * scores_2d_data = scores_2d.data<float>();
    float * dis_to_lidar_3d_data = dis_to_lidar_3d.data<float>();
    float * overlap_data = overlap.data<float>();

    clocscomputeiousparseLauncher(num_3d, 
                                num_2d, 
                                boxes3d_data, 
                                boxes2d_data, 
                                scores_3d_data, 
                                scores_2d_data, 
                                dis_to_lidar_3d_data,
                                overlap_data);
    return 1;
}


int clocs_compute_iou_gpu_dense(at::Tensor boxes3d_projected, 
                                at::Tensor boxes_2d, 
                                at::Tensor scores_3d, 
                                at::Tensor scores_2d,
                                at::Tensor dis_to_lidar_3d,
                                at::Tensor max_num,
                                at::Tensor overlap,
                                at::Tensor tensor_idx,
                                at::Tensor count){

    
    CHECK_INPUT(boxes3d_projected);
    CHECK_INPUT(boxes_2d);
    CHECK_INPUT(scores_3d);
    CHECK_INPUT(scores_2d);
    CHECK_INPUT(dis_to_lidar_3d);
    CHECK_INPUT(overlap);
    CHECK_INPUT(tensor_idx);
    CHECK_INPUT(count);
    CHECK_INPUT(max_num);

    const int num_3d = boxes3d_projected.size(0);
    const int num_2d = boxes_2d.size(0);

    const float * boxes3d_data = boxes3d_projected.data<float>();
    const float * boxes2d_data = boxes_2d.data<float>();
    float * scores_3d_data = scores_3d.data<float>();
    float * scores_2d_data = scores_2d.data<float>();
    float * dis_to_lidar_3d_data = dis_to_lidar_3d.data<float>();
    float * overlap_data = overlap.data<float>();
    int * tensor_idx_data = tensor_idx.data<int>();
    const int * max_num_data = max_num.data<int>();
    int * count_data = count.data<int>();

    clocscomputeioudenseLauncher(num_3d, 
                                num_2d, 
                                max_num_data,
                                boxes3d_data, 
                                boxes2d_data, 
                                scores_3d_data, 
                                scores_2d_data, 
                                dis_to_lidar_3d_data,
                                overlap_data,
                                tensor_idx_data,
                                count_data);
    return 1;
};

int cos_similarity_gpu(at::Tensor lidar_features, 
                                at::Tensor camera_features, 
                                at::Tensor cos_similarity){
    CHECK_INPUT(lidar_features);
    CHECK_INPUT(camera_features);
    CHECK_INPUT(cos_similarity);
    const int lidar_num = lidar_features.size(0);
    const int camera_num = camera_features.size(1);
    const int feature_dim = camera_features.size(2);
    const float * lidar_features_data = lidar_features.data<float>();
    const float * camera_features_data = camera_features.data<float>();
    float * cos_similarity_data = cos_similarity.data<float>();
    cossimilarityLauncher(lidar_features_data, 
                          camera_features_data, 
                          lidar_num, 
                          camera_num, 
                          feature_dim, 
                          cos_similarity_data);
    return 1;

};