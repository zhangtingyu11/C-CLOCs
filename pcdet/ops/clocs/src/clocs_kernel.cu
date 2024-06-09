/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void clocs_compute_iou_sparse_kernel(const int num_3d, 
                                            const float * boxes_3d, 
                                            const int num_2d, 
                                            const float * boxes_2d, 
                                            const float * scores_3d, 
                                            const float * scores_2d, 
                                            const float * dis_to_lidar_3d,
                                            float * overlap){
    const int boxes3d_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const int boxes2d_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    if (boxes3d_idx >= num_3d || boxes2d_idx >= num_2d){
        return;
    }

    const float * box_3d = boxes_3d + boxes3d_idx * 4;
    const float * box_2d = boxes_2d + boxes2d_idx * 4;
    float * cur_overlap = overlap + (boxes3d_idx * num_2d + boxes2d_idx) * 4;
    // float * cur_overlap = overlap + (boxes3d_idx * num_2d + boxes2d_idx) * 3;
    
    float qbox_area = (*(box_2d+2) - *(box_2d+0)) *
                        (*(box_2d+3) - *(box_2d+1));
    float iw = min(*(box_3d+2), *(box_2d+2)) - 
                max(*(box_3d+0), *(box_2d+0));

    if(iw > 0){
        float ih = min(*(box_3d+3), *(box_2d+3)) - 
                    max(*(box_3d+1), *(box_2d+1));
        if(ih > 0){
            float ua = ((*(box_3d+2) - *(box_3d+0)) * 
                        (*(box_3d+3) - *(box_3d+1)) + qbox_area - iw * ih);
            cur_overlap[0] = (iw * ih /ua);
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = scores_2d[boxes2d_idx];
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
        }
        else{
            //* 填写-10主要是为了和iou接近于0的区别开
            cur_overlap[0] = -10;
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = scores_2d[boxes2d_idx];
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
        }
    }
    else{
        cur_overlap[0] = -10;
        cur_overlap[1] = scores_3d[boxes3d_idx];
        cur_overlap[2] = scores_2d[boxes2d_idx];
        cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
    }
}

__global__ void clocs_compute_iou_dense_kernel(const int num_3d, 
                                                const float * boxes_3d, 
                                                const int num_2d, 
                                                const float * boxes_2d, 
                                                const int * max_num,
                                                const float * scores_3d, 
                                                const float * scores_2d, 
                                                const float * dis_to_lidar_3d,
                                                float * overlap,
                                                int * tensor_idx,
                                                int * count){
    const int boxes3d_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const int boxes2d_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    if (boxes3d_idx >= num_3d || boxes2d_idx >= num_2d){
        return;
    }

    const float * box_3d = boxes_3d + boxes3d_idx * 4;
    const float * box_2d = boxes_2d + boxes2d_idx * 4;
    
    float qbox_area = (*(box_2d+2) - *(box_2d+0)) *
                        (*(box_2d+3) - *(box_2d+1));
    float iw = min(*(box_3d+2), *(box_2d+2)) - 
                max(*(box_3d+0), *(box_2d+0));

    if(iw > 0){
        float ih = min(*(box_3d+3), *(box_2d+3)) - 
                    max(*(box_3d+1), *(box_2d+1));
        if(ih > 0){
            float ua = ((*(box_3d+2) - *(box_3d+0)) * 
                        (*(box_3d+3) - *(box_3d+1)) + qbox_area - iw * ih);
            int cur_idx = atomicAdd(count, 1);
            if(cur_idx>=(*max_num)) return;
            float * cur_overlap = overlap + cur_idx*4;
            int * cur_tensor_idx = tensor_idx +cur_idx*2;
            cur_overlap[0] = iw * ih /ua;
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = scores_2d[boxes2d_idx];
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
            cur_tensor_idx[0] = boxes2d_idx;
            cur_tensor_idx[1] = boxes3d_idx;
        }
        else if(boxes2d_idx == num_2d-1){
            int cur_idx = atomicAdd(count, 1);
            if(cur_idx>=(*max_num)) return;
            float * cur_overlap = overlap + cur_idx*4;
            int * cur_tensor_idx = tensor_idx +cur_idx*2;
            cur_overlap[0] = -10;
            cur_overlap[1] = scores_3d[boxes3d_idx];
            cur_overlap[2] = -10;
            cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
            cur_tensor_idx[0] = boxes2d_idx;
            cur_tensor_idx[1] = boxes3d_idx;
        }
    }
    else if(boxes2d_idx == num_2d-1){
        int cur_idx = atomicAdd(count, 1);
        if(cur_idx>=(*max_num)) return;
        float * cur_overlap = overlap + cur_idx*4;
        int * cur_tensor_idx = tensor_idx +cur_idx*2;
        cur_overlap[0] = -10;
        cur_overlap[1] = scores_3d[boxes3d_idx];
        cur_overlap[2] = -10;
        cur_overlap[3] = dis_to_lidar_3d[boxes3d_idx];
        cur_tensor_idx[0] = boxes2d_idx;
        cur_tensor_idx[1] = boxes3d_idx;
    }
}

__global__ void cos_similarity_kernel(const float * lidar_features, 
                                const float * camera_features, 
                                const int lidar_num, 
                                const int camera_num,
                                const int feature_dim,
                                float * cos_similarity){
    const long long boxes3d_idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    const long long boxes2d_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    const float * lidar_feature = lidar_features + boxes3d_idx * feature_dim;
    const float * camera_feature = camera_features + boxes2d_idx * feature_dim;
    float * target_pos = cos_similarity + boxes3d_idx*camera_num + boxes2d_idx;
    if(boxes3d_idx >= lidar_num || boxes2d_idx >= camera_num){
        return;
    }
    float inner_product = 0;
    float lidar_distance = 0;
    float camera_distance = 0;
    for(int i=0; i < feature_dim;i++){
        inner_product += lidar_feature[i] * camera_feature[i];
        lidar_distance += lidar_feature[i] * lidar_feature[i];
        camera_distance += camera_feature[i] * camera_feature[i];
    }
    float eps = 1e-8;
    target_pos[0] = inner_product / (sqrt(lidar_distance) * sqrt(camera_distance) + eps);
    // if(boxes3d_idx == 1 && boxes2d_idx == 0){
    //     printf("inner_product: %f\n", inner_product);
    //     printf("lidar_distance: %f\n", lidar_distance);
    //     printf("camera_distance: %f\n", camera_distance);
    //     printf("target: %f\n", target_pos[0]);
    //     printf("pos: %d\n", boxes3d_idx*camera_num + boxes2d_idx);
    // }
}

void clocscomputeiousparseLauncher(const int num_3d, 
                                const int num_2d, 
                                const float * boxes_3d, 
                                const float * boxes_2d, 
                                const float * scores_3d, 
                                const float * scores_2d, 
                                const float * dis_to_lidar_3d,
                                float * overlap){
    dim3 blocks(DIVUP(num_3d, THREADS_PER_BLOCK), DIVUP(num_2d, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    clocs_compute_iou_sparse_kernel<<<blocks, threads>>>(num_3d, 
                                                        boxes_3d, 
                                                        num_2d, 
                                                        boxes_2d, 
                                                        scores_3d, 
                                                        scores_2d, 
                                                        dis_to_lidar_3d, 
                                                        overlap);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void clocscomputeioudenseLauncher(const int num_3d, 
                                  const int num_2d, 
                                  const int * max_num,
                                  const float * boxes_3d, 
                                  const float* boxes_2d, 
                                  const float * scores_3d, 
                                  const float * scores_2d, 
                                  const float* dis_to_lidar_3d,
                                  float * overlap,
                                  int * tensor_idx, 
                                  int * count){
    dim3 blocks(DIVUP(num_3d, THREADS_PER_BLOCK), DIVUP(num_2d, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    clocs_compute_iou_dense_kernel<<<blocks, threads>>>(num_3d, 
                                                        boxes_3d, 
                                                        num_2d, 
                                                        boxes_2d, 
                                                        max_num,
                                                        scores_3d, 
                                                        scores_2d, 
                                                        dis_to_lidar_3d, 
                                                        overlap,
                                                        tensor_idx,
                                                        count);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

void cossimilarityLauncher(const float * lidar_features, 
                                const float * camera_features, 
                                const int lidar_num, 
                                const int camera_num,
                                const int feature_dim,
                                float * cos_similarity){
    dim3 blocks(DIVUP(lidar_num, THREADS_PER_BLOCK), DIVUP(camera_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    cos_similarity_kernel<<<blocks, threads>>>(
        lidar_features,
        camera_features,
        lidar_num,
        camera_num,
        feature_dim,
        cos_similarity
    );
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
