#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "clocs.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clocs_comput_iou_sparse", &clocs_compute_iou_gpu_sparse, "compute clocs iou sparsely");
    m.def("clocs_comput_iou_dense", &clocs_compute_iou_gpu_dense, "compute clocs iou densely");
    m.def("cos_similarity_gpu", &cos_similarity_gpu, "cos_similarity");

}