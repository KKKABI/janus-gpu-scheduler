#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

static std::vector<cudaStream_t> global_streams;

// 创建带优先级的 CUDA streams
void create_priority_streams(int num_streams) {
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    global_streams.clear();

    

    for (int i = 0; i < num_streams; ++i) {
        

        cudaStream_t stream;
        cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatestPriority);
        global_streams.push_back(stream);
    }
}


// 获取所有 CUDA stream 指针（以 int64_t 数组返回）
std::vector<int64_t> get_all_stream_ptrs() {
    std::vector<int64_t> stream_ptrs;
    for (auto& s : global_streams) {
        stream_ptrs.push_back(reinterpret_cast<int64_t>(s));
    }
    return stream_ptrs;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_priority_streams", &create_priority_streams, "Create CUDA Streams with Priority");
    m.def("get_all_stream_ptrs", &get_all_stream_ptrs, "Get All CUDA Stream Pointers");
   
}
