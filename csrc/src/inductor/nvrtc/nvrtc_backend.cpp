#include "lamp3/inductor/nvrtc/nvrtc_backend.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace lmp::inductor {

namespace {

struct LoadedKernel {
  CUmodule module;
  CUfunction func;
};

struct KernelCacheKey {
  std::string source;
  CUcontext context;
  int compute_major;
  int compute_minor;

  bool operator==(const KernelCacheKey& other) const {
    return source == other.source && context == other.context &&
           compute_major == other.compute_major &&
           compute_minor == other.compute_minor;
  }
};

void hash_combine(size_t& seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct KernelCacheKeyHash {
  size_t operator()(const KernelCacheKey& key) const {
    size_t hash = std::hash<std::string>{}(key.source);
    hash_combine(
        hash,
        std::hash<std::uintptr_t>{}(
            reinterpret_cast<std::uintptr_t>(key.context)));
    hash_combine(hash, std::hash<int>{}(key.compute_major));
    hash_combine(hash, std::hash<int>{}(key.compute_minor));
    return hash;
  }
};

void initialize_cuda_driver() {
  static std::once_flag init_flag;
  std::call_once(init_flag, [] { CU_CHECK(cuInit(0)); });
}

KernelCacheKey make_cache_key(const std::string& source) {
  initialize_cuda_driver();

  CUcontext context = nullptr;
  CU_CHECK(cuCtxGetCurrent(&context));
  LMP_CHECK(context != nullptr)
      << "NVRTC kernel compilation requires a current CUDA context";

  CUdevice device;
  CU_CHECK(cuCtxGetDevice(&device));

  int compute_major = 0;
  int compute_minor = 0;
  CU_CHECK(cuDeviceGetAttribute(
      &compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_CHECK(cuDeviceGetAttribute(
      &compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  return {source, context, compute_major, compute_minor};
}

LoadedKernel compile_and_load(const KernelCacheKey& key) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(
      &prog, key.source.c_str(), "fused.cu", 0, nullptr, nullptr));

  const std::string arch = "--gpu-architecture=sm_" +
                           std::to_string(key.compute_major) +
                           std::to_string(key.compute_minor);
  const char* opts[] = {arch.c_str(), "--std=c++17"};

  const nvrtcResult cres = nvrtcCompileProgram(prog, 2, opts);
  if (cres != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    LMP_CHECK(false) << "NVRTC compile failed:\n"
                     << log << "\n--- generated source ---\n"
                     << key.source;
  }

  size_t cubin_size = 0;
  NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
  std::vector<char> cubin(cubin_size);
  NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  LoadedKernel k;
  CU_CHECK(cuModuleLoadData(&k.module, cubin.data()));
  CU_CHECK(cuModuleGetFunction(&k.func, k.module, kFusedKernelName));
  return k;
}

class KernelCache {
 public:
  LoadedKernel get_or_compile(const std::string& source) {
    KernelCacheKey key = make_cache_key(source);
    std::lock_guard<std::mutex> lock(mutex_);

    const auto cached = kernels_.find(key);
    if (cached != kernels_.end())
      return cached->second;

    const LoadedKernel kernel = compile_and_load(key);
    const auto [inserted, did_insert] =
        kernels_.emplace(std::move(key), kernel);
    LMP_CHECK(did_insert) << "failed to insert compiled kernel into cache";
    return inserted->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<KernelCacheKey, LoadedKernel, KernelCacheKeyHash> kernels_;
};

KernelCache& kernel_cache() {
  static KernelCache* cache = new KernelCache;
  return *cache;
}

void launch(CUfunction f, std::vector<void*>& args, size_t n) {
  const unsigned int block = 256;
  const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
  // Launches are asynchronous, matching eager kernel launches; callers that
  // need completion (timing, host reads) must synchronize explicitly.
  CU_CHECK(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, args.data(),
                          nullptr));
}

}

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  if (!impl->is_deferred())
    return;

  if (!impl->lazy_op()->is_fusible()) {
    for (const std::shared_ptr<tensor::TensorImpl>& in :
         impl->lazy_op()->inputs)
      tensor::lazy::realize(in.get());
    impl->lazy_op()->run_eager(*impl);
    return;
  }

  FusedGraph g = build_fused_graph(impl);
  const std::string src = codegen_kernel(g);

  const size_t n = impl->numel();
  const size_t bytes = LMP_DISPATCH_ALL_TYPES(
      impl->type(), [&] { return n * sizeof(scalar_t); });

  tensor::Storage out(bytes, tensor::DeviceType::CUDA);

  if (n == 0) {
    impl->set_realized(out);
    return;
  }

  const LoadedKernel k = kernel_cache().get_or_compile(src);

  void* out_ptr = out.data();
  std::vector<void*> in_ptrs;
  in_ptrs.reserve(g.inputs.size());
  for (tensor::TensorImpl* leaf : g.inputs)
    in_ptrs.push_back(leaf->storage().data());
  size_t n_arg = n;

  std::vector<void*> args;
  args.reserve(g.inputs.size() + 2);
  args.push_back(&out_ptr);
  for (void*& p : in_ptrs)
    args.push_back(&p);
  args.push_back(&n_arg);

  launch(k.func, args, n);

  impl->set_realized(out);
}

LMP_REGISTER_LAZY_BACKEND(tensor::DeviceType::CUDA, NVRTCInductorBackend)

}
