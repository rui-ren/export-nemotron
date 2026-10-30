#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct TestTensor {
  const char* name;
  void* data;
  size_t bytes;
  const int64_t* shape;
  size_t rank;
  int dtype;
};

// Use the qualified ORT library directly, not the unrelated installed Python wheel.
extern "C" const char* RunTestModel(
    const char* model, const char* provider, const char* library,
    const TestTensor* inputs, size_t input_count,
    const TestTensor* outputs, size_t output_count) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  static std::unique_ptr<Ort::Env> env;
  static std::string provider_path;
  static std::string error;
  try {
    if (!env) {
      auto candidate = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "native-regression");
      candidate->RegisterExecutionProviderLibrary("CUDAExecutionProvider", provider);
      provider_path = provider;
      env = std::move(candidate);
    }
    if (provider_path != provider)
      throw std::runtime_error("Conflicting CUDA provider");
    std::vector<Ort::ConstEpDevice> cuda;
    for (auto device : env->GetEpDevices()) {
      if (std::strcmp(device.EpName(), "CUDAExecutionProvider") == 0) {
        cuda.push_back(device);
        break;
      }
    }
    if (cuda.empty()) throw std::runtime_error("CUDA EP required");
    Ort::SessionOptions options;
    options.AppendExecutionProvider_V2(
        *env, cuda, std::unordered_map<std::string, std::string>{});
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.RegisterCustomOpsLibrary(library);
    Ort::Session session(*env, model, options);
    Ort::MemoryInfo memory("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> in_values, out_values;
    std::vector<const char*> in_names, out_names;
    auto add = [&](const TestTensor* tensors, size_t count,
                   std::vector<Ort::Value>& values, std::vector<const char*>& names) {
      for (size_t i = 0; i < count; ++i) {
        const auto& t = tensors[i];
        values.push_back(Ort::Value::CreateTensor(
            memory, t.data, t.bytes, t.shape, t.rank,
            static_cast<ONNXTensorElementDataType>(t.dtype)));
        names.push_back(t.name);
      }
    };
    add(inputs, input_count, in_values, in_names);
    add(outputs, output_count, out_values, out_names);
    session.Run(Ort::RunOptions{}, in_names.data(), in_values.data(), input_count,
                out_names.data(), out_values.data(), output_count);
    return nullptr;
  } catch (const std::exception& e) {
    error = e.what();
    return error.c_str();
  }
}
