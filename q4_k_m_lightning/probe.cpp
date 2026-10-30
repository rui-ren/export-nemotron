#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: probe model.gguf prompts.tokens scores.f32 generated.txt");
        std::ifstream input(argv[2]);
        std::ofstream scores(argv[3], std::ios::binary), generated(argv[4]);
        if (!input || !scores || !generated) throw std::runtime_error("Cannot open probe files");
        ggml_backend_load_all();
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.check_tensors = true;
        auto* model = llama_model_load_from_file(argv[1], model_params);
        if (!model) throw std::runtime_error("Model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = 2048;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_seq_max = 1;
        params.n_threads = 8;
        params.n_threads_batch = 8;
        auto* context = llama_init_from_model(model, params);
        if (!context) throw std::runtime_error("Context creation failed");
        const auto* vocab = llama_model_get_vocab(model);
        const int vocabulary = llama_vocab_n_tokens(vocab);
        int count, budget;
        while (input >> count >> budget) {
            if (count < 1 || count > 512 || budget < 0 || budget > 32)
                throw std::runtime_error("Invalid prompt length or generation budget");
            std::vector<llama_token> tokens(count);
            for (auto& token : tokens)
                if (!(input >> token) || token < 0 || token >= vocabulary)
                    throw std::runtime_error("Invalid input token");
            llama_memory_clear(llama_get_memory(context), true);
            auto batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            if (llama_decode(context, batch))
                throw std::runtime_error("Prompt decode failed");
            llama_batch_free(batch);
            auto* logits = llama_get_logits_ith(context, -1);
            for (int row = 0; row < count; ++row) {
                const auto* values = llama_get_logits_ith(context, row);
                for (int i = 0; i < vocabulary; ++i)
                    if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite logits");
                scores.write(reinterpret_cast<const char*>(values), vocabulary * sizeof(float));
            }
            for (int step = 0; step < budget; ++step) {
                logits = llama_get_logits_ith(context, -1);
                for (int i = 0; i < vocabulary; ++i)
                    if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite generation logits");
                llama_token next = std::max_element(logits, logits + vocabulary) - logits;
                generated << next << ' ';
                if (llama_vocab_is_eog(vocab, next)) break;
                if (llama_decode(context, llama_batch_get_one(&next, 1)))
                    throw std::runtime_error("Generation decode failed");
            }
            generated << '\n';
            generated.flush();
            scores.flush();
            if (!scores || !generated) throw std::runtime_error("Output write failed");
        }
        if (!input.eof()) throw std::runtime_error("Malformed prompt input");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
