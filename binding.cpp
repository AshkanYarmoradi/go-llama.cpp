// binding.cpp - Go-Llama.cpp binding for latest llama.cpp API
// Rewritten for llama.cpp with new sampler and vocab APIs

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include "binding.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <regex>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

// Binding state structure
struct llama_binding_state {
    llama_model * model;
    llama_context * ctx;
};

// Parameters structure to pass sampling/generation config
struct binding_params {
    std::string prompt;
    std::string grammar;
    std::vector<std::string> antiprompt;
    
    int32_t seed = LLAMA_DEFAULT_SEED;
    int32_t n_threads = 4;
    int32_t n_predict = 128;
    int32_t n_ctx = 512;
    int32_t n_batch = 512;
    int32_t n_keep = 0;
    int32_t repeat_last_n = 64;
    int32_t n_draft = 8;
    
    float top_p = 0.95f;
    float min_p = 0.05f;
    float temp = 0.80f;
    float repeat_penalty = 1.10f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float tfs_z = 1.0f;
    float typical_p = 1.0f;
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    float rope_freq_base = 0.0f;
    float rope_freq_scale = 0.0f;
    
    // XTC sampling parameters
    float xtc_probability = 0.0f;
    float xtc_threshold = 0.5f;
    
    // DRY sampling parameters
    float dry_multiplier = 0.0f;
    float dry_base = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    
    // Top-N Sigma sampling
    float top_n_sigma = 0.0f;
    
    int32_t top_k = 40;
    int32_t mirostat = 0;
    
    bool ignore_eos = false;
    bool memory_f16 = true;
    bool use_mmap = true;
    bool use_mlock = false;
    bool penalize_nl = true;
    bool prompt_cache_all = false;
    bool prompt_cache_ro = false;
    
    std::string path_prompt_cache;
    std::string main_gpu;
    std::string tensor_split;
    std::vector<llama_logit_bias> logit_bias;
};

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        _exit(130);
    }
}
#endif

// Helper function to tokenize with new API
static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, true);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.c_str(), text.length(), result.data(), result.size(), add_special, true);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

// Helper function to convert token to string
static std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special = true) {
    std::string result;
    result.resize(32);
    int n_chars = llama_token_to_piece(vocab, token, &result[0], result.size(), 0, special);
    if (n_chars < 0) {
        result.resize(-n_chars);
        n_chars = llama_token_to_piece(vocab, token, &result[0], result.size(), 0, special);
        GGML_ASSERT(n_chars <= (int)result.size());
    }
    result.resize(n_chars);
    return result;
}

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings) {
    binding_params* params_p = (binding_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    llama_model* model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Tokenize the prompt
    bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = tokenize_prompt(vocab, params_p->prompt, add_bos);
    
    if (tokens.empty()) {
        fprintf(stderr, "%s: error: prompt is empty\n", __func__);
        return 1;
    }
    
    // Create batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    
    // Decode
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: failed to decode\n", __func__);
        return 1;
    }
    
    const int n_embd = llama_model_n_embd(model);
    const float * embeddings = llama_get_embeddings(ctx);
    
    if (embeddings == nullptr) {
        fprintf(stderr, "%s: embeddings not available\n", __func__);
        return 1;
    }
    
    for (int i = 0; i < n_embd; i++) {
        res_embeddings[i] = embeddings[i];
    }
    
    return 0;
}

int get_token_embeddings(void* params_ptr, void* state_pr, int *tokens, int tokenSize, float * res_embeddings) {
    binding_params* params_p = (binding_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_model* model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Convert tokens to prompt string
    std::string prompt;
    for (int i = 0; i < tokenSize; i++) {
        prompt += token_to_piece(vocab, tokens[i]);
    }
    params_p->prompt = prompt;
    
    return get_embeddings(params_ptr, state_pr, res_embeddings);
}

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    binding_params* params_p = (binding_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    llama_model* model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    const int n_ctx = llama_n_ctx(ctx);
    
    // Set seed if specified
    if (params_p->seed != LLAMA_DEFAULT_SEED) {
        // Seed is now set during sampler initialization
    }
    
    // Tokenize prompt
    bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> embd_inp = tokenize_prompt(vocab, params_p->prompt, add_bos);
    
    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_vocab_bos(vocab));
    }
    
    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    
    // Initialize sampler chain
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    
    // Add samplers based on parameters
    if (params_p->temp <= 0) {
        // Greedy sampling
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        // Add DRY sampler if enabled (before other samplers)
        if (params_p->dry_multiplier > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                vocab,
                llama_model_n_ctx_train(model),
                params_p->dry_multiplier,
                params_p->dry_base,
                params_p->dry_allowed_length,
                params_p->dry_penalty_last_n,
                nullptr, 0  // no custom sequence breakers
            ));
        }
        
        // Add penalty sampler if needed
        if (params_p->repeat_penalty != 1.0f || params_p->frequency_penalty != 0.0f || params_p->presence_penalty != 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                params_p->repeat_last_n,
                params_p->repeat_penalty,
                params_p->frequency_penalty,
                params_p->presence_penalty
            ));
        }
        
        if (params_p->mirostat == 1) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params_p->temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab),
                params_p->seed,
                params_p->mirostat_tau,
                params_p->mirostat_eta,
                100 // m
            ));
        } else if (params_p->mirostat == 2) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params_p->temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
                params_p->seed,
                params_p->mirostat_tau,
                params_p->mirostat_eta
            ));
        } else {
            // Standard sampling chain
            
            // Top-N Sigma sampling (if enabled)
            if (params_p->top_n_sigma > 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_top_n_sigma(params_p->top_n_sigma));
            }
            
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params_p->top_k));
            if (params_p->tfs_z < 1.0f) {
                // Note: TFS is removed in new API, skip
            }
            if (params_p->typical_p < 1.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_typical(params_p->typical_p, 1));
            }
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params_p->top_p, 1));
            if (params_p->min_p > 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params_p->min_p, 1));
            }
            
            // XTC sampling (if enabled)
            if (params_p->xtc_probability > 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
                    params_p->xtc_probability,
                    params_p->xtc_threshold,
                    1,  // min_keep
                    params_p->seed
                ));
            }
            
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params_p->temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(params_p->seed));
        }
    }
    
    // Add grammar sampler if specified
    if (!params_p->grammar.empty()) {
        llama_sampler * grammar_smpl = llama_sampler_init_grammar(vocab, params_p->grammar.c_str(), "root");
        if (grammar_smpl != nullptr) {
            llama_sampler_chain_add(smpl, grammar_smpl);
        }
    }
    
    std::string res = "";
    std::vector<llama_token> embd;
    
    int n_past = 0;
    int n_remain = params_p->n_predict;
    int n_consumed = 0;
    
    bool is_antiprompt = false;
    
    while (n_remain != 0) {
        // Process tokens
        if (!embd.empty()) {
            // Check context overflow
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params_p->n_keep;
                n_past = std::max(1, params_p->n_keep);
                embd.insert(embd.begin(), embd_inp.begin() + params_p->n_keep, embd_inp.begin() + params_p->n_keep + n_left/2);
            }
            
            // Create batch and decode
            for (int i = 0; i < (int) embd.size(); i += params_p->n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params_p->n_batch) {
                    n_eval = params_p->n_batch;
                }
                
                llama_batch batch = llama_batch_get_one(&embd[i], n_eval);
                
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "%s: failed to decode\n", __func__);
                    llama_sampler_free(smpl);
                    return 1;
                }
                n_past += n_eval;
            }
        }
        
        embd.clear();
        
        if ((int) embd_inp.size() <= n_consumed) {
            // Sample next token
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, id);
            
            // Add to output
            embd.push_back(id);
            --n_remain;
            
            // Get token string and callback
            std::string token_str = token_to_piece(vocab, id);
            if (!tokenCallback(state_pr, (char*)token_str.c_str())) {
                break;
            }
            
            // Append to result
            res += token_str;
        } else {
            // Still processing input
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params_p->n_batch) {
                    break;
                }
            }
        }
        
        // Check for antiprompt
        if ((int) embd_inp.size() <= n_consumed) {
            for (const std::string & antiprompt : params_p->antiprompt) {
                if (res.length() >= antiprompt.length()) {
                    if (res.substr(res.length() - antiprompt.length()) == antiprompt) {
                        is_antiprompt = true;
                        break;
                    }
                }
            }
        }
        
        if (is_antiprompt) {
            break;
        }
        
        // Check for EOS
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back())) {
            break;
        }
    }
    
    if (debug) {
        llama_perf_context_print(ctx);
    }
    
    llama_sampler_free(smpl);
    
    strcpy(result, res.c_str());
    return 0;
}

void llama_binding_free_model(void *state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (state->ctx != nullptr) {
        llama_free(state->ctx);
    }
    if (state->model != nullptr) {
        llama_model_free(state->model);
    }
    delete state;
}

void llama_free_params(void* params_ptr) {
    binding_params* params = (binding_params*) params_ptr;
    delete params;
}

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result) {
    binding_params* params_p = (binding_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_model* model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = tokenize_prompt(vocab, params_p->prompt, add_bos);
    
    for (size_t i = 0; i < tokens.size(); i++) {
        result[i] = tokens[i];
    }
    
    return (int)tokens.size();
}

std::vector<std::string> create_vector(const char** strings, int count) {
    std::vector<std::string> vec;
    for (int i = 0; i < count; i++) {
        vec.push_back(std::string(strings[i]));
    }
    return vec;
}

void delete_vector(std::vector<std::string>* vec) {
    delete vec;
}

int load_state(void *ctx, char *statefile, char*modes) {
    llama_binding_state* state = (llama_binding_state*) ctx;
    llama_context* lctx = state->ctx;
    
    const size_t state_size = llama_state_get_size(lctx);
    uint8_t * state_mem = new uint8_t[state_size];
    
    FILE *fp_read = fopen(statefile, modes);
    if (fp_read == nullptr) {
        fprintf(stderr, "%s: failed to open state file for reading\n", __func__);
        delete[] state_mem;
        return 1;
    }
    
    const size_t ret = fread(state_mem, 1, state_size, fp_read);
    if (ret != state_size) {
        fprintf(stderr, "%s: failed to read state\n", __func__);
        fclose(fp_read);
        delete[] state_mem;
        return 1;
    }
    
    size_t read_size = llama_state_set_data(lctx, state_mem, state_size);
    if (read_size == 0) {
        fprintf(stderr, "%s: failed to set state data\n", __func__);
        fclose(fp_read);
        delete[] state_mem;
        return 1;
    }
    
    fclose(fp_read);
    delete[] state_mem;
    return 0;
}

void save_state(void *ctx, char *dst, char*modes) {
    llama_binding_state* state = (llama_binding_state*) ctx;
    llama_context* lctx = state->ctx;
    
    const size_t state_size = llama_state_get_size(lctx);
    uint8_t * state_mem = new uint8_t[state_size];
    
    FILE *fp_write = fopen(dst, modes);
    if (fp_write == nullptr) {
        fprintf(stderr, "%s: failed to open state file for writing\n", __func__);
        delete[] state_mem;
        return;
    }
    
    size_t written = llama_state_get_data(lctx, state_mem, state_size);
    if (written > 0) {
        fwrite(state_mem, 1, written, fp_write);
    }
    
    fclose(fp_write);
    delete[] state_mem;
}

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float min_p, float temp, float repeat_penalty, int repeat_last_n, 
                            bool ignore_eos, bool memory_f16, int n_batch, int n_keep, 
                            const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, 
                            int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, 
                            const char *logit_bias, const char *session_file, bool prompt_cache_all, 
                            bool mlock, bool mmap, const char *maingpu, const char *tensorsplit, 
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, 
                            float rope_freq_scale, int n_draft,
                            float xtc_probability, float xtc_threshold,
                            float dry_multiplier, float dry_base, int dry_allowed_length, int dry_penalty_last_n,
                            float top_n_sigma) {
    
    binding_params* params = new binding_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;
    params->prompt_cache_ro = prompt_cache_ro;
    params->top_k = top_k;
    params->top_p = top_p;
    params->min_p = min_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->use_mmap = mmap;
    params->use_mlock = mlock;
    params->repeat_penalty = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;
    params->grammar = std::string(grammar);
    params->rope_freq_base = rope_freq_base;
    params->rope_freq_scale = rope_freq_scale;
    params->n_draft = n_draft;
    params->main_gpu = std::string(maingpu);
    params->tensor_split = std::string(tensorsplit);
    params->prompt_cache_all = prompt_cache_all;
    params->path_prompt_cache = std::string(session_file);
    params->ignore_eos = ignore_eos;
    
    // New sampler parameters
    params->xtc_probability = xtc_probability;
    params->xtc_threshold = xtc_threshold;
    params->dry_multiplier = dry_multiplier;
    params->dry_base = dry_base;
    params->dry_allowed_length = dry_allowed_length;
    params->dry_penalty_last_n = dry_penalty_last_n;
    params->top_n_sigma = top_n_sigma;
    
    if (antiprompt_count > 0) {
        params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }
    
    params->tfs_z = tfs_z;
    params->typical_p = typical_p;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->penalize_nl = penalize_nl;
    params->frequency_penalty = frequency_penalty;
    params->prompt = std::string(prompt);
    
    // Parse logit bias if provided
    if (logit_bias != nullptr && logit_bias[0] != '\0') {
        std::stringstream ss(logit_bias);
        llama_token key;
        char sign;
        std::string value_str;
        if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
            llama_logit_bias bias;
            bias.token = key;
            bias.bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
            params->logit_bias.push_back(bias);
        }
    }
    
    return params;
}

void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, 
                 bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch, 
                 const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base, 
                 float rope_freq_scale, const char *lora, const char *lora_base) {
    
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname);
    
    // Initialize backend
    llama_backend_init();
    
    if (numa) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }
    
    // Setup model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = mmap;
    model_params.use_mlock = mlock;
    
    // Parse main GPU
    if (maingpu != nullptr && maingpu[0] != '\0') {
        model_params.main_gpu = std::stoi(maingpu);
    }
    
    // Parse tensor split
    static float tensor_split_values[128] = {0};
    if (tensorsplit != nullptr && tensorsplit[0] != '\0') {
        std::string arg_next = tensorsplit;
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        
        for (size_t i = 0; i < 128 && i < split_arg.size(); ++i) {
            tensor_split_values[i] = std::stof(split_arg[i]);
        }
        model_params.tensor_split = tensor_split_values;
    }
    
    // Load model
    llama_model * model = llama_model_load_from_file(fname, model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, fname);
        return nullptr;
    }
    
    // Setup context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch;
    ctx_params.embeddings = embeddings;
    
    if (rope_freq_base != 0.0f) {
        ctx_params.rope_freq_base = rope_freq_base;
    }
    if (rope_freq_scale != 0.0f) {
        ctx_params.rope_freq_scale = rope_freq_scale;
    }
    
    // Create context
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context\n", __func__);
        llama_model_free(model);
        return nullptr;
    }
    
    // Load LoRA adapter if specified
    if (lora != nullptr && lora[0] != '\0') {
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, lora);
        if (adapter != nullptr) {
            float scale = 1.0f;
            llama_set_adapters_lora(ctx, &adapter, 1, &scale);
        } else {
            fprintf(stderr, "%s: warning: failed to load LoRA adapter '%s'\n", __func__, lora);
        }
    }
    
    // Create and return state
    llama_binding_state * state = new llama_binding_state;
    state->model = model;
    state->ctx = ctx;
    
    return state;
}

// Model info functions
int get_model_n_vocab(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_n_tokens(vocab);
}

int get_model_n_ctx_train(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_ctx_train(state->model);
}

int get_model_n_embd(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_embd(state->model);
}

int get_model_n_layer(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_layer(state->model);
}

long long get_model_size(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (long long)llama_model_size(state->model);
}

long long get_model_n_params(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (long long)llama_model_n_params(state->model);
}

int get_model_description(void* state_ptr, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_desc(state->model, buf, buf_size);
}

int get_model_chat_template(void* state_ptr, const char* name, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const char* tmpl = llama_model_chat_template(state->model, name);
    if (tmpl == nullptr) {
        return 0;
    }
    int len = strlen(tmpl);
    if (len >= buf_size) {
        len = buf_size - 1;
    }
    strncpy(buf, tmpl, len);
    buf[len] = '\0';
    return len;
}

// Special token functions
int get_vocab_bos(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_bos(vocab);
}

int get_vocab_eos(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_eos(vocab);
}

int get_vocab_eot(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_eot(vocab);
}

int get_vocab_nl(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_nl(vocab);
}

int get_vocab_sep(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_sep(vocab);
}

bool get_vocab_add_bos(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_get_add_bos(vocab);
}

bool get_vocab_add_eos(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_vocab_get_add_eos(vocab);
}

// Model architecture queries
bool model_has_encoder(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_has_encoder(state->model);
}

bool model_has_decoder(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_has_decoder(state->model);
}

bool model_is_recurrent(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_is_recurrent(state->model);
}

// System info
int get_system_info(char* buf, int buf_size) {
    const char* info = llama_print_system_info();
    if (info == nullptr) {
        return 0;
    }
    int len = strlen(info);
    if (len >= buf_size) {
        len = buf_size - 1;
    }
    strncpy(buf, info, len);
    buf[len] = '\0';
    return len;
}

int apply_chat_template(void* state_ptr, const char* tmpl, const char* messages_json,
                        bool add_generation_prompt, char* result, int result_size) {
    // Note: This is a simplified implementation - full implementation would need JSON parsing
    // For now, return -1 to indicate not implemented
    // Users should use the chat template string directly and format messages themselves
    (void)state_ptr;
    (void)tmpl;
    (void)messages_json;
    (void)add_generation_prompt;
    (void)result;
    (void)result_size;
    return -1;
}
