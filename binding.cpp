// binding.cpp - Go-Llama.cpp binding for latest llama.cpp API
// Rewritten for llama.cpp with new sampler and vocab APIs

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include "binding.h"

#include <algorithm>
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
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// Binding state structure
struct llama_binding_state {
    llama_model * model;
    llama_context * ctx;
    // Active LoRA adapters applied to the context, kept so the whole set can be
    // re-applied (llama_set_adapters_lora replaces the set) and freed on teardown.
    std::vector<llama_adapter_lora *> lora_adapters;
    std::vector<float> lora_scales;
};

// Wrapper around a llama_batch that also remembers its capacity, so batch_add
// can bounds-check (llama_batch itself only stores the current token count).
struct binding_batch {
    llama_batch batch;
    int32_t capacity;
    int32_t n_seq_max;
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
    
    // Each call embeds its own prompt, so drop the cells left by earlier calls.
    // Without this the context fills up and llama_decode runs out of slots.
    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);

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

// ---------------------------------------------------------------------------
// Low-level batching, decoding, and output access
// ---------------------------------------------------------------------------

void* batch_init(int n_tokens, int n_seq_max) {
    binding_batch* w = new binding_batch;
    w->batch = llama_batch_init(n_tokens, 0, n_seq_max);
    w->capacity = n_tokens;
    w->n_seq_max = n_seq_max;
    return w;
}

void batch_free(void* batch_ptr) {
    binding_batch* w = (binding_batch*) batch_ptr;
    llama_batch_free(w->batch);
    delete w;
}

void batch_clear(void* batch_ptr) {
    ((binding_batch*) batch_ptr)->batch.n_tokens = 0;
}

int batch_n_tokens(void* batch_ptr) {
    return ((binding_batch*) batch_ptr)->batch.n_tokens;
}

// Append one token at position pos for the given sequence ids, flagging whether
// its output (logits/embeddings) is wanted. Returns the slot index, -1 if the
// batch is full, or -2 if n_seq_ids exceeds the batch's configured n_seq_max.
int batch_add(void* batch_ptr, int token, int pos, const int* seq_ids, int n_seq_ids, bool logits) {
    binding_batch* w = (binding_batch*) batch_ptr;
    llama_batch & b = w->batch;
    if (b.n_tokens >= w->capacity) {
        return -1;
    }
    if (n_seq_ids > w->n_seq_max) {
        return -2;
    }
    const int idx = b.n_tokens;
    b.token[idx]    = token;
    b.pos[idx]      = pos;
    b.n_seq_id[idx] = n_seq_ids;
    for (int k = 0; k < n_seq_ids; k++) {
        b.seq_id[idx][k] = seq_ids[k];
    }
    b.logits[idx] = logits ? 1 : 0;
    b.n_tokens++;
    return idx;
}

// Decode a batch using the KV cache. Returns llama_decode's status: 0 success,
// 1 = no KV slot, 2 = aborted, negative = error.
int decode_batch(void* state_ptr, void* batch_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_decode(state->ctx, ((binding_batch*) batch_ptr)->batch);
}

// Encode a batch (encoder-decoder models). Returns 0 on success, negative on error.
int encode_batch(void* state_ptr, void* batch_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_encode(state->ctx, ((binding_batch*) batch_ptr)->batch);
}

// Copy up to out_size logits for the i-th output token (-1 = last) into out.
// Returns the number copied, or 0 if unavailable.
int get_logits_ith(void* state_ptr, int i, float* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const float* logits = llama_get_logits_ith(state->ctx, i);
    if (logits == nullptr) {
        return 0;
    }
    int n = llama_vocab_n_tokens(llama_model_get_vocab(state->model));
    if (n > out_size) {
        n = out_size;
    }
    for (int k = 0; k < n; k++) {
        out[k] = logits[k];
    }
    return n;
}

// Copy up to out_size embeddings for the i-th output token (-1 = last) into out.
int get_embeddings_ith(void* state_ptr, int i, float* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const float* emb = llama_get_embeddings_ith(state->ctx, i);
    if (emb == nullptr) {
        return 0;
    }
    int n = llama_model_n_embd(state->model);
    if (n > out_size) {
        n = out_size;
    }
    for (int k = 0; k < n; k++) {
        out[k] = emb[k];
    }
    return n;
}

// Copy up to out_size pooled embeddings for an entire sequence into out.
int get_embeddings_seq(void* state_ptr, int seq_id, float* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const float* emb = llama_get_embeddings_seq(state->ctx, seq_id);
    if (emb == nullptr) {
        return 0;
    }
    int n = llama_model_n_embd(state->model);
    if (n > out_size) {
        n = out_size;
    }
    for (int k = 0; k < n; k++) {
        out[k] = emb[k];
    }
    return n;
}

// KV-cache / sequence management on the context memory.
void memory_clear(void* state_ptr, bool data) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_memory_clear(llama_get_memory(state->ctx), data);
}

bool memory_seq_rm(void* state_ptr, int seq_id, int p0, int p1) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_memory_seq_rm(llama_get_memory(state->ctx), seq_id, p0, p1);
}

void memory_seq_cp(void* state_ptr, int src, int dst, int p0, int p1) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_memory_seq_cp(llama_get_memory(state->ctx), src, dst, p0, p1);
}

void memory_seq_keep(void* state_ptr, int seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_memory_seq_keep(llama_get_memory(state->ctx), seq_id);
}

int llama_predict(void* params_ptr, void* state_pr, char* result, int result_size, bool debug) {
    binding_params* params_p = (binding_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    llama_model* model = state->model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_memory_t mem = llama_get_memory(ctx);

    const int n_ctx = llama_n_ctx(ctx);

    // Each prediction starts from a clean cache: n_past below counts from zero,
    // so cells left over from an earlier call would both desync the position
    // bookkeeping and fill up the context. Use save_state/load_state to carry
    // state across calls on purpose.
    llama_memory_seq_rm(mem, -1, -1, -1);

    // Note: the RNG seed is applied when the sampler chain is built below
    // (llama_sampler_init_dist / mirostat), not on the context.

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

    // Apply logit bias first so it influences every downstream sampler,
    // including greedy selection. params_p->logit_bias is populated only when
    // the caller passes a "token(+|-)value" bias string; previously it was
    // parsed but never wired into the chain, so the bias was silently ignored.
    if (!params_p->logit_bias.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab),
            (int32_t) params_p->logit_bias.size(),
            params_p->logit_bias.data()));
    }

    // Add samplers based on parameters
    if (params_p->temp <= 0) {
        // Greedy sampling
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        // Add DRY sampler if enabled (before other samplers)
        if (params_p->dry_multiplier > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                vocab,
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
                llama_vocab_n_tokens(vocab),
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

    // Tokens kept in front of the context when it has to be shifted. It can
    // never exceed the prompt, otherwise the shift would discard tokens that
    // were never there.
    const int n_keep = std::min(std::max(params_p->n_keep, 0), (int) embd_inp.size());

    bool is_antiprompt = false;

    while (n_remain != 0) {
        // Process tokens
        if (!embd.empty()) {
            // Context is full: discard the oldest half of the tokens after
            // n_keep and move the rest down, so the cache has free cells again.
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_discard = (n_past - n_keep) / 2;

                if (n_discard <= 0 || n_keep + (int) embd.size() > n_ctx) {
                    fprintf(stderr, "%s: error: context too small to shift (n_ctx = %d, n_keep = %d)\n",
                            __func__, n_ctx, n_keep);
                    llama_sampler_free(smpl);
                    return 1;
                }

                llama_memory_seq_rm (mem, 0, n_keep, n_keep + n_discard);
                llama_memory_seq_add(mem, 0, n_keep + n_discard, n_past, -n_discard);

                n_past -= n_discard;
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
            if (!tokenCallback(state_pr, &token_str[0])) {
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

    // Bounded copy: a token decodes to several bytes, so `res` is routinely
    // longer than the caller's token limit. Truncate instead of overrunning.
    if (result_size > 0) {
        snprintf(result, (size_t) result_size, "%s", res.c_str());
    }
    return 0;
}

void llama_binding_free_model(void *state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (state->ctx != nullptr) {
        llama_free(state->ctx);
    }
    // Free adapters added via apply_lora_adapter while the model is still alive
    // (llama_adapter_lora_free detaches each from the model's set), then free
    // the model itself, which releases any remaining untracked adapters.
    for (llama_adapter_lora * adapter : state->lora_adapters) {
        llama_adapter_lora_free(adapter);
    }
    state->lora_adapters.clear();
    state->lora_scales.clear();
    if (state->model != nullptr) {
        llama_model_free(state->model);
    }
    delete state;
}

// Load a LoRA adapter from file and add it to the set active on the context.
// llama_set_adapters_lora replaces the whole set, so the binding tracks every
// applied adapter and re-applies them together. Returns 0 on success, non-zero
// if the adapter could not be loaded or applied.
int apply_lora_adapter(void* state_ptr, const char* path, float scale) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora * adapter = llama_adapter_lora_init(state->model, path);
    if (adapter == nullptr) {
        return 1;
    }
    state->lora_adapters.push_back(adapter);
    state->lora_scales.push_back(scale);
    return llama_set_adapters_lora(state->ctx, state->lora_adapters.data(),
                                   state->lora_adapters.size(), state->lora_scales.data());
}

// Detach and free every LoRA adapter previously applied via apply_lora_adapter.
int clear_lora_adapters(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    int ret = llama_set_adapters_lora(state->ctx, nullptr, 0, nullptr);
    for (llama_adapter_lora * adapter : state->lora_adapters) {
        llama_adapter_lora_free(adapter);
    }
    state->lora_adapters.clear();
    state->lora_scales.clear();
    return ret;
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

int tokenize_text(void* state_ptr, const char* text, int text_len,
                  int* tokens_out, int max_tokens,
                  bool add_special, bool parse_special) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_tokenize(vocab, text, text_len, (llama_token*) tokens_out,
                          max_tokens, add_special, parse_special);
}

int detokenize_text(void* state_ptr, const int* tokens, int n_tokens,
                    char* buf, int buf_size,
                    bool remove_special, bool unparse_special) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_detokenize(vocab, (const llama_token*) tokens, n_tokens, buf,
                            buf_size, remove_special, unparse_special);
}

int token_to_piece_str(void* state_ptr, int token, char* buf, int buf_size, bool special) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);
    return llama_token_to_piece(vocab, token, buf, buf_size, 0, special);
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
    
    // Parse logit bias if provided. std::stof throws on a malformed value, and
    // an exception escaping into cgo aborts the process, so a bad bias string
    // is reported and ignored instead.
    if (logit_bias != nullptr && logit_bias[0] != '\0') {
        std::stringstream ss(logit_bias);
        llama_token key;
        char sign;
        std::string value_str;
        if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
            try {
                llama_logit_bias bias;
                bias.token = key;
                bias.bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                params->logit_bias.push_back(bias);
            } catch (const std::exception & e) {
                fprintf(stderr, "%s: ignoring malformed logit_bias %s: %s\n",
                        __func__, logit_bias, e.what());
            }
        } else {
            fprintf(stderr, "%s: ignoring malformed logit_bias %s (expected \"token(+|-)value\")\n",
                    __func__, logit_bias);
        }
    }
    
    return params;
}

// Shared implementation. paths holds n_paths GGUF files: one for an ordinary
// model, or every shard of a model whose filenames do not follow llama.cpp's
// "-00001-of-0000N.gguf" convention (which it can otherwise infer from the
// first shard alone).
static void* load_model_impl(const char **paths, int n_paths,
                 int n_ctx, int n_seed, bool memory_f16, bool mlock,
                 bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch,
                 const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base,
                 float rope_freq_scale, const char *lora, const char *lora_base) {

    if (paths == nullptr || n_paths <= 0 || paths[0] == nullptr) {
        fprintf(stderr, "%s: error: no model path given\n", __func__);
        return nullptr;
    }
    const char *fname = paths[0];

    // These parameters are retained for C ABI stability with the Go layer but
    // are no longer consumed by llama.cpp: the seed is applied when the sampler
    // chain is built, KV-cache precision is chosen via the context params, and
    // low_vram / lora_base were removed from the upstream API.
    (void) n_seed;
    (void) memory_f16;
    (void) low_vram;
    (void) lora_base;

    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname);
    
    // Initialize backend
    llama_backend_init();
    
    if (numa) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }
    
    // Setup model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    // llama.cpp replaced the use_mmap/use_mlock booleans with a single
    // load_mode enum. Preserve the binding's semantics: mlock implies mmap
    // (LLAMA_LOAD_MODE_MLOCK == "mmap + keep resident"), a plain mmap request
    // maps to LLAMA_LOAD_MODE_MMAP, and neither maps to LLAMA_LOAD_MODE_NONE.
    model_params.load_mode = mlock ? LLAMA_LOAD_MODE_MLOCK
                           : mmap  ? LLAMA_LOAD_MODE_MMAP
                                   : LLAMA_LOAD_MODE_NONE;
    
    // Parse main GPU and tensor split. Both use std::stoi/std::stof, which
    // throw on malformed input; an exception escaping into cgo aborts the
    // process, so a bad value is reported and the default kept.
    if (maingpu != nullptr && maingpu[0] != '\0') {
        try {
            model_params.main_gpu = std::stoi(maingpu);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: ignoring malformed main_gpu %s: %s\n", __func__, maingpu, e.what());
        }
    }

    static float tensor_split_values[128] = {0};
    if (tensorsplit != nullptr && tensorsplit[0] != '\0') {
        std::string arg_next = tensorsplit;
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};

        try {
            for (size_t i = 0; i < 128 && i < split_arg.size(); ++i) {
                tensor_split_values[i] = std::stof(split_arg[i]);
            }
            model_params.tensor_split = tensor_split_values;
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: ignoring malformed tensor_split %s: %s\n", __func__, tensorsplit, e.what());
        }
    }
    
    // Load model
    llama_model * model = n_paths == 1
        ? llama_model_load_from_file(fname, model_params)
        : llama_model_load_from_splits(paths, (size_t) n_paths, model_params);
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

    // llama_context_default_params sets no_perf = true, which makes the engine
    // skip its own timing calls and leaves llama_perf_context reporting zeros.
    // The binding exposes those counters through Perf(), so enable them; the
    // cost is a couple of clock reads per decode.
    ctx_params.no_perf = false;
    
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

void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock,
                 bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch,
                 const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base,
                 float rope_freq_scale, const char *lora, const char *lora_base) {
    const char *paths[1] = { fname };
    return load_model_impl(paths, 1, n_ctx, n_seed, memory_f16, mlock, embeddings, mmap,
                           low_vram, n_gpu_layers, n_batch, maingpu, tensorsplit, numa,
                           rope_freq_base, rope_freq_scale, lora, lora_base);
}

// Loads a model from an explicit list of shards. Only needed when the shard
// filenames do not follow llama.cpp's own naming scheme; otherwise load_model
// with the first shard is enough.
void* load_model_splits(const char **paths, int n_paths,
                        int n_ctx, int n_seed, bool memory_f16, bool mlock,
                        bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch,
                        const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base,
                        float rope_freq_scale, const char *lora, const char *lora_base) {
    return load_model_impl(paths, n_paths, n_ctx, n_seed, memory_f16, mlock, embeddings, mmap,
                           low_vram, n_gpu_layers, n_batch, maingpu, tensorsplit, numa,
                           rope_freq_base, rope_freq_scale, lora, lora_base);
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
    // snprintf semantics: report the length the template needs, so a caller
    // whose buffer was too small can size the retry exactly. Chat templates
    // routinely run past a naive 4 KiB guess.
    return snprintf(buf, (size_t) buf_size, "%s", tmpl);
}

// Extended model geometry
int get_model_n_head(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_head(state->model);
}

int get_model_n_head_kv(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_head_kv(state->model);
}

int get_model_n_swa(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_swa(state->model);
}

float get_model_rope_freq_scale_train(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_rope_freq_scale_train(state->model);
}

// Model metadata (GGUF key-value header)
int get_model_meta_count(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_meta_count(state->model);
}

int get_model_meta_val_str(void* state_ptr, const char* key, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_meta_val_str(state->model, key, buf, (size_t) buf_size);
}

int get_model_meta_key_by_index(void* state_ptr, int i, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_meta_key_by_index(state->model, i, buf, (size_t) buf_size);
}

int get_model_meta_val_str_by_index(void* state_ptr, int i, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_meta_val_str_by_index(state->model, i, buf, (size_t) buf_size);
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

// Extended special tokens (padding, mask, fill-in-the-middle)
int get_vocab_pad(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_pad(llama_model_get_vocab(state->model));
}

int get_vocab_mask(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_mask(llama_model_get_vocab(state->model));
}

int get_vocab_fim_pre(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_pre(llama_model_get_vocab(state->model));
}

int get_vocab_fim_suf(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_suf(llama_model_get_vocab(state->model));
}

int get_vocab_fim_mid(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_mid(llama_model_get_vocab(state->model));
}

int get_vocab_fim_pad(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_pad(llama_model_get_vocab(state->model));
}

int get_vocab_fim_rep(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_rep(llama_model_get_vocab(state->model));
}

int get_vocab_fim_sep(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_fim_sep(llama_model_get_vocab(state->model));
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

// Backend capability queries (no model required)
bool backend_supports_mmap(void)          { return llama_supports_mmap(); }
bool backend_supports_mlock(void)         { return llama_supports_mlock(); }
bool backend_supports_gpu_offload(void)   { return llama_supports_gpu_offload(); }
bool backend_supports_rpc(void)           { return llama_supports_rpc(); }
int  backend_max_devices(void)            { return (int) llama_max_devices(); }
int  backend_max_parallel_sequences(void) { return (int) llama_max_parallel_sequences(); }

// ---------------------------------------------------------------------------
// Composable samplers
// ---------------------------------------------------------------------------

void* sampler_chain_init(void) {
    return llama_sampler_chain_init(llama_sampler_chain_default_params());
}

void sampler_chain_add(void* chain, void* smpl) {
    llama_sampler_chain_add((llama_sampler*) chain, (llama_sampler*) smpl);
}

void sampler_free(void* smpl) {
    llama_sampler_free((llama_sampler*) smpl);
}

void sampler_reset(void* smpl) {
    llama_sampler_reset((llama_sampler*) smpl);
}

void sampler_accept(void* smpl, int token) {
    llama_sampler_accept((llama_sampler*) smpl, token);
}

int sampler_sample(void* state_ptr, void* smpl, int idx) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_sampler_sample((llama_sampler*) smpl, state->ctx, idx);
}

void* sampler_init_greedy(void)                    { return llama_sampler_init_greedy(); }
void* sampler_init_dist(unsigned int seed)         { return llama_sampler_init_dist(seed); }
void* sampler_init_top_k(int k)                    { return llama_sampler_init_top_k(k); }
void* sampler_init_top_p(float p, int min_keep)    { return llama_sampler_init_top_p(p, (size_t) min_keep); }
void* sampler_init_min_p(float p, int min_keep)    { return llama_sampler_init_min_p(p, (size_t) min_keep); }
void* sampler_init_typical(float p, int min_keep)  { return llama_sampler_init_typical(p, (size_t) min_keep); }
void* sampler_init_temp(float t)                   { return llama_sampler_init_temp(t); }
void* sampler_init_temp_ext(float t, float delta, float exponent) { return llama_sampler_init_temp_ext(t, delta, exponent); }
void* sampler_init_xtc(float p, float t, int min_keep, unsigned int seed) { return llama_sampler_init_xtc(p, t, (size_t) min_keep, seed); }
void* sampler_init_top_n_sigma(float n)            { return llama_sampler_init_top_n_sigma(n); }
void* sampler_init_mirostat_v2(unsigned int seed, float tau, float eta) { return llama_sampler_init_mirostat_v2(seed, tau, eta); }
void* sampler_init_penalties(void* state_ptr, int last_n, float repeat, float freq, float present) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    return llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), last_n, repeat, freq, present);
}

void* sampler_init_grammar(void* state_ptr, const char* grammar, const char* root) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    return llama_sampler_init_grammar(vocab, grammar, root);
}

void* sampler_init_dry(void* state_ptr, float multiplier, float base, int allowed_length, int penalty_last_n) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    return llama_sampler_init_dry(vocab, multiplier, base, allowed_length, penalty_last_n,
                                  nullptr, 0);
}

//
// Context runtime introspection and control
//
// The engine may clamp or round the values requested through llama_context_params,
// so these report what the context actually uses rather than what was asked for.
//

int context_n_ctx(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_ctx(state->ctx);
}

int context_n_ctx_seq(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_ctx_seq(state->ctx);
}

int context_n_batch(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_batch(state->ctx);
}

int context_n_ubatch(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_ubatch(state->ctx);
}

int context_n_seq_max(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_seq_max(state->ctx);
}

int context_n_rs_seq(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_n_rs_seq(state->ctx);
}

int context_pooling_type(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_pooling_type(state->ctx);
}

int context_n_threads(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_n_threads(state->ctx);
}

int context_n_threads_batch(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_n_threads_batch(state->ctx);
}

void context_set_n_threads(void* state_ptr, int n_threads, int n_threads_batch) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_set_n_threads(state->ctx, n_threads, n_threads_batch);
}

void context_set_embeddings(void* state_ptr, bool embeddings) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_set_embeddings(state->ctx, embeddings);
}

void context_set_causal_attn(void* state_ptr, bool causal_attn) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_set_causal_attn(state->ctx, causal_attn);
}

void context_synchronize(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_synchronize(state->ctx);
}

//
// KV-cache (memory) operations not already exposed
//

void memory_seq_add(void* state_ptr, int seq_id, int p0, int p1, int delta) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_memory_seq_add(llama_get_memory(state->ctx), seq_id, p0, p1, delta);
}

void memory_seq_div(void* state_ptr, int seq_id, int p0, int p1, int d) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_memory_seq_div(llama_get_memory(state->ctx), seq_id, p0, p1, d);
}

int memory_seq_pos_min(void* state_ptr, int seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_memory_seq_pos_min(llama_get_memory(state->ctx), seq_id);
}

int memory_seq_pos_max(void* state_ptr, int seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_memory_seq_pos_max(llama_get_memory(state->ctx), seq_id);
}

bool memory_can_shift(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_memory_can_shift(llama_get_memory(state->ctx));
}

//
// Performance counters
//

void perf_context(void* state_ptr, double* t_start_ms, double* t_load_ms,
                  double* t_p_eval_ms, double* t_eval_ms,
                  int* n_p_eval, int* n_eval, int* n_reused) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_perf_context_data d = llama_perf_context(state->ctx);
    if (t_start_ms)  *t_start_ms  = d.t_start_ms;
    if (t_load_ms)   *t_load_ms   = d.t_load_ms;
    if (t_p_eval_ms) *t_p_eval_ms = d.t_p_eval_ms;
    if (t_eval_ms)   *t_eval_ms   = d.t_eval_ms;
    if (n_p_eval)    *n_p_eval    = d.n_p_eval;
    if (n_eval)      *n_eval      = d.n_eval;
    if (n_reused)    *n_reused    = d.n_reused;
}

void perf_context_reset(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_perf_context_reset(state->ctx);
}

// llama_perf_sampler aborts the process on a null or non-chain sampler. The Go
// wrapper only calls these on a real chain; guard null here too so the exposed
// C API cannot abort on it (a non-null non-chain still must not be passed).
void perf_sampler(void* smpl, double* t_sample_ms, int* n_sample) {
    if (smpl == nullptr) {
        if (t_sample_ms) *t_sample_ms = 0.0;
        if (n_sample)    *n_sample    = 0;
        return;
    }
    const llama_perf_sampler_data d = llama_perf_sampler((const llama_sampler*) smpl);
    if (t_sample_ms) *t_sample_ms = d.t_sample_ms;
    if (n_sample)    *n_sample    = d.n_sample;
}

void perf_sampler_reset(void* smpl) {
    if (smpl == nullptr) {
        return;
    }
    llama_perf_sampler_reset((llama_sampler*) smpl);
}

//
// Library-level information
//

const char* llama_version_str(void) {
    return llama_version();
}

long long llama_time_us_val(void) {
    return (long long) llama_time_us();
}

int apply_chat_template(void* state_ptr, const char* tmpl,
                        const char** roles, const char** contents, int n_msg,
                        bool add_assistant, char* buf, int buf_size) {
    if (n_msg < 0 || (n_msg > 0 && (roles == nullptr || contents == nullptr))) {
        return -1;
    }

    // An empty tmpl means "use the template baked into the model".
    std::string tmpl_owned;
    if (tmpl == nullptr || tmpl[0] == '\0') {
        llama_binding_state* state = (llama_binding_state*) state_ptr;
        if (state == nullptr) {
            return -1;
        }
        const char* model_tmpl = llama_model_chat_template(state->model, nullptr);
        if (model_tmpl == nullptr) {
            return -1;  // the model carries no chat template
        }
        tmpl_owned = model_tmpl;
    } else {
        tmpl_owned = tmpl;
    }

    std::vector<llama_chat_message> messages;
    messages.reserve((size_t) n_msg);
    for (int i = 0; i < n_msg; i++) {
        if (roles[i] == nullptr || contents[i] == nullptr) {
            return -1;
        }
        messages.push_back({ roles[i], contents[i] });
    }

    // llama_chat_apply_template returns the full length even when it exceeds
    // the buffer, so the caller can size a retry exactly.
    return llama_chat_apply_template(tmpl_owned.c_str(), messages.data(), messages.size(),
                                     add_assistant, buf, buf_size);
}

int chat_builtin_template_count(void) {
    return llama_chat_builtin_templates(nullptr, 0);
}

int chat_builtin_template_name(int i, char* buf, int buf_size) {
    if (i < 0 || buf == nullptr || buf_size <= 0) {
        return -1;
    }
    const int n = llama_chat_builtin_templates(nullptr, 0);
    if (n <= 0 || i >= n) {
        return -1;
    }
    std::vector<const char*> names((size_t) n, nullptr);
    llama_chat_builtin_templates(names.data(), names.size());
    if (names[(size_t) i] == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", names[(size_t) i]);
}

//
// Vocabulary introspection
//

int get_vocab_type(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_vocab_type(llama_model_get_vocab(state->model));
}

// Returns the raw vocabulary entry for a token: the stored piece, before any
// byte-fallback or SentencePiece space decoding. Use token_to_piece_str for
// text that can be concatenated into output.
int get_vocab_token_text(void* state_ptr, int token, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        return -1;
    }
    const char* text = llama_vocab_get_text(vocab, token);
    if (text == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", text);
}

float get_vocab_token_score(void* state_ptr, int token) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        return 0.0f;
    }
    return llama_vocab_get_score(vocab, token);
}

int get_vocab_token_attr(void* state_ptr, int token) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        return 0;  // LLAMA_TOKEN_ATTR_UNDEFINED
    }
    return (int) llama_vocab_get_attr(vocab, token);
}

bool vocab_token_is_eog(void* state_ptr, int token) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        return false;
    }
    return llama_vocab_is_eog(vocab, token);
}

bool vocab_token_is_control(void* state_ptr, int token) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
        return false;
    }
    return llama_vocab_is_control(vocab, token);
}

bool get_vocab_add_sep(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_vocab_get_add_sep(llama_model_get_vocab(state->model));
}

// Copies the vocabulary's suppress list into tokens_out. Returns the number of
// suppressed tokens, or the negative of that count when max_tokens is too
// small, matching the convention used by tokenize_text.
int get_vocab_suppress_tokens(void* state_ptr, int* tokens_out, int max_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    int32_t n = 0;
    const llama_token* toks = llama_vocab_get_suppress_tokens(vocab, &n);
    if (toks == nullptr || n <= 0) {
        return 0;
    }
    if (n > max_tokens) {
        return -n;
    }
    for (int32_t i = 0; i < n; i++) {
        tokens_out[i] = toks[i];
    }
    return n;
}

//
// Further model introspection
//

int get_model_rope_type(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_model_rope_type(state->model);
}

int get_model_ftype(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_model_ftype(state->model);
}

int get_model_decoder_start_token(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_decoder_start_token(state->model);
}

int get_model_n_embd_inp(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_embd_inp(state->model);
}

int get_model_n_embd_out(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_embd_out(state->model);
}

int get_model_n_layer_nextn(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_n_layer_nextn(state->model);
}

int get_model_n_cls_out(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) llama_model_n_cls_out(state->model);
}

int get_model_cls_label(void* state_ptr, int i, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (i < 0 || (uint32_t) i >= llama_model_n_cls_out(state->model)) {
        return -1;
    }
    const char* label = llama_model_cls_label(state->model, (uint32_t) i);
    if (label == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", label);
}

bool model_is_hybrid(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_is_hybrid(state->model);
}

bool model_is_diffusion(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_model_is_diffusion(state->model);
}

// Names for enum values, independent of any loaded model.
int ftype_name(int ftype, char* buf, int buf_size) {
    const char* name = llama_ftype_name((enum llama_ftype) ftype);
    if (name == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", name);
}

int flash_attn_type_name(int type, char* buf, int buf_size) {
    const char* name = llama_flash_attn_type_name((enum llama_flash_attn_type) type);
    if (name == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", name);
}

//
// Guards for the enum values mirrored in llama.go.
//
// The Go layer declares PoolingType, VocabType, TokenAttr and RopeType as
// typed constants so callers get real names instead of bare ints. Those values
// are copies, so if llama.cpp ever renumbers an enum the Go names would keep
// compiling while silently decoding to the wrong thing. These assertions make
// that a build failure here instead.

static_assert(LLAMA_POOLING_TYPE_UNSPECIFIED == -1, "PoolingUnspecified out of sync with llama.go");
static_assert(LLAMA_POOLING_TYPE_NONE        ==  0, "PoolingNone out of sync with llama.go");
static_assert(LLAMA_POOLING_TYPE_MEAN        ==  1, "PoolingMean out of sync with llama.go");
static_assert(LLAMA_POOLING_TYPE_CLS         ==  2, "PoolingCLS out of sync with llama.go");
static_assert(LLAMA_POOLING_TYPE_LAST        ==  3, "PoolingLast out of sync with llama.go");
static_assert(LLAMA_POOLING_TYPE_RANK        ==  4, "PoolingRank out of sync with llama.go");

static_assert(LLAMA_VOCAB_TYPE_NONE   == 0, "VocabNone out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_SPM    == 1, "VocabSPM out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_BPE    == 2, "VocabBPE out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_WPM    == 3, "VocabWPM out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_UGM    == 4, "VocabUGM out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_RWKV   == 5, "VocabRWKV out of sync with llama.go");
static_assert(LLAMA_VOCAB_TYPE_PLAMO2 == 6, "VocabPLaMo2 out of sync with llama.go");

static_assert(LLAMA_TOKEN_ATTR_UNDEFINED    == 0,      "TokenAttrUndefined out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_UNKNOWN      == 1 << 0, "TokenAttrUnknown out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_UNUSED       == 1 << 1, "TokenAttrUnused out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_NORMAL       == 1 << 2, "TokenAttrNormal out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_CONTROL      == 1 << 3, "TokenAttrControl out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_USER_DEFINED == 1 << 4, "TokenAttrUserDefined out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_BYTE         == 1 << 5, "TokenAttrByte out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_NORMALIZED   == 1 << 6, "TokenAttrNormalized out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_LSTRIP       == 1 << 7, "TokenAttrLStrip out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_RSTRIP       == 1 << 8, "TokenAttrRStrip out of sync with llama.go");
static_assert(LLAMA_TOKEN_ATTR_SINGLE_WORD  == 1 << 9, "TokenAttrSingleWord out of sync with llama.go");

static_assert(LLAMA_ROPE_TYPE_NONE   == -1, "RopeNone out of sync with llama.go");
static_assert(LLAMA_ROPE_TYPE_NORM   ==  0, "RopeNorm out of sync with llama.go");
static_assert(LLAMA_ROPE_TYPE_NEOX   ==  2, "RopeNeox out of sync with llama.go");
static_assert(LLAMA_ROPE_TYPE_MROPE  ==  8, "RopeMrope out of sync with llama.go");
static_assert(LLAMA_ROPE_TYPE_VISION == 24, "RopeVision out of sync with llama.go");
static_assert(LLAMA_ROPE_TYPE_IMROPE == 40, "RopeImrope out of sync with llama.go");

//
// State and session persistence
//
// The whole-context helpers (load_state / save_state) round-trip everything.
// These add the two things they cannot express: a session file that carries
// its own token list, and per-sequence state, which is what lets a server
// checkpoint one conversation slot without disturbing the others.
//

long long state_get_size(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (long long) llama_state_get_size(state->ctx);
}

// Serializes the whole context into buf. Returns the number of bytes written,
// or the negative of the required size when buf_size is too small.
long long state_get_data(void* state_ptr, unsigned char* buf, long long buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const size_t need = llama_state_get_size(state->ctx);
    if (buf_size < 0 || (size_t) buf_size < need) {
        return -(long long) need;
    }
    return (long long) llama_state_get_data(state->ctx, buf, (size_t) buf_size);
}

// Restores a context previously serialized by state_get_data. Returns the
// number of bytes consumed, or 0 on failure.
long long state_set_data(void* state_ptr, const unsigned char* buf, long long buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (buf == nullptr || buf_size <= 0) {
        return 0;
    }
    // A malformed or truncated buffer makes llama.cpp throw rather than return;
    // an exception crossing the cgo boundary aborts the process, so translate it
    // into the 0 = failure this already reports.
    try {
        return (long long) llama_state_set_data(state->ctx, buf, (size_t) buf_size);
    } catch (const std::exception &) {
        return 0;
    }
}

// Session files carry the prompt tokens alongside the context state, so a
// process can resume a conversation it did not itself start.
bool state_save_file(void* state_ptr, const char* path, const int* tokens, int n_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (n_tokens < 0) {
        return false;
    }
    // An unwritable path makes llama.cpp throw; keep that from aborting via cgo.
    try {
        return llama_state_save_file(state->ctx, path, (const llama_token*) tokens, (size_t) n_tokens);
    } catch (const std::exception &) {
        return false;
    }
}

// Loads a session file into the context. Returns the number of tokens read,
// or -1 on failure -- including when max_tokens is smaller than the file's
// token count, which the engine rejects outright rather than truncating.
int state_load_file(void* state_ptr, const char* path, int* tokens_out, int max_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (max_tokens < 0) {
        return -1;
    }
    size_t n_out = 0;
    // An unreadable or malformed file makes llama.cpp throw; keep that from
    // crossing cgo and aborting the process.
    try {
        if (!llama_state_load_file(state->ctx, path, (llama_token*) tokens_out,
                                   (size_t) max_tokens, &n_out)) {
            return -1;
        }
    } catch (const std::exception &) {
        return -1;
    }
    return (int) n_out;
}

long long state_seq_get_size(void* state_ptr, int seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (long long) llama_state_seq_get_size(state->ctx, seq_id);
}

// As state_get_data, but for a single sequence.
long long state_seq_get_data(void* state_ptr, unsigned char* buf, long long buf_size, int seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const size_t need = llama_state_seq_get_size(state->ctx, seq_id);
    if (buf_size < 0 || (size_t) buf_size < need) {
        return -(long long) need;
    }
    return (long long) llama_state_seq_get_data(state->ctx, buf, (size_t) buf_size, seq_id);
}

// Restores a sequence into dest_seq_id, which need not be the id it was saved
// from. Returns the number of bytes consumed, or 0 on failure.
long long state_seq_set_data(void* state_ptr, const unsigned char* buf, long long buf_size, int dest_seq_id) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (buf == nullptr || buf_size <= 0) {
        return 0;
    }
    // A malformed sequence buffer makes llama.cpp throw; translate to 0 = failure.
    try {
        return (long long) llama_state_seq_set_data(state->ctx, buf, (size_t) buf_size, dest_seq_id);
    } catch (const std::exception &) {
        return 0;
    }
}

bool state_seq_save_file(void* state_ptr, const char* path, int seq_id, const int* tokens, int n_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (n_tokens < 0) {
        return false;
    }
    try {
        return llama_state_seq_save_file(state->ctx, path, seq_id,
                                         (const llama_token*) tokens, (size_t) n_tokens) > 0;
    } catch (const std::exception &) {
        return false;
    }
}

// Reports how many tokens a per-sequence state file holds, without loading any
// state. Returns -1 if the file cannot be read. There is no equivalent probe
// for whole-session files: llama_state_load_file has no such mode, so callers
// must size that buffer from the context.
int state_seq_file_token_count(void* state_ptr, const char* path) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    size_t n_out = 0;
    // An unreadable or malformed file makes llama.cpp throw; keep it out of cgo.
    try {
        if (llama_state_seq_load_file(state->ctx, path, 0, nullptr, 0, &n_out) == 0) {
            return -1;
        }
    } catch (const std::exception &) {
        return -1;
    }
    return (int) n_out;
}

// Loads a per-sequence file into dest_seq_id, which need not be the id it was
// saved from. Returns the number of tokens read, or -1 on failure — including
// when max_tokens is smaller than the file's token count, which the engine
// treats as an error rather than a truncation. Size the buffer with
// state_seq_file_token_count first.
int state_seq_load_file(void* state_ptr, const char* path, int dest_seq_id,
                        int* tokens_out, int max_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (tokens_out == nullptr || max_tokens < 0) {
        return -1;
    }
    size_t n_out = 0;
    // An unreadable or malformed file makes llama.cpp throw; keep it out of cgo.
    try {
        if (llama_state_seq_load_file(state->ctx, path, dest_seq_id, (llama_token*) tokens_out,
                                      (size_t) max_tokens, &n_out) == 0) {
            return -1;
        }
    } catch (const std::exception &) {
        return -1;
    }
    return (int) n_out;
}

//
// Remaining sampler stages and chain introspection
//

// Fill-in-the-middle infill sampler. Meant to run after top-k and top-p: it
// merges same-prefix candidates and prefers an EOG token once the infill is
// complete, which is what stops FIM generation from running past the hole.
void* sampler_init_infill(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    return llama_sampler_init_infill(vocab);
}

// Adaptive-p. Upstream recommends running it with min-p as the only other
// active truncation stage in the chain.
void* sampler_init_adaptive_p(float target, float decay, unsigned int seed) {
    return llama_sampler_init_adaptive_p(target, decay, seed);
}

// Logit bias. biases is a flat array of n_bias (token, bias) pairs, kept flat
// so cgo does not have to mirror llama_logit_bias.
void* sampler_init_logit_bias(void* state_ptr, int n_bias, const int* tokens, const float* biases) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (n_bias < 0 || (n_bias > 0 && (tokens == nullptr || biases == nullptr))) {
        return nullptr;
    }
    std::vector<llama_logit_bias> lb;
    lb.reserve((size_t) n_bias);
    for (int i = 0; i < n_bias; i++) {
        lb.push_back({ tokens[i], biases[i] });
    }
    return llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), n_bias, lb.data());
}

// Lazy grammar: stays inactive until one of the trigger patterns or trigger
// tokens is seen, then constrains the rest of the output. This is how tool-call
// grammars are applied only from the point the model starts emitting a call.
void* sampler_init_grammar_lazy(void* state_ptr, const char* grammar, const char* root,
                                const char** trigger_patterns, int n_patterns,
                                const int* trigger_tokens, int n_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const llama_vocab* vocab = llama_model_get_vocab(state->model);
    if (n_patterns < 0 || n_tokens < 0) {
        return nullptr;
    }
    return llama_sampler_init_grammar_lazy_patterns(
        vocab, grammar, root,
        trigger_patterns, (size_t) n_patterns,
        (const llama_token*) trigger_tokens, (size_t) n_tokens);
}

// Chain introspection. sampler_chain_get borrows a stage (the chain keeps
// ownership); sampler_chain_remove detaches one and hands ownership to the
// caller, who must free it.
int sampler_chain_n(void* chain) {
    if (chain == nullptr) {
        return 0;
    }
    return llama_sampler_chain_n((const llama_sampler*) chain);
}

void* sampler_chain_get(void* chain, int i) {
    if (chain == nullptr) {
        return nullptr;
    }
    return llama_sampler_chain_get((llama_sampler*) chain, i);
}

void* sampler_chain_remove(void* chain, int i) {
    if (chain == nullptr) {
        return nullptr;
    }
    return llama_sampler_chain_remove((llama_sampler*) chain, i);
}

int sampler_name(void* smpl, char* buf, int buf_size) {
    if (smpl == nullptr) {
        return -1;
    }
    const char* name = llama_sampler_name((const llama_sampler*) smpl);
    if (name == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", name);
}

void* sampler_clone(void* smpl) {
    if (smpl == nullptr) {
        return nullptr;
    }
    return llama_sampler_clone((const llama_sampler*) smpl);
}

unsigned int sampler_get_seed(void* smpl) {
    if (smpl == nullptr) {
        return LLAMA_DEFAULT_SEED;
    }
    return llama_sampler_get_seed((const llama_sampler*) smpl);
}

static_assert(LLAMA_DEFAULT_SEED == 0xFFFFFFFF, "DefaultSeed out of sync with llama.go");

//
// Backend sampling (llama.cpp v0.3.0, [EXPERIMENTAL] upstream)
//
// Normally sampling happens on the CPU: decode produces logits, they are
// copied back to host memory, and a sampler chain picks a token. Attaching a
// chain to a sequence lets the backend sample as part of the graph, so the
// full vocabulary of logits never crosses the device boundary.
//
// The caller keeps ownership of the chain and must keep it alive for as long
// as it is attached to the context.
//

bool set_sequence_sampler(void* state_ptr, int seq_id, void* chain) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_set_sampler(state->ctx, seq_id, (llama_sampler*) chain);
}

// Returns the token the backend sampled for the i-th output, or -1
// (LLAMA_TOKEN_NULL) when nothing was sampled for it.
int get_sampled_token(void* state_ptr, int i) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return llama_get_sampled_token_ith(state->ctx, i);
}

// The three accessors below share a shape: passing out = NULL returns the
// number of available values so the caller can size a buffer, and passing a
// buffer copies min(count, out_size) values and returns how many were written.
// Returns 0 when the backend sampled nothing for this index.

int get_sampled_probs(void* state_ptr, int i, float* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const int n = (int) llama_get_sampled_probs_count_ith(state->ctx, i);
    if (out == nullptr || n <= 0) {
        return n > 0 ? n : 0;
    }
    const float* src = llama_get_sampled_probs_ith(state->ctx, i);
    if (src == nullptr) {
        return 0;
    }
    const int n_copy = n < out_size ? n : out_size;
    for (int j = 0; j < n_copy; j++) {
        out[j] = src[j];
    }
    return n_copy;
}

int get_sampled_logits(void* state_ptr, int i, float* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const int n = (int) llama_get_sampled_logits_count_ith(state->ctx, i);
    if (out == nullptr || n <= 0) {
        return n > 0 ? n : 0;
    }
    const float* src = llama_get_sampled_logits_ith(state->ctx, i);
    if (src == nullptr) {
        return 0;
    }
    const int n_copy = n < out_size ? n : out_size;
    for (int j = 0; j < n_copy; j++) {
        out[j] = src[j];
    }
    return n_copy;
}

// Candidate token ids, which is what maps a probs/logits index back to a
// vocabulary token. Its count matches the probs count.
int get_sampled_candidates(void* state_ptr, int i, int* out, int out_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const int n = (int) llama_get_sampled_candidates_count_ith(state->ctx, i);
    if (out == nullptr || n <= 0) {
        return n > 0 ? n : 0;
    }
    const llama_token* src = llama_get_sampled_candidates_ith(state->ctx, i);
    if (src == nullptr) {
        return 0;
    }
    const int n_copy = n < out_size ? n : out_size;
    for (int j = 0; j < n_copy; j++) {
        out[j] = src[j];
    }
    return n_copy;
}

//
// LoRA adapter introspection and control vectors
//
// Adapters are addressed by their index in the set applied through
// apply_lora_adapter, in the order they were applied. Every function here
// returns -1 for an index outside that set.
//

int lora_adapter_count(void* state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (int) state->lora_adapters.size();
}

// Returns the adapter at index i, or nullptr when i is out of range.
static llama_adapter_lora* lora_at(llama_binding_state* state, int i) {
    if (state == nullptr || i < 0 || (size_t) i >= state->lora_adapters.size()) {
        return nullptr;
    }
    return state->lora_adapters[(size_t) i];
}

int lora_adapter_meta_count(void* state_ptr, int i) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora* a = lora_at(state, i);
    if (a == nullptr) {
        return -1;
    }
    return llama_adapter_meta_count(a);
}

// The three metadata accessors follow snprintf semantics and return -1 when
// the adapter index, key or entry index is absent.
int lora_adapter_meta_val_str(void* state_ptr, int i, const char* key, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora* a = lora_at(state, i);
    if (a == nullptr || buf_size <= 0) {
        return -1;
    }
    return llama_adapter_meta_val_str(a, key, buf, (size_t) buf_size);
}

int lora_adapter_meta_key_by_index(void* state_ptr, int i, int j, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora* a = lora_at(state, i);
    if (a == nullptr || buf_size <= 0) {
        return -1;
    }
    return llama_adapter_meta_key_by_index(a, j, buf, (size_t) buf_size);
}

int lora_adapter_meta_val_str_by_index(void* state_ptr, int i, int j, char* buf, int buf_size) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora* a = lora_at(state, i);
    if (a == nullptr || buf_size <= 0) {
        return -1;
    }
    return llama_adapter_meta_val_str_by_index(a, j, buf, (size_t) buf_size);
}

// Activated LoRA: the adapter only takes effect once the model has emitted its
// invocation tokens. Returns the token count, or the negative of it when
// max_tokens is too small, or -1 for an out-of-range adapter. A plain (non
// activated) LoRA reports 0.
int lora_adapter_alora_tokens(void* state_ptr, int i, int* tokens_out, int max_tokens) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_adapter_lora* a = lora_at(state, i);
    if (a == nullptr) {
        return -1;
    }
    const int n = (int) llama_adapter_get_alora_n_invocation_tokens(a);
    if (n <= 0) {
        return 0;
    }
    if (tokens_out == nullptr || n > max_tokens) {
        return -n;
    }
    const llama_token* src = llama_adapter_get_alora_invocation_tokens(a);
    if (src == nullptr) {
        return 0;
    }
    for (int j = 0; j < n; j++) {
        tokens_out[j] = src[j];
    }
    return n;
}

// Control vector ("steering vector"): a direction added to the residual stream
// of layers [il_start, il_end] to push generation toward or away from some
// behaviour. data is n_embd x n_layers, starting from layer 1. Passing data =
// NULL clears the active vector.
int set_control_vector(void* state_ptr, const float* data, int len, int n_embd, int il_start, int il_end) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (data == nullptr || len <= 0) {
        return llama_set_adapter_cvec(state->ctx, nullptr, 0, 0, 0, 0);
    }
    if (n_embd <= 0 || len % n_embd != 0) {
        return -1;
    }
    return llama_set_adapter_cvec(state->ctx, data, (size_t) len, n_embd, il_start, il_end);
}

//
// Log routing
//
// llama.cpp writes everything to stderr unless a callback is installed. The
// bridge below forwards each record to Go, which dispatches it to whatever the
// caller registered.
//
// The engine's logger state is global and, as llama.h notes, not thread safe,
// so installing and clearing is serialized on the Go side.

extern "C" void goLogCallback(int level, char* text);

static void binding_log_callback(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (text == nullptr) {
        return;
    }
    // cgo generates a non-const char* signature for exported Go functions, and
    // the Go side only reads the string, so dropping the qualifier is safe.
    goLogCallback((int) level, const_cast<char*>(text));
}

void set_log_callback(bool enable) {
    if (enable) {
        llama_log_set(binding_log_callback, nullptr);
    } else {
        // NULL restores llama.cpp's own stderr logging.
        llama_log_set(nullptr, nullptr);
    }
}

// Reports whether the engine is currently routing through this binding's
// bridge. Comparing against binding_log_callback is what makes the answer
// meaningful: llama_log_set(nullptr) does not clear the callback, it installs
// llama.cpp's own stderr default, so llama_log_get never returns null and a
// plain non-null test would always be true.
bool has_log_callback(void) {
    ggml_log_callback cb = nullptr;
    void* user_data = nullptr;
    llama_log_get(&cb, &user_data);
    return cb == binding_log_callback;
}

static_assert(GGML_LOG_LEVEL_NONE  == 0, "LogLevelNone out of sync with llama.go");
static_assert(GGML_LOG_LEVEL_DEBUG == 1, "LogLevelDebug out of sync with llama.go");
static_assert(GGML_LOG_LEVEL_INFO  == 2, "LogLevelInfo out of sync with llama.go");
static_assert(GGML_LOG_LEVEL_WARN  == 3, "LogLevelWarn out of sync with llama.go");
static_assert(GGML_LOG_LEVEL_ERROR == 4, "LogLevelError out of sync with llama.go");
static_assert(GGML_LOG_LEVEL_CONT  == 5, "LogLevelCont out of sync with llama.go");

//
// Model file utilities
//
// These operate on model files rather than on a loaded model, so most take no
// binding state.
//

// Quantizes a GGUF file. ftype is a llama_ftype value; nthread <= 0 lets the
// engine pick. Returns 0 on success, non-zero on failure.
int quantize_model(const char* fname_in, const char* fname_out, int ftype, int nthread,
                   bool allow_requantize, bool quantize_output_tensor,
                   bool pure, bool keep_split) {
    if (fname_in == nullptr || fname_out == nullptr) {
        return 1;
    }
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.ftype                  = (enum llama_ftype) ftype;
    params.nthread                = nthread;
    params.allow_requantize       = allow_requantize;
    params.quantize_output_tensor = quantize_output_tensor;
    params.pure                   = pure;
    params.keep_split             = keep_split;
    return (int) llama_model_quantize(fname_in, fname_out, &params);
}

// Reports the size a quantization would produce, without writing anything.
// Returns 0 on success.
int quantize_model_dry_run(const char* fname_in, int ftype, int nthread) {
    if (fname_in == nullptr) {
        return 1;
    }
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.ftype   = (enum llama_ftype) ftype;
    params.nthread = nthread;
    params.dry_run = true;
    return (int) llama_model_quantize(fname_in, fname_in, &params);
}

void save_model_to_file(void* state_ptr, const char* path) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_model_save_to_file(state->model, path);
}

// Sharded ("split") GGUF paths. split_path builds the path of one shard from a
// prefix; split_prefix recovers the prefix from a shard path. Both follow
// snprintf-style semantics and return 0 when the input does not match the
// expected naming scheme.
int build_split_path(char* buf, int buf_size, const char* prefix, int split_no, int split_count) {
    if (buf == nullptr || buf_size <= 0) {
        return 0;
    }
    return llama_split_path(buf, (size_t) buf_size, prefix, split_no, split_count);
}

int build_split_prefix(char* buf, int buf_size, const char* split_path, int split_no, int split_count) {
    if (buf == nullptr || buf_size <= 0) {
        return 0;
    }
    return llama_split_prefix(buf, (size_t) buf_size, split_path, split_no, split_count);
}

// Model load modes (mmap, mlock, direct I/O). Names round-trip through
// load_mode_from_str.
//
// The engine throws std::invalid_argument for an unrecognised name. A C++
// exception crossing into cgo aborts the process, so it is caught here and
// reported as LLAMA_LOAD_MODE_AUTO, which is the engine's own "decide for me"
// value.
int load_mode_from_str(const char* str) {
    if (str == nullptr) {
        return -1;  // LLAMA_LOAD_MODE_AUTO
    }
    try {
        return (int) llama_load_mode_from_str(str);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        return -1;
    }
}

int load_mode_name(int mode, char* buf, int buf_size) {
    const char* name = llama_load_mode_name((enum llama_load_mode) mode);
    if (name == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", name);
}

// The name llama.cpp uses in GGUF for a well-known metadata key.
int model_meta_key_str(int key, char* buf, int buf_size) {
    const char* name = llama_model_meta_key_str((enum llama_model_meta_key) key);
    if (name == nullptr) {
        return -1;
    }
    return snprintf(buf, (size_t) buf_size, "%s", name);
}

int max_tensor_buft_overrides(void) {
    return (int) llama_max_tensor_buft_overrides();
}

void backend_free(void) {
    llama_backend_free();
}

//
// Abort callback
//
// Lets a caller stop a decode that is already running on the backend, which is
// otherwise uninterruptible. The engine polls this between graph nodes.

extern "C" unsigned char goAbortCallback(void* state_ptr);

static bool binding_abort_callback(void* data) {
    return goAbortCallback(data) != 0;
}

void set_abort_callback(void* state_ptr, bool enable) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (enable) {
        llama_set_abort_callback(state->ctx, binding_abort_callback, state_ptr);
    } else {
        llama_set_abort_callback(state->ctx, nullptr, nullptr);
    }
}

static_assert(LLAMA_LOAD_MODE_AUTO       == -1, "LoadModeAuto out of sync with llama.go");
static_assert(LLAMA_LOAD_MODE_NONE       ==  0, "LoadModeNone out of sync with llama.go");
static_assert(LLAMA_LOAD_MODE_MMAP       ==  1, "LoadModeMmap out of sync with llama.go");
static_assert(LLAMA_LOAD_MODE_MLOCK      ==  2, "LoadModeMlock out of sync with llama.go");
static_assert(LLAMA_LOAD_MODE_MMAP_MLOCK ==  3, "LoadModeMmapMlock out of sync with llama.go");
static_assert(LLAMA_LOAD_MODE_DIRECT_IO  ==  4, "LoadModeDirectIO out of sync with llama.go");

//
// Sequence state with flags
//
// The plain state_seq_* functions above capture a whole sequence. These take a
// llama_state_seq_flags mask, which lets a caller capture only part of it —
// LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, for instance, saves just the
// sliding-window part of an SWA cache, which is far smaller than the whole
// sequence and is all that is needed to resume from the current position.
//

long long state_seq_get_size_ext(void* state_ptr, int seq_id, unsigned int flags) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    return (long long) llama_state_seq_get_size_ext(state->ctx, seq_id, flags);
}

long long state_seq_get_data_ext(void* state_ptr, unsigned char* buf, long long buf_size,
                                 int seq_id, unsigned int flags) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    const size_t need = llama_state_seq_get_size_ext(state->ctx, seq_id, flags);
    if (buf_size < 0 || (size_t) buf_size < need) {
        return -(long long) need;
    }
    return (long long) llama_state_seq_get_data_ext(state->ctx, buf, (size_t) buf_size, seq_id, flags);
}

long long state_seq_set_data_ext(void* state_ptr, const unsigned char* buf, long long buf_size,
                                 int dest_seq_id, unsigned int flags) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    if (buf == nullptr || buf_size <= 0) {
        return 0;
    }
    // With SeqStateOnDevice, llama.cpp validates the header before its own
    // try block and throws on a bad magic; without this guard that exception
    // crosses cgo and aborts. Translate it into 0 = failure, matching the
    // flags=0 path. (A seq-id mismatch trips a GGML_ASSERT upstream, which
    // abort()s and cannot be caught here.)
    try {
        return (long long) llama_state_seq_set_data_ext(state->ctx, buf, (size_t) buf_size, dest_seq_id, flags);
    } catch (const std::exception &) {
        return 0;
    }
}

static_assert(LLAMA_STATE_SEQ_FLAGS_NONE         == 0, "SeqStateAll out of sync with llama.go");
static_assert(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY == 1, "SeqStatePartialOnly out of sync with llama.go");
static_assert(LLAMA_STATE_SEQ_FLAGS_ON_DEVICE    == 2, "SeqStateOnDevice out of sync with llama.go");
