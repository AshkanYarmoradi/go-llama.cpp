#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

int load_state(void *ctx, char *statefile, char*modes);

void save_state(void *ctx, char *dst, char*modes);

void* load_model(const char *fname, 
                 int n_ctx, 
                 int n_seed, 
                 bool memory_f16, 
                 bool mlock, 
                 bool embeddings, 
                 bool mmap, 
                 bool low_vram, 
                 int n_gpu, 
                 int n_batch, 
                 const char *maingpu, 
                 const char *tensorsplit, 
                 bool numa, 
                 float rope_freq_base, 
                 float rope_freq_scale,
                 const char *lora, const char *lora_base
                 );

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings);

int get_token_embeddings(void* params_ptr, void* state_pr, int *tokens, int tokenSize, float * res_embeddings);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float min_p, float temp, float repeat_penalty, 
                            int repeat_last_n, bool ignore_eos, bool memory_f16, 
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, 
                            int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, 
                            const char *logit_bias, const char *session_file, bool prompt_cache_all, 
                            bool mlock, bool mmap, const char *maingpu, const char *tensorsplit, 
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, 
                            float rope_freq_scale, int n_draft,
                            float xtc_probability, float xtc_threshold,
                            float dry_multiplier, float dry_base, int dry_allowed_length, int dry_penalty_last_n,
                            float top_n_sigma);

void llama_free_params(void* params_ptr);

void llama_binding_free_model(void* state);

// LoRA adapters. apply_lora_adapter loads an adapter and adds it to the set
// active on the context (returns 0 on success). clear_lora_adapters detaches
// and frees every adapter previously applied this way.
int apply_lora_adapter(void* state_ptr, const char* path, float scale);
int clear_lora_adapters(void* state_ptr);

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result);

// Direct tokenization helpers that do not require a binding_params struct.
// tokenize_text returns the token count, or the negative of the required count
// when max_tokens is too small (matching llama_tokenize). detokenize_text and
// token_to_piece_str return the bytes written, or the negative of the required
// size when the buffer is too small (matching llama.cpp).
int tokenize_text(void* state_ptr, const char* text, int text_len,
                  int* tokens_out, int max_tokens,
                  bool add_special, bool parse_special);
int detokenize_text(void* state_ptr, const int* tokens, int n_tokens,
                    char* buf, int buf_size,
                    bool remove_special, bool unparse_special);
int token_to_piece_str(void* state_ptr, int token, char* buf, int buf_size, bool special);

int llama_predict(void* params_ptr, void* state_pr, char* result, int result_size, bool debug);

// Low-level batching, decoding, and output access. batch_init allocates an
// opaque batch (free with batch_free); batch_add appends tokens; decode_batch /
// encode_batch run it through the model; get_logits_ith / get_embeddings_ith /
// get_embeddings_seq copy outputs; the memory_* helpers manage the KV cache.
void* batch_init(int n_tokens, int n_seq_max);
void batch_free(void* batch_ptr);
void batch_clear(void* batch_ptr);
int batch_n_tokens(void* batch_ptr);
int batch_add(void* batch_ptr, int token, int pos, const int* seq_ids, int n_seq_ids, bool logits);
int decode_batch(void* state_ptr, void* batch_ptr);
int encode_batch(void* state_ptr, void* batch_ptr);
int get_logits_ith(void* state_ptr, int i, float* out, int out_size);
int get_embeddings_ith(void* state_ptr, int i, float* out, int out_size);
int get_embeddings_seq(void* state_ptr, int seq_id, float* out, int out_size);
void memory_clear(void* state_ptr, bool data);
bool memory_seq_rm(void* state_ptr, int seq_id, int p0, int p1);
void memory_seq_cp(void* state_ptr, int src, int dst, int p0, int p1);
void memory_seq_keep(void* state_ptr, int seq_id);

// Model info functions
int get_model_n_vocab(void* state_ptr);
int get_model_n_ctx_train(void* state_ptr);
int get_model_n_embd(void* state_ptr);
int get_model_n_layer(void* state_ptr);
long long get_model_size(void* state_ptr);
long long get_model_n_params(void* state_ptr);
int get_model_description(void* state_ptr, char* buf, int buf_size);
int get_model_chat_template(void* state_ptr, const char* name, char* buf, int buf_size);

// Extended model geometry
int get_model_n_head(void* state_ptr);
int get_model_n_head_kv(void* state_ptr);
int get_model_n_swa(void* state_ptr);
float get_model_rope_freq_scale_train(void* state_ptr);

// Model metadata (GGUF key-value header). The *_str functions follow snprintf
// semantics: they return the length that would be written (>= buf_size means
// the value was truncated), or -1 if the key/index is absent.
int get_model_meta_count(void* state_ptr);
int get_model_meta_val_str(void* state_ptr, const char* key, char* buf, int buf_size);
int get_model_meta_key_by_index(void* state_ptr, int i, char* buf, int buf_size);
int get_model_meta_val_str_by_index(void* state_ptr, int i, char* buf, int buf_size);

// Chat template application
int apply_chat_template(void* state_ptr, const char* tmpl, const char* messages_json,
                        bool add_generation_prompt, char* result, int result_size);

// Special token functions
int get_vocab_bos(void* state_ptr);
int get_vocab_eos(void* state_ptr);
int get_vocab_eot(void* state_ptr);
int get_vocab_nl(void* state_ptr);
int get_vocab_sep(void* state_ptr);
bool get_vocab_add_bos(void* state_ptr);
bool get_vocab_add_eos(void* state_ptr);

// Extended special tokens (padding, mask, and fill-in-the-middle). Return -1
// (LLAMA_TOKEN_NULL) when the model's vocabulary does not define the token.
int get_vocab_pad(void* state_ptr);
int get_vocab_mask(void* state_ptr);
int get_vocab_fim_pre(void* state_ptr);
int get_vocab_fim_suf(void* state_ptr);
int get_vocab_fim_mid(void* state_ptr);
int get_vocab_fim_pad(void* state_ptr);
int get_vocab_fim_rep(void* state_ptr);
int get_vocab_fim_sep(void* state_ptr);

// Model architecture queries
bool model_has_encoder(void* state_ptr);
bool model_has_decoder(void* state_ptr);
bool model_is_recurrent(void* state_ptr);

// System info
int get_system_info(char* buf, int buf_size);

// Backend capability queries. These reflect how the llama.cpp library was
// compiled and require no loaded model.
bool backend_supports_mmap(void);
bool backend_supports_mlock(void);
bool backend_supports_gpu_offload(void);
bool backend_supports_rpc(void);
int backend_max_devices(void);
int backend_max_parallel_sequences(void);

// Composable samplers. Build a chain with sampler_chain_init, append stages
// created by the sampler_init_* helpers with sampler_chain_add (the chain takes
// ownership), then sampler_sample from a decoded context. sampler_free releases
// a sampler and, for a chain, every stage added to it.
void* sampler_chain_init(void);
void sampler_chain_add(void* chain, void* smpl);
void sampler_free(void* smpl);
void sampler_reset(void* smpl);
void sampler_accept(void* smpl, int token);
int sampler_sample(void* state_ptr, void* smpl, int idx);
void* sampler_init_greedy(void);
void* sampler_init_dist(unsigned int seed);
void* sampler_init_top_k(int k);
void* sampler_init_top_p(float p, int min_keep);
void* sampler_init_min_p(float p, int min_keep);
void* sampler_init_typical(float p, int min_keep);
void* sampler_init_temp(float t);
void* sampler_init_temp_ext(float t, float delta, float exponent);
void* sampler_init_xtc(float p, float t, int min_keep, unsigned int seed);
void* sampler_init_top_n_sigma(float n);
void* sampler_init_mirostat_v2(unsigned int seed, float tau, float eta);
void* sampler_init_penalties(int last_n, float repeat, float freq, float present);
void* sampler_init_grammar(void* state_ptr, const char* grammar, const char* root);
void* sampler_init_dry(void* state_ptr, float multiplier, float base, int allowed_length, int penalty_last_n);

#ifdef __cplusplus
}

std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
