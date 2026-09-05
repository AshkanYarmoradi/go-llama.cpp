#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

// Log routing. set_log_callback(true) installs a bridge that forwards every
// llama.cpp log record to goLogCallback; set_log_callback(false) restores the
// engine's own stderr output. The engine's logger state is global and not
// thread safe, so the Go layer serializes these calls.
extern void goLogCallback(int level, char* text);
void set_log_callback(bool enable);
bool has_log_callback(void);

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


// Loads a model from an explicit list of shards. Only needed when the shard
// filenames do not follow llama.cpp's own naming scheme; otherwise load_model
// with the first shard is enough.
void* load_model_splits(const char **paths, int n_paths,
                        int n_ctx, int n_seed, bool memory_f16, bool mlock,
                        bool embeddings, bool mmap, bool low_vram, int n_gpu_layers, int n_batch,
                        const char *maingpu, const char *tensorsplit, bool numa, float rope_freq_base,
                        float rope_freq_scale, const char *lora, const char *lora_base);

// Sequence state with a llama_state_seq_flags mask, which lets a caller
// capture part of a sequence rather than all of it. Same buffer contract as
// the plain state_seq_* functions.
long long state_seq_get_size_ext(void* state_ptr, int seq_id, unsigned int flags);
long long state_seq_get_data_ext(void* state_ptr, unsigned char* buf, long long buf_size,
                                 int seq_id, unsigned int flags);
long long state_seq_set_data_ext(void* state_ptr, const unsigned char* buf, long long buf_size,
                                 int dest_seq_id, unsigned int flags);
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

// LoRA adapter introspection and control vectors. Adapters are addressed by
// their index in the set applied through apply_lora_adapter, in application
// order; every accessor returns -1 for an index outside that set. The metadata
// functions follow snprintf semantics. lora_adapter_alora_tokens returns the
// invocation-token count (0 for a plain LoRA), or the negative of it when
// max_tokens is too small. set_control_vector takes an n_embd x n_layers
// buffer starting from layer 1, or data = NULL to clear the active vector.
int lora_adapter_count(void* state_ptr);
int lora_adapter_meta_count(void* state_ptr, int i);
int lora_adapter_meta_val_str(void* state_ptr, int i, const char* key, char* buf, int buf_size);
int lora_adapter_meta_key_by_index(void* state_ptr, int i, int j, char* buf, int buf_size);
int lora_adapter_meta_val_str_by_index(void* state_ptr, int i, int j, char* buf, int buf_size);
int lora_adapter_alora_tokens(void* state_ptr, int i, int* tokens_out, int max_tokens);
int set_control_vector(void* state_ptr, const float* data, int len, int n_embd, int il_start, int il_end);

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

// Backend sampling (llama.cpp v0.3.0, [EXPERIMENTAL] upstream). Attaching a
// sampler chain to a sequence lets the backend sample inside the graph, so the
// full vocabulary of logits never crosses the device boundary. The caller
// keeps ownership of the chain and must keep it alive while it is attached.
//
// The three array accessors take out = NULL to report the available count,
// or a buffer to copy into, returning how many values were written. All
// return 0 when the backend sampled nothing for that index.
bool set_sequence_sampler(void* state_ptr, int seq_id, void* chain);
int get_sampled_token(void* state_ptr, int i);
int get_sampled_probs(void* state_ptr, int i, float* out, int out_size);
int get_sampled_logits(void* state_ptr, int i, float* out, int out_size);
int get_sampled_candidates(void* state_ptr, int i, int* out, int out_size);


// State and session persistence. The load_state / save_state pair above
// round-trips a whole context as raw bytes. These add what it cannot express:
// a session file that carries its own token list, and per-sequence state, so a
// server can checkpoint one conversation slot without touching the others.
//
// The state_*_get_data functions return the bytes written, or the negative of
// the required size when the buffer is too small. The file loaders return the
// token count, or -1 on failure -- a buffer smaller than the file's token
// count is a failure, not a truncation, so probe the size first.
long long state_get_size(void* state_ptr);
long long state_get_data(void* state_ptr, unsigned char* buf, long long buf_size);
long long state_set_data(void* state_ptr, const unsigned char* buf, long long buf_size);
bool state_save_file(void* state_ptr, const char* path, const int* tokens, int n_tokens);
int state_load_file(void* state_ptr, const char* path, int* tokens_out, int max_tokens);
long long state_seq_get_size(void* state_ptr, int seq_id);
long long state_seq_get_data(void* state_ptr, unsigned char* buf, long long buf_size, int seq_id);
long long state_seq_set_data(void* state_ptr, const unsigned char* buf, long long buf_size, int dest_seq_id);
bool state_seq_save_file(void* state_ptr, const char* path, int seq_id, const int* tokens, int n_tokens);
int state_seq_file_token_count(void* state_ptr, const char* path);
int state_seq_load_file(void* state_ptr, const char* path, int dest_seq_id,
                        int* tokens_out, int max_tokens);

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

// Chat templates. apply_chat_template formats roles[i]/contents[i] (parallel
// arrays of n_msg messages) with the given Jinja template; pass an empty tmpl
// to use the template baked into the model. It returns the full byte length of
// the result following snprintf semantics -- a value >= buf_size means the
// output was truncated and the caller should retry with that size -- or a
// negative value if the template is missing or unsupported.
// chat_builtin_template_* enumerate the templates llama.cpp recognises by name.
int apply_chat_template(void* state_ptr, const char* tmpl,
                        const char** roles, const char** contents, int n_msg,
                        bool add_assistant, char* buf, int buf_size);
int chat_builtin_template_count(void);
int chat_builtin_template_name(int i, char* buf, int buf_size);

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


// Vocabulary introspection. The *_str-style functions follow snprintf
// semantics (see above) and return -1 for an out-of-range token.
// get_vocab_token_text returns the raw stored vocabulary entry, which is not
// the same as printable text -- use token_to_piece_str for output.
// get_vocab_token_attr returns a bitmask of llama_token_attr values.
int get_vocab_type(void* state_ptr);
int get_vocab_token_text(void* state_ptr, int token, char* buf, int buf_size);
float get_vocab_token_score(void* state_ptr, int token);
int get_vocab_token_attr(void* state_ptr, int token);
bool vocab_token_is_eog(void* state_ptr, int token);
bool vocab_token_is_control(void* state_ptr, int token);
bool get_vocab_add_sep(void* state_ptr);
int get_vocab_suppress_tokens(void* state_ptr, int* tokens_out, int max_tokens);

// Further model introspection. get_model_cls_label names the i-th output of a
// classifier head; get_model_decoder_start_token returns -1 when the model is
// not an encoder-decoder. ftype_name and flash_attn_type_name translate enum
// values to strings and need no loaded model.
int get_model_rope_type(void* state_ptr);
int get_model_ftype(void* state_ptr);
int get_model_decoder_start_token(void* state_ptr);
int get_model_n_embd_inp(void* state_ptr);
int get_model_n_embd_out(void* state_ptr);
int get_model_n_layer_nextn(void* state_ptr);
int get_model_n_cls_out(void* state_ptr);
int get_model_cls_label(void* state_ptr, int i, char* buf, int buf_size);
bool model_is_hybrid(void* state_ptr);
bool model_is_diffusion(void* state_ptr);
int ftype_name(int ftype, char* buf, int buf_size);
int flash_attn_type_name(int type, char* buf, int buf_size);

// Model architecture queries
bool model_has_encoder(void* state_ptr);
bool model_has_decoder(void* state_ptr);
bool model_is_recurrent(void* state_ptr);

// System info
int get_system_info(char* buf, int buf_size);

// Model file utilities. quantize_model writes a requantized copy and returns 0
// on success; quantize_model_dry_run reports the resulting size without
// writing. The split helpers translate between a sharded-GGUF prefix and one
// shard's path, returning 0 when the input does not match the naming scheme.
// The *_name functions follow snprintf semantics.
int quantize_model(const char* fname_in, const char* fname_out, int ftype, int nthread,
                   bool allow_requantize, bool quantize_output_tensor,
                   bool pure, bool keep_split);
int quantize_model_dry_run(const char* fname_in, int ftype, int nthread);
void save_model_to_file(void* state_ptr, const char* path);
int build_split_path(char* buf, int buf_size, const char* prefix, int split_no, int split_count);
int build_split_prefix(char* buf, int buf_size, const char* split_path, int split_no, int split_count);
int load_mode_from_str(const char* str);
int load_mode_name(int mode, char* buf, int buf_size);
int model_meta_key_str(int key, char* buf, int buf_size);
int max_tensor_buft_overrides(void);
void backend_free(void);

// Abort callback. set_abort_callback(state, true) makes the engine poll
// goAbortCallback between graph nodes so a running decode can be stopped;
// passing false detaches it.
extern unsigned char goAbortCallback(void* state_ptr);
void set_abort_callback(void* state_ptr, bool enable);

// Backend capability queries. These reflect how the llama.cpp library was
// compiled and require no loaded model.
bool backend_supports_mmap(void);
bool backend_supports_mlock(void);
bool backend_supports_gpu_offload(void);
bool backend_supports_rpc(void);
int backend_max_devices(void);
int backend_max_parallel_sequences(void);


// Context runtime introspection and control. The engine may clamp or round the
// values requested via load_model, so these report what the context actually
// uses. context_pooling_type returns a llama_pooling_type enum value.
int context_n_ctx(void* state_ptr);
int context_n_ctx_seq(void* state_ptr);
int context_n_batch(void* state_ptr);
int context_n_ubatch(void* state_ptr);
int context_n_seq_max(void* state_ptr);
int context_n_rs_seq(void* state_ptr);
int context_pooling_type(void* state_ptr);
int context_n_threads(void* state_ptr);
int context_n_threads_batch(void* state_ptr);
void context_set_n_threads(void* state_ptr, int n_threads, int n_threads_batch);
void context_set_embeddings(void* state_ptr, bool embeddings);
void context_set_causal_attn(void* state_ptr, bool causal_attn);
void context_synchronize(void* state_ptr);

// Further KV-cache operations. memory_seq_add shifts, and memory_seq_div
// divides, the positions of a sequence in [p0, p1); negative p0/p1 mean "from
// the start" / "to the end". The pos accessors return -1 for an empty sequence.
void memory_seq_add(void* state_ptr, int seq_id, int p0, int p1, int delta);
void memory_seq_div(void* state_ptr, int seq_id, int p0, int p1, int d);
int memory_seq_pos_min(void* state_ptr, int seq_id);
int memory_seq_pos_max(void* state_ptr, int seq_id);
bool memory_can_shift(void* state_ptr);

// Performance counters. Every out-parameter is optional (pass NULL to skip).
// perf_sampler only reports data for samplers built with sampler_chain_init.
void perf_context(void* state_ptr, double* t_start_ms, double* t_load_ms,
                  double* t_p_eval_ms, double* t_eval_ms,
                  int* n_p_eval, int* n_eval, int* n_reused);
void perf_context_reset(void* state_ptr);
void perf_sampler(void* smpl, double* t_sample_ms, int* n_sample);
void perf_sampler_reset(void* smpl);

// Library-level information, available without a loaded model.
const char* llama_version_str(void);
long long llama_time_us_val(void);

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
void* sampler_init_penalties(void* state_ptr, int last_n, float repeat, float freq, float present);
void* sampler_init_grammar(void* state_ptr, const char* grammar, const char* root);
void* sampler_init_dry(void* state_ptr, float multiplier, float base, int allowed_length, int penalty_last_n);

// Remaining sampler stages. sampler_init_logit_bias takes the (token, bias)
// pairs as two parallel arrays. sampler_init_grammar_lazy builds a grammar
// that stays inactive until a trigger pattern or token appears. Every
// sampler_init_* returns NULL on invalid input, which sampler_chain_add
// ignores.
void* sampler_init_infill(void* state_ptr);
void* sampler_init_adaptive_p(float target, float decay, unsigned int seed);
void* sampler_init_logit_bias(void* state_ptr, int n_bias, const int* tokens, const float* biases);
void* sampler_init_grammar_lazy(void* state_ptr, const char* grammar, const char* root,
                                const char** trigger_patterns, int n_patterns,
                                const int* trigger_tokens, int n_tokens);

// Chain introspection. sampler_chain_get borrows a stage -- the chain keeps
// ownership -- while sampler_chain_remove detaches one and transfers ownership
// to the caller, who must free it. sampler_clone also returns an owned sampler.
int sampler_chain_n(void* chain);
void* sampler_chain_get(void* chain, int i);
void* sampler_chain_remove(void* chain, int i);
int sampler_name(void* smpl, char* buf, int buf_size);
void* sampler_clone(void* smpl);
unsigned int sampler_get_seed(void* smpl);

#ifdef __cplusplus
}

std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
