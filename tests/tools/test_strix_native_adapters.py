"""Compile and execute the production native loops against public API doubles."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMMON = r'''
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <set>
inline void check(bool ok) { if (!ok) throw std::runtime_error("public API contract violation"); }
'''
VLLM = COMMON + r'''
struct vllm_engine {};
enum { VLLM_OK = 0 };
struct vllm_model_params {
 const char* model_path=nullptr; int max_model_len=0,max_num_seqs=0,enable_prefix_caching=0;
 const char* kv_cache_dtype=nullptr; int block_size=0,num_blocks=0;
};
struct vllm_sampling_params { float temperature=1,top_p=1; int max_tokens=0,min_tokens=0,ignore_eos=0; };
struct vllm_completion { int completion_tokens=0,prompt_tokens=0; const char* finish_reason=nullptr; };
inline vllm_model_params vllm_model_params_default() { return {}; }
inline vllm_sampling_params vllm_sampling_params_default() { return {}; }
inline const char* vllm_last_error() { return "fake load error"; }
inline int vllm_engine_load(const vllm_model_params* p, vllm_engine** e) {
 check(p->max_model_len==2048 && p->max_num_seqs==4 && p->enable_prefix_caching==2 &&
 std::string(p->kv_cache_dtype)=="bfloat16" && p->block_size==32 && p->num_blocks==256);
 *e=new vllm_engine; return 0;
}
inline std::atomic<int> peak{0};
inline void vllm_engine_free(vllm_engine* e) { check(peak==std::atoi(std::getenv("CONCURRENCY"))); delete e; }
inline void vllm_completion_free(vllm_completion*) {}
inline std::atomic<int> active{0};
inline int vllm_complete_tokens(vllm_engine*,const int32_t* p,int32_t n,const vllm_sampling_params* s,
 int32_t* out,int32_t cap,int32_t* count,vllm_completion* completion) {
 check(n==2 && cap>=128 && s->temperature==0 && s->top_p==1 && s->max_tokens==128 && s->min_tokens==0 && s->ignore_eos==1);
 int c=++active; check(c<=std::atoi(std::getenv("CONCURRENCY")));
 int old=peak.load();while(old<c && !peak.compare_exchange_weak(old,c)){}
 std::this_thread::sleep_for(std::chrono::milliseconds(2));
 for(int i=0;i<128;++i) out[i]=i==1?151645:p[0]+i;
 *count=128; completion->completion_tokens=128; completion->prompt_tokens=n; completion->finish_reason="length";
 --active; return 0;
}
'''
LLAMA = COMMON + r'''
using llama_token=int32_t; using llama_seq_id=int32_t; using llama_pos=int32_t;
enum {GGML_TYPE_BF16=30,LLAMA_FLASH_ATTN_TYPE_AUTO=-1};
struct llama_vocab {};
struct llama_model {};
struct llama_model_params {int n_gpu_layers=0;};
struct llama_context_params {int n_batch=2048,n_ubatch=512,n_threads=4,n_threads_batch=4;
 bool offload_kqv=true,op_offload=true,kv_unified=false;
 int n_ctx=512,n_seq_max=1,type_k=1,type_v=1,flash_attn_type=-1;};
struct llama_context { std::vector<int> rows, seeds, lengths; int pending=0; bool cleared[4]={false,false,false,false};
 llama_context():seeds(4),lengths(4){} };
using llama_memory_t=llama_context*;
struct llama_sampler {int accepted=0;};
struct llama_batch {int n_tokens=0; int32_t* token; int32_t* pos; int32_t* n_seq_id; int32_t** seq_id; int8_t* logits;};
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline bool llama_supports_gpu_offload() {return true;}
inline llama_model_params llama_model_default_params(){return {};}
inline llama_context_params llama_context_default_params(){return {};}
inline llama_model* llama_model_load_from_file(const char*,llama_model_params p){check(p.n_gpu_layers>=37);return new llama_model;}
inline const llama_vocab* llama_model_get_vocab(llama_model*) { static llama_vocab v; return &v; }
inline int llama_tokenize(const llama_vocab*,const char* text,int,int32_t* out,int,bool special,bool parse) {
 check(!special && parse); out[0]=std::atoi(text);out[1]=11;return 2;
}
inline llama_context* llama_init_from_model(llama_model*,llama_context_params p) {
 check(p.n_ctx==8192 && p.n_seq_max==4 && !p.kv_unified && p.type_k==30 && p.type_v==30 && p.flash_attn_type==-1);
 return new llama_context;
}
inline int llama_n_ctx_seq(llama_context*) {return 2048;}
inline void llama_free(llama_context* c){delete c;}
inline void llama_model_free(llama_model* m){delete m;}
inline llama_memory_t llama_get_memory(llama_context* c){return c;}
inline bool llama_memory_seq_rm(llama_memory_t c,int seq,int p0,int p1){
 check(p0==-1 && p1==-1);c->lengths[seq]=0;c->cleared[seq]=true;return true;
}
inline llama_sampler* llama_sampler_init_greedy(){return new llama_sampler;}
inline void llama_sampler_accept(llama_sampler* s,int32_t){++s->accepted;}
inline void llama_sampler_free(llama_sampler* s){check(s->accepted==128);delete s;}
inline int batch_peak_sequences=0;
inline llama_batch llama_batch_init(int n,int emb,int seq){
 check(n==2048 && emb==0 && seq==1);
 batch_peak_sequences=0;
 llama_batch b; b.token=new int32_t[n]; b.pos=new int32_t[n]; b.n_seq_id=new int32_t[n];
 b.seq_id=new int32_t*[n];b.logits=new int8_t[n];
 for(int i=0;i<n;++i)b.seq_id[i]=new int32_t[1];return b;
}
inline void llama_batch_free(llama_batch b){
 std::fprintf(stderr,"fixture_peak_sequences=%d\n",batch_peak_sequences);
 for(int i=0;i<2048;++i)delete[]b.seq_id[i];
 delete[]b.token;delete[]b.pos;delete[]b.n_seq_id;delete[]b.seq_id;delete[]b.logits;
}
inline int llama_decode(llama_context* c,llama_batch b){
 check(c->pending==0); c->rows.assign(b.n_tokens,-1);
 std::set<int> independent_logits;
 for(int i=0;i<b.n_tokens;++i){
  int seq=b.seq_id[i][0];check(seq>=0 && seq<std::atoi(std::getenv("CONCURRENCY")) && b.n_seq_id[i]==1);
  check(b.pos[i]==c->lengths[seq]);
  if(b.pos[i]==0){check(c->cleared[seq]);c->cleared[seq]=false;c->seeds[seq]=b.token[i];}
  else if(b.pos[i]>=2){int step=b.pos[i]-2;check(b.token[i]==(step==1?151645:c->seeds[seq]+step));}
  ++c->lengths[seq];
  if(b.logits[i]){c->rows[i]=seq;++c->pending;check(independent_logits.insert(seq).second);}
 }
 batch_peak_sequences=std::max(batch_peak_sequences,int(independent_logits.size()));
 return 0;
}
inline llama_token llama_sampler_sample(llama_sampler* s,llama_context* c,int row){
 check(row>=0 && row<int(c->rows.size()) && c->rows[row]>=0);
 int seq=c->rows[row];c->rows[row]=-1;--c->pending;
 int step=s->accepted++;return step==1?151645:c->seeds[seq]+step;
}
'''


class NativeAdaptersTests(unittest.TestCase):
    def test_production_public_api_loops(self):
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler, 'native contract tests require a C++ compiler')
        for engine, header, define in (('vllm.cpp', VLLM, []), ('llama.cpp', LLAMA, ['-DSTRIX_LLAMA'])):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ('llama.h' if define else 'vllm.h')).write_text(header)
                binary = root / 'adapter'
                build = subprocess.run([compiler, '-std=c++20', '-pthread', *define, '-I', str(root),
                    '-I', str(ROOT / 'third_party'), str(ROOT / 'tools/bench/strix_four_engine/native_adapter.cpp'),
                    '-o', str(binary)], capture_output=True, text=True, timeout=60)
                self.assertEqual(build.returncode, 0, build.stderr)
                for c in (1, 4):
                    commands = [dict(schema=1, id=1, command='configure', engine=engine, model='fixture', gguf='fixture',
                        prompts=[str(i + 10) for i in range(6)], prompt_ids=[[i + 10, 11] for i in range(6)])]
                    for rep in range(2):
                        commands.append(dict(schema=1, id=rep + 2, command='run', phase='qualification', concurrency=c, repetition=rep))
                    commands.append(dict(schema=1, id=4, command='shutdown'))
                    run = subprocess.run([str(binary)], input=''.join(json.dumps(x)+'\n' for x in commands),
                        capture_output=True, text=True, timeout=20, env=dict(os.environ, CONCURRENCY=str(c)))
                    self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                    if engine == 'llama.cpp':
                        # Measure production decode batches, not configured slot capacity.
                        peaks = [int(line.split('=')[1]) for line in run.stderr.splitlines()
                                 if line.startswith('fixture_peak_sequences=')]
                        self.assertEqual(peaks, [c, c], run.stderr)
                    results = [json.loads(line) for line in run.stdout.splitlines()]
                    self.assertEqual([r['id'] for r in results], [1, 2, 3, 4])
                    for corpus in results[1:3]:
                        self.assertEqual(len(corpus['requests']), 6)
                        for i, request in enumerate(corpus['requests']):
                            expected = [i + 10 + token for token in range(128)]
                            expected[1] = 151645  # EOS remains eligible and does not stop generation.
                            self.assertEqual(request['output_ids'], expected)
                            self.assertEqual(request['prompt_ids'], [i + 10, 11])
                            self.assertEqual(request['index'], i)


if __name__ == '__main__':
    unittest.main()
