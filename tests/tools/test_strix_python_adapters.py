"""Execute production Python adapters with public-API doubles."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = r'''
import asyncio, os, runpy, sys, types
def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result
    return result
mode = os.environ.get('CASE', '')
class Tokenizer:
    @classmethod
    def from_pretrained(cls, path, **kw):
        assert kw == dict(local_files_only=True, trust_remote_code=False)
        return cls()
    def encode(self, text, **kw):
        assert kw == dict(add_special_tokens=False)
        return [10, 11]
module('transformers', AutoTokenizer=Tokenizer)
module('torch', version=types.SimpleNamespace(hip=None if mode=='hip' else '7.2',
    cuda='12.0' if mode=='cuda' else None), cuda=types.SimpleNamespace(
    is_available=lambda: mode!='unavailable', get_device_properties=lambda _: types.SimpleNamespace(
        gcnArchName='gfx942' if mode=='arch' else 'gfx1151')))
active=0
peak=0
def params(**kw): return types.SimpleNamespace(**kw)
class Args:
    def __init__(self, **kw):
        assert kw['dtype']=='bfloat16' and kw['kv_cache_dtype']=='auto'
        assert kw['max_model_len']==2048 and kw['max_num_seqs']==4 and not kw['enable_prefix_caching']
        assert kw['enforce_eager'] is False and kw['kv_cache_memory_bytes']==1207959552
class VLLM:
    @classmethod
    def from_engine_args(cls, args): return cls()
    def __init__(self):
        self.vllm_config=params(model_config=params(dtype='torch.bfloat16', enforce_eager=False, max_model_len=2048),
            scheduler_config=params(max_num_seqs=4), cache_config=params(enable_prefix_caching=False,
                cache_dtype='auto', kv_cache_memory_bytes=1207959552))
        if mode.startswith('vllm:'):
            section, field = mode[5:].split('.')
            setattr(getattr(self.vllm_config, section), field, json.loads(os.environ['INVALID_VALUE']))
    async def generate(self, prompt, sampling, rid):
        global active, peak
        assert prompt == {'prompt_token_ids':[10,11]}
        assert sampling.temperature==0 and sampling.top_p==1 and sampling.max_tokens==128 and sampling.ignore_eos
        assert sampling.min_tokens==0 and sampling.output_kind=='cumulative'
        active+=1
        peak=max(peak,active)
        assert active <= int(os.environ['CONCURRENCY'])
        await asyncio.sleep(0.001)
        ids=[20]*128;ids[1]=151645
        def output(ids, final):
            return params(request_id=rid, prompt_token_ids=[10,11], finished=final,
                          outputs=[params(token_ids=ids, finish_reason='length' if final else None)])
        yield output(ids[:20],False)
        yield output(ids[:20],False) # Repeated cumulative chunks must not duplicate IDs.
        if mode=='stream': ids[0]=99
        yield output(ids,True)
        active-=1
    def shutdown(self):
        if mode != 'stream': assert peak==int(os.environ['CONCURRENCY'])
module('vllm', SamplingParams=params)
module('vllm.engine');module('vllm.engine.arg_utils', AsyncEngineArgs=Args)
module('vllm.v1');module('vllm.v1.engine');module('vllm.v1.engine.async_llm', AsyncLLM=VLLM)
module('vllm.sampling_params', RequestOutputKind=types.SimpleNamespace(CUMULATIVE='cumulative'))
class SG:
    instance = None
    def __init__(self, **kw):
        SG.instance = self
        assert kw['dtype']=='bfloat16' and kw['kv_cache_dtype']=='bf16'
        assert kw['context_length']==2048 and kw['max_running_requests']==4 and kw['max_total_tokens']==8192
        assert kw['disable_radix_cache'] and kw['attention_backend']=='triton' and not kw['disable_cuda_graph']
        self.info=dict(kw,max_total_num_tokens=8192)
        if mode.startswith('sg:'):
            self.info[mode[3:]] = json.loads(os.environ['INVALID_VALUE'])
        # f63458b5 engine.py:277-281 selects a RUNNING loop, not a merely set loop.
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    def get_server_info(self):
        # engine.py:999 and tokenizer_manager.py:1822 bind the receiver on first use.
        async def internal_state():
            self.receiver_loop = asyncio.get_running_loop()
            return self.info
        return self.loop.run_until_complete(internal_state())
    async def async_generate(self, **kw):
        global active, peak
        # A receiver bound by server-info cannot deliver on another event loop.
        received = self.receiver_loop.create_future()
        self.receiver_loop.call_soon(received.set_result, None)
        await received
        assert kw['input_ids']==[10,11] and kw['stream'] is False
        assert kw['sampling_params']==dict(temperature=0,top_p=1,max_new_tokens=128,ignore_eos=True)
        active+=1;assert active <= int(os.environ['CONCURRENCY'])
        peak=max(peak,active)
        await asyncio.sleep(0.001)
        ids=[20]*128;ids[1]=151645
        active-=1
        return dict(output_ids=ids, meta_info=dict(id=kw['rid'], prompt_tokens=2, completion_tokens=128,
                    finish_reason=dict(type='length')))
    def shutdown(self):assert peak==int(os.environ['CONCURRENCY'])
module('sglang', Engine=SG)
import json
try:
    runpy.run_path(sys.argv[1],run_name='__main__')
except SystemExit as exit_status:
    if exit_status.code == 0 and SG.instance is not None:
        assert SG.instance.loop.is_closed(), 'SGLang receiver loop survived shutdown'
    raise
'''


class PythonAdapterTests(unittest.TestCase):
    def test_invalid_resolved_runtime_is_refused_despite_valid_requests(self):
        prompts = ['The capital city of France is', 'The three primary colors are',
                   'Water boils at a temperature of', 'The Pythagorean theorem states that',
                   'In 1969, humans first walked on', 'A prime number is a natural number']
        shared = [('hip', None, 'ROCm backend required'), ('cuda', None, 'ROCm backend required'),
                  ('unavailable', None, 'ROCm backend required'), ('arch', None, 'gfx1151 required')]
        vllm = [('vllm:' + key, value, 'vLLM resolved configuration mismatch') for key, value in (
            ('model_config.dtype', 'torch.float32'), ('model_config.enforce_eager', True),
            ('model_config.max_model_len', 1024), ('scheduler_config.max_num_seqs', 1),
            ('cache_config.enable_prefix_caching', True), ('cache_config.cache_dtype', 'fp8'),
            ('cache_config.kv_cache_memory_bytes', 1207959551))]
        sglang = [('sg:' + key, value, 'SGLang resolved configuration mismatch') for key, value in (
            ('dtype', 'float32'), ('kv_cache_dtype', 'fp8'), ('context_length', 1024),
            ('max_running_requests', 1), ('disable_radix_cache', False),
            ('attention_backend', 'other'), ('disable_cuda_graph', True))]
        sglang.append(('sg:max_total_num_tokens', 8191, 'SGLang resolved KV capacity below 8192'))
        for name, cases in (('vLLM', shared + vllm), ('patched SGLang', shared + sglang)):
            for mode, value, error in cases:
                with self.subTest(engine=name, mode=mode):
                    commands = [dict(schema=1, id=1, command='configure', engine=name, model='fixture',
                                     prompts=prompts, prompt_ids=[[10, 11]] * 6),
                                dict(schema=1, id=2, command='run', phase='qualification', concurrency=1, repetition=0),
                                dict(schema=1, id=3, command='shutdown')]
                    result = subprocess.run([sys.executable, '-c', HARNESS,
                        str(ROOT / 'tools/bench/strix_four_engine/python_adapter.py')],
                        input=''.join(json.dumps(x) + '\n' for x in commands), capture_output=True, text=True,
                        env=dict(os.environ, CASE=mode, INVALID_VALUE=json.dumps(value), CONCURRENCY='1'), timeout=10)
                    replies = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertEqual(len(replies), 1)
                    self.assertEqual(replies[0]['id'], 1)
                    self.assertEqual(replies[0]['status'], 'error')
                    self.assertIn(error, replies[0]['error'])

    def test_real_python_adapters_consume_ids_and_bound_active_requests(self):
        prompts = ['The capital city of France is', 'The three primary colors are',
                   'Water boils at a temperature of', 'The Pythagorean theorem states that',
                   'In 1969, humans first walked on', 'A prime number is a natural number']
        for name in ('vLLM', 'patched SGLang'):
            for c in (1, 4):
                for mode in ('', 'stream') if name == 'vLLM' else ('',):
                    with self.subTest(engine=name, concurrency=c, mode=mode):
                        commands = [dict(schema=1,id=1,command='configure',engine=name,model='fixture',
                                         prompts=prompts,prompt_ids=[[10,11]]*6),
                                    dict(schema=1,id=2,command='run',phase='qualification',concurrency=c,repetition=0),
                                    dict(schema=1,id=3,command='run',phase='qualification',concurrency=c,repetition=1),
                                    dict(schema=1,id=4,command='shutdown')]
                        result = subprocess.run([sys.executable,'-c',HARNESS,
                            str(ROOT/'tools/bench/strix_four_engine/python_adapter.py')],
                            input=''.join(json.dumps(x)+'\n' for x in commands), capture_output=True,text=True,
                            env=dict(os.environ,CASE=mode,CONCURRENCY=str(c)),timeout=10)
                        replies=[json.loads(line) for line in result.stdout.splitlines()]
                        if mode:
                            self.assertEqual(result.returncode,1,result.stderr)
                            self.assertIn('invalid cumulative token stream',replies[-1]['error'])
                        else:
                            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                            self.assertEqual([r['id'] for r in replies],[1,2,3,4])
                            for corpus in replies[1:3]:
                                self.assertEqual(len(corpus['requests']),6)
                                for request in corpus['requests']:
                                    ids=[20]*128;ids[1]=151645
                                    self.assertEqual(request['output_ids'],ids)
                                    self.assertEqual(request['prompt_ids'],[10,11])


if __name__=='__main__': unittest.main()
