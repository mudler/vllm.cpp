"""CPU CLI contracts for the current-pin Strix oracle (#3043)."""
import os
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import time
import signal
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/bench/strix_vllm_oracle/worker.py"
RUNTIME = WORKER.with_name("runtime.py")


class OracleCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strix3043-test-")
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        for record in self.tmp.rglob("local.json"):
            local = Path(json.loads(record.read_text())["local"])
            if local.parent == Path("/tmp") and local.name.startswith("strix-vllm3043-"):
                shutil.rmtree(local, ignore_errors=True)

    def artifact(self, name, content):
        path = self.tmp / name
        path.write_bytes(content)
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    def archive(self, name, revision, unsafe=False):
        path = self.tmp / name
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": revision}) as archive:
            for filename, data in {("../escape" if unsafe else "setup.py"): b"# fixture\n",
                                   "pyproject.toml": b"[build-system]\nrequires=[]\n",
                                   "tests/kernels/test_fused_recurrent_packed_decode.py": b"def test_fixture(): pass\n"}.items():
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "revision": revision}

    def fixture(self):
        self.manifest = {
            "schema": 1,
            "sources": {
                "vllm": self.archive("vllm.tar", "e126687a9a828d513c01a07cd69f025f27d63280"),
                "plugin": self.archive("plugin.tar", "d4c1f0d082fc7cd4350da56689109a01c1f29d6c"),
            },
            "model": self.artifact("model.gguf", b"gguf"),
            "assets": [],
            "dependencies": {"requirements": self.artifact("requirements.txt", b"pytest==9.1.1\n"),
                             "expected_versions": {"torch": "2.13.0+rocm7.2", "torchvision": "0.28.0+rocm7.2",
                                                   "triton-rocm": "3.7.1", "triton_runtime": "3.7.1"},
                             "indexes": ["https://download.pytorch.org/whl/rocm7.2/", "https://pypi.org/simple/"],
                             "source_archives": []},
        }
        for i, name in enumerate(("model/config.json", "tokenizer/config.json", "tokenizer/tokenizer.json",
                                  "tokenizer/tokenizer_config.json", "mmproj/mmproj-BF16.gguf")):
            self.manifest["assets"].append({**self.artifact(f"asset{i}", b"{}"), "relative_path": name})
        self.path = self.tmp / "manifest.json"
        self.plugin_sha = self.manifest["sources"]["plugin"]["sha256"]
        self.model_sha = self.manifest["model"]["sha256"]
        self.tool = self.tmp / "python-tool"
        self.tool.write_text(FAKE_TOOL)
        self.tool.chmod(0o755)
        self.write_manifest()

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def cli(self, phase="build", output="out", state=None, as_process=False, **env):
        # The real parser, build/run functions and monitors execute. Only pinned
        # fixture bytes and the external toolchain boundary differ from hardware.
        bootstrap = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0,{str(WORKER.parent)!r}); import worker; "
            f"worker.PLUGIN_SHA={self.plugin_sha!r}; "
            f"worker.MODEL_SHA={self.model_sha!r}; worker.MODEL_SIZE=4; "
            f"worker.PYTHON={str(self.tool)!r}; worker.TOOLCHAIN={str(self.tool)!r}; "
            "worker.main()"
        )
        args = [sys.executable, "-c", bootstrap, "--phase", phase, "--manifest", str(self.path),
                "--output", str(self.tmp / output)]
        if state:
            args += ["--state", str(state)]
        child_env = {**os.environ, "RC_DEVICE": "strix:gpu0", "RC_JOB_ID": "cpu-fixture",
                     "STRIX_TEST_LOG": str(self.tmp / "commands.jsonl"), **env}
        if as_process:
            return subprocess.Popen(args, env=child_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return subprocess.run(args, env=child_env, text=True, capture_output=True, timeout=30)

    def built(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.tmp / "out/build-state.json"

    def test_cli_builds_fresh_and_runs_all_prompts(self):
        self.fixture()
        state = self.built()
        value = json.loads(state.read_text())
        self.addCleanup(__import__("shutil").rmtree, value["local"], True)
        self.assertTrue(Path(value["local"]).is_relative_to("/tmp"))
        self.assertNotIn("vllm-gfx1151-2740", value["local"])
        self.assertEqual(value["status"], "BUILT")
        result = self.cli(phase="run", output="run", state=state)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = json.loads((self.tmp / "run/result.json").read_text())
        self.assertEqual(result["gateability"], "PASS")
        self.assertEqual(result["token_gate"], "FAIL (carried, not remeasured)")
        self.assertEqual(result["upstream_tests"]["status"], "PASS")
        commands = [json.loads(line) for line in (self.tmp / "commands.jsonl").read_text().splitlines()]
        self.assertTrue(any("wheel" in c["argv"] for c in commands))
        self.assertTrue(any("check" in c["argv"] for c in commands))
        for c in commands:
            self.assertEqual(c["jobs"], "4")
            self.assertEqual(c["arch"], "gfx1151")

    def test_dependencies_cannot_target_a_donor_environment(self):
        self.fixture()
        result = self.cli(PIP_TARGET="/tmp/vllm-gfx1151-2740/venv", PYTHONUSERBASE="/tmp/vllm-gfx1151-2740")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [json.loads(line) for line in (self.tmp / "commands.jsonl").read_text().splitlines()]
        self.assertTrue(commands)
        for command in commands:
            self.assertIsNone(command["pip_target"])
            self.assertIsNone(command["python_userbase"])
            self.assertEqual(command["pip_config"], os.devnull)

    def test_cli_rejects_stale_identity_and_manifest(self):
        self.fixture()
        state = self.built()
        data = json.loads(state.read_text())
        self.addCleanup(__import__("shutil").rmtree, data["local"], True)
        self.manifest["extra"] = "different"
        self.write_manifest()
        result = self.cli(phase="run", output="different", state=state)
        self.assertIn("build manifest mismatch", result.stderr)
        del self.manifest["extra"]
        self.write_manifest()
        path = Path(data["identity"]["extensions"][0])
        path.write_bytes(b"tampered")
        result = self.cli(phase="run", output="stale", state=state)
        self.assertIn("installed identity mismatch", result.stderr)

    def test_cli_preserves_failed_build_and_runtime_reports(self):
        self.fixture()
        result = self.cli(STRIX_TEST_FAIL="pip")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.tmp / "out/failure.json").exists())
        self.assertFalse((self.tmp / "out/build-state.json").exists())
        self.assertTrue(list((self.tmp / "out/logs").glob("*.log")))
        commands = [json.loads(line) for line in (self.tmp / "commands.jsonl").read_text().splitlines()]
        self.assertIn("command failed: bootstrap: 3", result.stderr)
        self.assertEqual(sum(c["argv"][:2] == ["-m", "pip"] for c in commands), 1)

    def test_failed_generation_preserves_partial_capture(self):
        self.fixture()
        state = self.built()
        result = self.cli(phase="run", output="partial-failure", state=state, STRIX_TEST_FAIL="generation-exit")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.tmp / "partial-failure/tokens.json").exists())
        self.assertFalse((self.tmp / "partial-failure/result.json").exists())

    def test_cli_refuses_wrong_runtime_partial_tokens_and_skipped_tests(self):
        self.fixture()
        state = self.built()
        self.addCleanup(__import__("shutil").rmtree, json.loads(state.read_text())["local"], True)
        for mode, expected in (("device", "runtime identity"), ("version", "runtime identity"),
                               ("torch", "runtime identity"), ("vision", "runtime identity"),
                               ("triton-overlap", "runtime identity"), ("predicate", "runtime identity"),
                               ("partial", "six complete"), ("prompts", "six complete"),
                               ("eager", "production compilation"), ("skip", "upstream tests")):
            with self.subTest(mode=mode):
                result = self.cli(phase="run", output=mode, state=state, STRIX_TEST_FAIL=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertTrue((self.tmp / mode / "failure.json").exists())

    def test_cli_rejects_changed_model_and_missing_assets_before_generation(self):
        self.fixture()
        state = self.built()
        self.addCleanup(__import__("shutil").rmtree, json.loads(state.read_text())["local"], True)
        Path(self.manifest["model"]["path"]).write_bytes(b"evil")
        result = self.cli(phase="run", output="badmodel", state=state)
        self.assertIn("sha256 mismatch", result.stderr)
        self.assertFalse((self.tmp / "badmodel/logs/generation.command.json").exists())

    def test_cli_rejects_source_modification_and_wrong_offload_targets(self):
        self.fixture()
        for mode, expected in (("source", "source changed"), ("offload", "device targets")):
            result = self.cli(output=mode, STRIX_TEST_FAIL=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)
            self.assertFalse((self.tmp / mode / "build-state.json").exists())

    def test_cli_stops_log_growth_and_resource_exhaustion(self):
        self.fixture()
        for key, value, expected in (("max_log_bytes", 100, "log output limit"),
                                     ("min_mem_bytes", 10**18, "memory headroom"),
                                     ("min_disk_bytes", 10**18, "disk headroom")):
            self.manifest["limits"] = {key: value, "build_timeout": 0.2}
            self.write_manifest()
            result = self.cli(output=key, STRIX_TEST_FAIL="flood")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)

    def test_continuous_writer_obeys_timeout_and_persists_logs(self):
        self.fixture()
        self.manifest["limits"] = {"build_timeout": 0.15, "max_log_bytes": 536870912}
        self.write_manifest()
        result = self.cli(STRIX_TEST_FAIL="stream")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command timeout", result.stderr)
        self.assertTrue((self.tmp / "out/logs/venv.log").stat().st_size > 0)

    def test_cli_stops_fault_and_timeout_writers_and_preserves_logs(self):
        self.fixture()
        self.manifest["limits"] = {"build_timeout": 1}
        self.write_manifest()
        for mode in ("fault", "timeout"):
            with self.subTest(mode=mode):
                result = self.cli(output=mode, STRIX_TEST_FAIL=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fatal GPU diagnostic" if mode == "fault" else "command timeout", result.stderr)
                self.assertTrue((self.tmp / mode / "failure.json").exists())
                logs = list((self.tmp / mode / "logs").glob("*.log"))
                self.assertTrue(logs)
                text = "".join(path.read_text() for path in logs)
                self.assertIn("writer started", text)
                pid = int((self.tmp / "writer.pid").read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_runtime_cli_executes_production_generation_contract(self):
        for bad in (False, True):
            with self.subTest(bad_tokenizer=bad):
                output = self.tmp / ("bad.json" if bad else "tokens.json")
                result = subprocess.run([sys.executable, "-c", RUNTIME_LOADER, str(RUNTIME),
                                         str(self.tmp), str(output), str(int(bad))],
                                        text=True, capture_output=True, timeout=10)
                if bad:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("tokenization", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    data = json.loads(output.read_text())
                    self.assertEqual(len(data["records"]), 6)
                    self.assertTrue(all(len(r["gen_ids"]) == 48 for r in data["records"]))
                    self.assertEqual(data["resolved_config"]["dtype"], "bfloat16")

    def test_termination_signal_cleans_up_owned_child(self):
        self.fixture()
        process = self.cli(as_process=True, STRIX_TEST_FAIL="timeout")
        child = None
        try:
            deadline = time.monotonic() + 5
            while not (self.tmp / "writer.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            child = int((self.tmp / "writer.pid").read_text())
            process.terminate()
            process.communicate(timeout=5)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
            self.assertTrue((self.tmp / "out/failure.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            if child:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_cli_refuses_wrong_archive_hash_before_build(self):
        self.fixture()
        self.manifest["sources"]["vllm"]["sha256"] = "0" * 64
        self.write_manifest()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha256 mismatch", result.stderr)

    def test_cli_refuses_wrong_revision_and_unsafe_archives(self):
        self.fixture()
        self.manifest["sources"]["vllm"]["revision"] = "1" * 40
        self.write_manifest()
        result = self.cli()
        self.assertIn("source pin mismatch", result.stderr)
        self.manifest["sources"]["vllm"] = self.archive("unsafe.tar", "e126687a9a828d513c01a07cd69f025f27d63280", True)
        self.write_manifest()
        result = self.cli(output="unsafe")
        self.assertIn("unsafe archive", result.stderr)
        self.assertFalse((self.tmp / "escape").exists())

    def test_cli_requires_archive_marker_and_fixed_plugin_hash(self):
        self.fixture()
        record = self.archive("wrong-marker.tar", "1" * 40)
        record["revision"] = self.manifest["sources"]["vllm"]["revision"]
        self.manifest["sources"]["vllm"] = record
        self.write_manifest()
        result = self.cli()
        self.assertIn("archive commit marker mismatch", result.stderr)
        self.manifest["sources"]["vllm"] = self.archive("good.tar", record["revision"])
        self.manifest["sources"]["plugin"]["sha256"] = "0" * 64
        self.write_manifest()
        result = self.cli(output="plugin")
        self.assertIn("plugin archive pin mismatch", result.stderr)

    def test_cli_rejects_import_and_compiler_injection(self):
        self.fixture()
        for key in ("PYTHONPATH", "LD_PRELOAD", "CMAKE_ARGS"):
            result = self.cli(output=key, **{key: "/not-approved"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inherited tuning", result.stderr)

    def test_cli_refuses_tuning_and_reused_output(self):
        self.fixture()
        result = self.cli(VLLM_USE_PRECOMPILED="1")
        self.assertIn("inherited tuning", result.stderr)
        (self.tmp / "existing").mkdir()
        result = self.cli(output="existing")
        self.assertIn("output already exists", result.stderr)

    def test_cli_requires_successful_same_manifest_build_state(self):
        self.fixture()
        state = self.tmp / "state.json"
        state.write_text(json.dumps({"status": "FAILED"}))
        result = self.cli(phase="run", state=state)
        self.assertIn("successful build state", result.stderr)

    def test_cli_refuses_without_the_named_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.pop("RC_DEVICE", None)
            env.pop("RC_JOB_ID", None)
            result = subprocess.run(
                [sys.executable, str(WORKER), "--phase", "build", "--manifest",
                 str(Path(tmp) / "missing.json"), "--output", str(Path(tmp) / "out")],
                env=env, capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires a strix:gpu0 lease", result.stderr)
            self.assertFalse((Path(tmp) / "out").exists())


FAKE_TOOL = r'''#!/usr/bin/python3
import hashlib, json, os, pathlib, shutil, sys, time
args = sys.argv[1:]
log = pathlib.Path(os.environ['STRIX_TEST_LOG'])
with log.open('a') as stream:
    stream.write(json.dumps({'argv': args, 'jobs': os.environ.get('MAX_JOBS'),
                             'arch': os.environ.get('PYTORCH_ROCM_ARCH'),
                             'pip_target':os.environ.get('PIP_TARGET'),
                             'pip_config':os.environ.get('PIP_CONFIG_FILE'),
                             'python_userbase':os.environ.get('PYTHONUSERBASE')}) + '\n')
mode = os.environ.get('STRIX_TEST_FAIL', '')
if mode == 'flood':
    print('X'*10000,flush=True); time.sleep(30)
if mode == 'stream':
    while True:
        os.write(1,b'X'*65536)
        time.sleep(0.0001)
if mode in ('fault','timeout'):
    (log.parent/'writer.pid').write_text(str(os.getpid()))
    print('writer started',flush=True)
    if mode == 'fault': print('GPU Hang',flush=True)
    time.sleep(30)
if args[:2] == ['-m', 'venv']:
    target = pathlib.Path(args[2])/'bin/python'
    target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, target); target.chmod(0o755)
elif args[:2] == ['-m', 'pip']:
    if mode == 'pip': print('simulated pip failure'); sys.exit(3)
    if 'wheel' in args:
        dest = pathlib.Path(args[args.index('--wheel-dir')+1]); dest.mkdir(exist_ok=True)
        name = 'vllm_gguf_plugin' if 'plugin' in args[-1] else 'vllm'
        (dest/(name+'-1.0-py3-none-any.whl')).write_bytes(b'fresh wheel')
        if mode=='source': (pathlib.Path(args[-1])/'setup.py').write_text('changed source')
    if '--report' in args:
        pathlib.Path(args[args.index('--report')+1]).write_text('{"install": []}')
    if 'freeze' in args: print('torch==2.13.0+rocm7.2\ntriton-rocm==3.7.1')
elif '--mode' in args:
    env = pathlib.Path(__file__).parent.parent
    packages = env/'lib/site-packages'
    for name in ('vllm', 'vllm_gguf_plugin'):
        folder = packages/name; folder.mkdir(parents=True,exist_ok=True)
        for file in ('__init__.py','_C.so'):
            path=folder/file
            if not path.exists(): path.write_bytes(b'new built code')
    identity = {'runtime_version':'0.28.1rc1.dev132+ge126687a9.rocm724',
                'distribution_version':'0.28.1rc1.dev132+ge126687a9.rocm724',
                'platform':'rocm','device_arch':'gfx1100' if mode=='device' else 'gfx1151',
                'torch':'2.13.0+rocm7.2','torchvision':'0.28.0+rocm7.2','triton':'3.7.1',
                'triton_rocm_distribution':'3.7.1','triton_distribution':None,
                'vllm_path':str(packages/'vllm/__init__.py'),
                'plugin_path':str(packages/'vllm_gguf_plugin/__init__.py'),
                'extensions':[str(packages/'vllm/_C.so'),str(packages/'vllm_gguf_plugin/_C.so')],
                'plugin_predicates':dict(Q4_K=True,Q5_K=True,Q6_K=True,Q8_0=True)}
    result=identity
    if mode=='version':identity['runtime_version']='0.26.0.dev0+g5559679229.rocm724'
    if mode=='torch':identity['torch']='2.14.0+rocm7.2'
    if mode=='vision':identity['torchvision']='0.29.0+rocm7.2'
    if mode=='triton-overlap':identity['triton_distribution']='3.7.1'
    if mode=='predicate':identity['plugin_predicates']['Q6_K']=False
    if args[args.index('--mode')+1]=='generate':
        import runpy
        runtime = runpy.run_path(args[0])
        result={'identity':identity,'engine_kwargs':runtime['ENGINE_KWARGS'],
                'resolved_config':{'dtype':'bfloat16','compilation_config':{'mode':3}},
                'records':[{'prompt_ids':p,'gen_ids':[42]*(47 if mode=='partial' else 48)}
                           for p in runtime['PROMPT_IDS']]}
        if mode=='eager':result['engine_kwargs']['enforce_eager']=True
        if mode=='prompts':result['records'][0]['prompt_ids']=[1]
    pathlib.Path(args[args.index('--output')+1]).write_text(json.dumps(result))
    if mode=='generation-exit' and args[args.index('--mode')+1]=='generate': sys.exit(3)
elif args[:2] == ['-m','pytest']:
    path=pathlib.Path(next(a.split('=',1)[1] for a in args if a.startswith('--junitxml=')))
    skipped='1' if mode=='skip' else '0'
    path.write_text('<testsuites><testsuite tests="8" failures="0" errors="0" skipped="'+skipped+'"/></testsuites>')
else:
    print('hipv4-amdgcn-amd-amdhsa--'+('gfx1100' if mode=='offload' else 'gfx1151'))
'''

RUNTIME_LOADER = r'''
import sys, types, pathlib, runpy, importlib.metadata
runtime, root, out, bad = sys.argv[1:]
root=pathlib.Path(root); sys.prefix=str(root)
values=runpy.run_path(runtime)
def module(name, **values):
    result=types.ModuleType(name); result.__dict__.update(values)
    file=root/(name.replace('.','/')+'.py'); file.parent.mkdir(parents=True,exist_ok=True); file.write_text('fixture')
    result.__file__=str(file); sys.modules[name]=result; return result
torch=module('torch',__version__='2.13.0+rocm7.2',version=types.SimpleNamespace(hip='7.2'),
             cuda=types.SimpleNamespace(get_device_properties=lambda i:types.SimpleNamespace(gcnArchName='gfx1151')))
module('triton',__version__='3.7.1')
module('torchvision',__version__='0.28.0+rocm7.2')
class LLM:
    def __init__(self, **kwargs):
        assert kwargs['enforce_eager'] is False
        assert kwargs['quantization']=='gguf' and kwargs['trust_remote_code'] is False
        assert kwargs['max_model_len']==2048 and kwargs['max_num_batched_tokens']==2048
        assert kwargs['max_num_seqs']==1 and kwargs['gpu_memory_utilization']==0.60
        assert kwargs['limit_mm_per_prompt']=={'image':0,'video':0}
        self.llm_engine=types.SimpleNamespace(vllm_config=types.SimpleNamespace(
            model_config=types.SimpleNamespace(dtype='bfloat16'), compilation_config=types.SimpleNamespace(mode=3)))
    def apply_model(self, function):
        return [function(types.SimpleNamespace(named_modules=lambda:[]))]
    def generate(self, prompts, sampling):
        assert sampling==dict(temperature=0.0,top_p=1.0,max_tokens=48,ignore_eos=True)
        assert [p['prompt_token_ids'] for p in prompts]==values['PROMPT_IDS']
        return [types.SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],
                outputs=[types.SimpleNamespace(token_ids=[42]*48,text='fixture')]) for p in prompts]
module('vllm',__version__='0.28.1rc1.dev132+ge126687a9.rocm724',LLM=LLM,SamplingParams=lambda **kw:kw)
module('vllm.inputs',TokensPrompt=lambda **kw:kw)
module('vllm.platforms',current_platform=types.SimpleNamespace(is_rocm=lambda:True))
for name in ('vllm._C','vllm._rocm_C','vllm_gguf_plugin','vllm_gguf_plugin._C_gguf'):module(name)
module('vllm_gguf_plugin.ops',_cuda_kernel_available=lambda op,q:True,
       GGML_TYPE_Q4_K=12,GGML_TYPE_Q5_K=13,GGML_TYPE_Q6_K=14,GGML_TYPE_Q8_0=8)
class Tokenizer:
    def __call__(self,text,**kw):
        assert kw=={'add_special_tokens':False}
        return {'input_ids':[0] if bad=='1' else values['PROMPT_IDS'][values['PROMPTS'].index(text)]}
module('transformers',AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer()))
def version(name):
    if name=='triton':raise importlib.metadata.PackageNotFoundError(name)
    return {'vllm':'0.28.1rc1.dev132+ge126687a9.rocm724','triton-rocm':'3.7.1'}.get(name,'0.0.5')
importlib.metadata.version=version
importlib.metadata.entry_points=lambda **kw:[types.SimpleNamespace(name='gguf',value='vllm_gguf_plugin:register')]
sys.argv=[runtime,'--mode','generate','--assets',str(root),'--output',out]
runpy.run_path(runtime,run_name='__main__')
'''


if __name__ == "__main__":
    unittest.main()
