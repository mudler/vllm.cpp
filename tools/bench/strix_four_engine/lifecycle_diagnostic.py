"""Bound six-request lifecycle diagnostics; never a performance acceptance gate."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

ENGINES = ('vLLM', 'patched SGLang', 'vllm.cpp', 'llama.cpp')
PROMPTS = ['The capital city of France is', 'The three primary colors are',
           'Water boils at a temperature of', 'The Pythagorean theorem states that',
           'In 1969, humans first walked on', 'A prime number is a natural number']
SETTINGS = dict(backend='rocm', model_dtype='bfloat16', kv_dtype='bfloat16',
                context_per_sequence=2048, slots=4, prefix_caching=False)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lease():
    if os.environ.get('RC_DEVICE') != 'strix:gpu0' or not os.environ.get('RC_JOB_ID', '').strip():
        raise RuntimeError('operator Strix lease required')


def verify_inputs(args, modules=None):
    lease()
    if digest(args.source) != args.source_sha256:
        raise RuntimeError('controller archive hash mismatch')
    if digest(args.manifest) != args.manifest_sha256:
        raise RuntimeError('manifest hash mismatch')
    if args.native_first_c1 and digest(args.canonical_reference) != args.canonical_reference_sha256:
        raise RuntimeError('canonical reference hash mismatch')
    for name, record in (modules or {}).items():
        module = sys.modules[name]
        if (str(Path(module.__file__).resolve()) != record['path'] or
                digest(record['path']) != record['sha256']):
            raise RuntimeError('controller module changed: ' + name)


@contextmanager
def bootstrap(args):
    """Import only the explicitly verified archive, before any engine launch."""
    verify_inputs(args)
    with tempfile.TemporaryDirectory(prefix='strix-lifecycle-source.', dir='/tmp') as directory:
        source = Path(directory).resolve()
        with tarfile.open(args.source) as archive:
            if archive.pax_headers.get('comment') != args.source_revision:
                raise RuntimeError('controller source revision mismatch')
            archive.extractall(source, filter='data')
        verify_inputs(args)
        names = ('tools', 'tools.bench', 'tools.bench.strix_four_engine.qualify',
                 'tools.bench.strix_four_engine.child_lifecycle', 'tools.bench.strix_four_engine.audit')
        for name in names:
            module = sys.modules.get(name)
            if module is not None and not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError('controller import escaped pinned source: ' + name)
        sys.path.insert(0, str(source))
        try:
            importlib.invalidate_caches()
            qualify, lifecycle, audit = [importlib.import_module(name) for name in names[2:]]
            modules = {}
            for name in names:
                path = Path(sys.modules[name].__file__).resolve()
                if not path.is_relative_to(source):
                    raise RuntimeError('controller import escaped pinned source: ' + name)
                modules[name] = dict(path=str(path), sha256=digest(path))
            if (qualify.PROMPTS != PROMPTS or
                    json.dumps(qualify.RESOLVED, sort_keys=True) != json.dumps(SETTINGS, sort_keys=True)):
                raise RuntimeError('controller source is not the canonical six-request c4 workload')
            provenance = dict(controller_head=args.source_revision, controller_sha256=args.source_sha256,
                controller_source=str(source), modules=modules, manifest_sha256=args.manifest_sha256,
                driver_path=str(Path(__file__).resolve()), driver_sha256=digest(__file__),
                rc_device=os.environ['RC_DEVICE'], rc_job_id=os.environ['RC_JOB_ID'])
            yield qualify, lifecycle, audit.publish_result, provenance
        finally:
            sys.path.remove(str(source))


def error_record(item, label, error):
    item['errors'].append(label + ': ' + repr(error))
    if hasattr(error, 'lifecycle_evidence'):
        item.setdefault('exception_lifecycles', []).append(copy.deepcopy(error.lifecycle_evidence))


def lifecycle_record(item, adapter):
    item['parent_status'] = adapter.process.returncode
    item['lifecycle'] = copy.deepcopy(adapter.lifecycle_evidence)
    try:
        os.killpg(adapter.process.pid, 0)
    except ProcessLookupError:
        item['group_absent'] = True
    else:
        raise RuntimeError('adapter process group remains present')
    evidence = item['lifecycle']
    if (type(item['parent_status']) is not int or item['parent_status'] != 0 or
            not isinstance(evidence, dict) or evidence.get('restored') is not True or
            type(evidence.get('cleanup_errors')) is not list or evidence['cleanup_errors'] or
            type(evidence.get('unresolved_children', [])) is not list or evidence.get('unresolved_children') or
            type(evidence.get('adopted')) is not list or
            any(not isinstance(row, dict) or type(row.get('pid')) is not int or row['pid'] <= 0 or
                row.get('cleanup') is not False or type(row.get('status')) is not int or
                not os.WIFEXITED(row['status']) or os.WEXITSTATUS(row['status']) != 0
                for row in evidence['adopted'])):
        raise RuntimeError('lifecycle evidence refuses acceptance')


def canonical_reference(args, manifest, qualify):
    """Bind previously observed input tokenization, never an output-token oracle."""
    data = args.canonical_reference.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.canonical_reference_sha256:
        raise ValueError('canonical reference hash mismatch')
    reference = qualify.parse(data)
    if type(reference.get('schema')) is not int or reference['schema'] != 1:
        raise ValueError('canonical reference requires integer schema 1')
    if type(manifest.get('schema')) is not int or manifest['schema'] != 1:
        raise ValueError('native first-c1 requires a schema-1 controller manifest')
    if qualify.canonical(reference.get('manifest')) != qualify.canonical(manifest):
        raise ValueError('canonical reference manifest mismatch')
    if reference['manifest']['engines']['vLLM']['source_revision'] != qualify.PINS['vLLM']:
        raise ValueError('canonical reference oracle pin mismatch')
    config = reference['engines']['vLLM']['configuration']
    if (type(config.get('schema')) is not int or config['schema'] != 1 or
            type(config.get('id')) is not int or config['id'] != 1 or config.get('status') != 'ok'):
        raise ValueError('canonical reference configuration identity mismatch')
    values = reference.get('prompt_ids')
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError('canonical reference requires six prompt arrays')
    prompts = [qualify.tokens(value) for value in values]
    received = [qualify.tokens(value) for value in config['prompt_ids']]
    if received != prompts:
        raise ValueError('canonical reference configuration prompt mismatch')
    if qualify.canonical(config['requested']) != qualify.canonical(qualify.RESOLVED):
        raise ValueError('canonical reference requested settings mismatch')
    workload = qualify.canonical(dict(prompts=qualify.PROMPTS, prompt_ids=prompts,
        tokens=128, concurrency=[1, 4], settings=qualify.RESOLVED))
    if reference.get('workload_sha256') != workload:
        raise ValueError('canonical reference workload mismatch')
    return prompts, dict(path=str(args.canonical_reference.resolve()),
        sha256=args.canonical_reference_sha256, workload_sha256=workload,
        purpose='previously observed input tokenization only; not an output-token oracle')


def retain_exchanges(adapter, item):
    """Retain exchanges at the existing transport boundary, including close()."""
    exchange = adapter.exchange
    item['exchanges'] = []

    def retained(command):
        record = dict(command=copy.deepcopy(command))
        item['exchanges'].append(record)
        try:
            reply = exchange(command)
            record['reply'] = copy.deepcopy(reply)
            return reply
        except BaseException as error:
            record['error'] = repr(error)
            raise

    adapter.exchange = retained


def run_engine(args, name, manifest, prompts, qualify, lifecycle, modules):
    item = dict(status='FAIL', errors=[])
    directory = args.output / name.replace(' ', '-')
    adapter, before = None, None
    complete = closed = False
    try:
        directory.mkdir()
        verify_inputs(args, modules)
        before = copy.deepcopy(qualify.validate_manifest(manifest))
        item['before_bindings'] = before
        with lifecycle.controller_lifecycle():
            try:
                timeout = min(manifest['timeout_seconds'], 180) if args.native_first_c1 else manifest['timeout_seconds']
                adapter = qualify.Adapter(manifest['engines'][name], directory,
                                          manifest['log_bytes'], timeout)
                if args.native_first_c1:
                    retain_exchanges(adapter, item)
                item['pgid'] = adapter.process.pid
                config = adapter.exchange(dict(command='configure', engine=name,
                    model=manifest['model']['directory'], gguf=manifest['gguf']['path'],
                    prompts=PROMPTS, prompt_ids=prompts))
                item['configuration'] = config
                received = [qualify.tokens(value) for value in config['prompt_ids']]
                if len(received) != 6 or (prompts is not None and received != prompts):
                    raise RuntimeError('canonical prompt mismatch')
                if config['requested'] != SETTINGS:
                    raise RuntimeError('requested settings mismatch')
                prompts = received
                phase, concurrency = ('qualification', 1) if args.native_first_c1 else ('warmup', 4)
                item[phase] = adapter.exchange(dict(command='run', phase=phase, concurrency=concurrency, repetition=0))
                qualify.validate_run(item[phase], prompts, concurrency)  # Structural only, not cross-engine token parity.
                complete = True
            finally:
                if adapter is not None:
                    try:
                        adapter.close()
                        closed = True
                    except BaseException as error:
                        error_record(item, 'teardown', error)
    except BaseException as error:
        error_record(item, 'execution', error)
    finally:
        if adapter is not None:
            try:
                lifecycle_record(item, adapter)
            except BaseException as error:
                error_record(item, 'lifecycle', error)
        try:
            after = qualify.validate_manifest(manifest)
            item['after_bindings'] = after
            verify_inputs(args, modules)
            if before is None or before != after:
                raise RuntimeError('before/after bindings changed')
            item['bindings_unchanged'] = True
        except BaseException as error:
            error_record(item, 'post-run binding check', error)
    if complete and closed and item.get('group_absent') and not item['errors']:
        item['status'] = 'PASS'
    return copy.deepcopy(item), prompts


def execute(args, qualify, lifecycle, publish_result, provenance):
    manifest = qualify.parse(args.manifest.read_text())
    names = args.engine or (['vllm.cpp'] if args.native_first_c1 else list(ENGINES))
    if len(names) != len(set(names)):
        raise ValueError('duplicate selected engine')
    if args.native_first_c1 and names != ['vllm.cpp']:
        raise ValueError('native first-c1 requires exactly vllm.cpp')
    if not args.native_first_c1 and names[0] in ('vllm.cpp', 'llama.cpp'):
        raise ValueError('native-first selection requires canonical prompt IDs; select a Python engine first')
    prompts, reference = None, None
    if args.native_first_c1:
        prompts, reference = canonical_reference(args, manifest, qualify)
    args.output.mkdir(parents=False, exist_ok=False)
    result = dict(schema=1, status='FAIL', purpose='lifecycle validation only; no parity or throughput acceptance',
                  selected_engines=names, engines={}, errors=[], **provenance)
    if args.native_first_c1:
        result.update(diagnostic='native-first-c1', canonical_reference=reference, prompt_ids=copy.deepcopy(prompts))
    for name in names:
        item, prompts = run_engine(args, name, manifest, prompts, qualify, lifecycle, provenance['modules'])
        result['engines'][name] = item
        try:
            publish_result(item, args.output / name.replace(' ', '-') / 'result.json')
        except BaseException as error:
            result['errors'].append('engine publication ' + name + ': ' + repr(error))
            break
        if item['status'] != 'PASS':
            break
    if (len(result['engines']) == len(names) and not result['errors'] and
            all(item['status'] == 'PASS' for item in result['engines'].values())):
        result['status'] = 'PASS'
    try:
        publish_result(result, args.output / 'result.json')
    except BaseException as error:
        print('final publication: ' + repr(error), file=sys.stderr)
        return 1
    print(result['status'], args.output / 'result.json', flush=True)
    return 0 if result['status'] == 'PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', choices=ENGINES, action='append')
    parser.add_argument('--native-first-c1', action='store_true')
    parser.add_argument('--canonical-reference', type=Path)
    parser.add_argument('--canonical-reference-sha256')
    args = parser.parse_args(argv)
    try:
        if args.native_first_c1:
            if args.canonical_reference is None or not args.canonical_reference_sha256:
                raise ValueError('native first-c1 requires canonical reference and expected SHA256')
        elif args.canonical_reference is not None or args.canonical_reference_sha256 is not None:
            raise ValueError('canonical reference arguments require native first-c1 mode')
        with bootstrap(args) as (qualify, lifecycle, publisher, provenance):
            return execute(args, qualify, lifecycle, publisher, provenance)
    except BaseException as error:
        print('diagnostic refusal: ' + repr(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
