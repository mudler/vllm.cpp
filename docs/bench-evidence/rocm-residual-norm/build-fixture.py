from pathlib import Path
import concurrent.futures,hashlib,json,os,shlex,shutil,subprocess,time
work=Path('/home/vikash/vllm.cpp-residual-norm-repair1');original=Path('/home/vikash/vllm.cpp-residual-norm-impl');root=Path('/home/vikash/.cache/residual-norm-repair1');frozen=Path('/home/vikash/.cache/residual-norm-impl/green-freeze-2')
inputs={};commands=[]
def seal(p):
 d=p.read_bytes();return {'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest()}
def run(name,argv,cwd):
 p=root/(name+'.log');start=time.monotonic()
 with p.open('w') as out:r=subprocess.run(argv,cwd=cwd,stdout=out,stderr=subprocess.STDOUT)
 entry={'name':name,'argv':argv,'cwd':str(cwd),'exit_code':r.returncode,'seconds':time.monotonic()-start,'log':str(p),'log_sha256':seal(p)['sha256']};commands.append(entry)
 if r.returncode:raise RuntimeError(json.dumps(entry))
 return entry
# Rebuild only the changed test and its own test main. Production archives are
# private copies of the unchanged, already-built base and are sealed inputs.
compile_jobs=[];plans={}
for flavor in ('cpu','hip'):
 oldbuild=original/('build-residual-'+flavor);build=work/('build-repair-'+flavor);build.mkdir(exist_ok=True);(build/'tests').mkdir(exist_ok=True)
 shutil.copytree(oldbuild/'include',build/'include',dirs_exist_ok=True)
 libsource=frozen/'build-residual-hip/libvllm.a' if flavor=='hip' else oldbuild/'libvllm.a'
 for source,target in ((libsource,build/'libvllm.a'),(oldbuild/'libblake3_vendored.a',build/'libblake3_vendored.a')):
  inputs[str(source)]=seal(source);shutil.copyfile(source,target);assert seal(target)==inputs[str(source)];inputs[str(target)]=seal(target)
 db=json.loads((oldbuild/'compile_commands.json').read_text())
 for suffix in ('/tests/vt/test_ops_residual_rmsnorm.cpp','/tests/doctest_main.cpp'):
  entry=next(x for x in db if x['file'].endswith(suffix));argv=shlex.split(entry['command']);argv=[a.replace(str(original),str(work)).replace(str(work/('build-residual-'+flavor)),str(build)) for a in argv]
  obj=build/('test.o' if suffix.endswith('test_ops_residual_rmsnorm.cpp') else 'main.o');argv[argv.index('-o')+1]=str(obj)
  compile_jobs.append((flavor+'-'+obj.stem,argv,build))
 plans[flavor]=build
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f in [pool.submit(run,*j) for j in compile_jobs]:f.result()
for flavor,build in plans.items():
 run(flavor+'-main-archive',['ar','rcs',str(build/'tests/libvllm_test_main.a'),str(build/'main.o')],build)
 if flavor=='hip':
  argv=['/opt/rocm/lib/llvm/bin/clang++','--rocm-path=/opt/rocm','-O3','-DNDEBUG','--offload-arch=gfx1100','-Xlinker','--whole-archive','-Xlinker',str(build/'libvllm.a'),'-Xlinker','--no-whole-archive',str(build/'test.o'),'-o',str(build/'tests/test_ops_residual_rmsnorm'),str(build/'libvllm.a'),str(build/'tests/libvllm_test_main.a'),str(build/'libblake3_vendored.a'),'/opt/rocm/lib/libamdhip64.so','/opt/rocm/lib/libhipblas.so','/opt/rocm/lib/libhipblaslt.so','/opt/rocm/lib/libamdhip64.so.7.15.26333-0000000','-lgcc','-lgcc','-lgcc','-lgcc']
 else:
  argv=['/usr/bin/c++','-O3','-DNDEBUG','-Wl,--whole-archive,'+str(build/'libvllm.a')+',--no-whole-archive',str(build/'test.o'),'-o',str(build/'tests/test_ops_residual_rmsnorm'),str(build/'libvllm.a'),str(build/'tests/libvllm_test_main.a'),str(build/'libblake3_vendored.a')]
 run(flavor+'-link',argv,build)
 if flavor=='hip':hip_link=argv
# This scratch-only mutation changes the real provider launch to the default
# stream. No tracked product file or frozen base archive is modified.
mutation=root/'wrong-stream';mutation.mkdir(exist_ok=True);source=work/'src/vt/rocm/rocm_residual_rmsnorm.hip';text=source.read_text();before='<<<rows, kBlock, 0, static_cast<hipStream_t>(queue.handle)>>>';assert text.count(before)==1;mutated=mutation/source.name;mutated.write_text(text.replace(before,'<<<rows, kBlock, 0, nullptr>>>'));inputs[str(source)]=seal(source)
entry=next(x for x in db if x['file'].endswith('/src/vt/rocm/rocm_residual_rmsnorm.hip'));argv=shlex.split(entry['command']);argv=[a.replace(str(original),str(work)).replace(str(work/'build-residual-hip'),str(plans['hip'])) for a in argv];argv[argv.index('-o')+1]=str(mutation/'rocm_residual_rmsnorm.hip.o');argv[-1]=str(mutated);run('wrong-stream-compile',argv,mutation)
archive=mutation/'libvllm.a';shutil.copyfile(plans['hip']/'libvllm.a',archive);members=subprocess.check_output(['ar','t',str(archive)],text=True).splitlines();assert members.count('rocm_residual_rmsnorm.hip.o')==1
run('wrong-stream-archive',['ar','r',str(archive),str(mutation/'rocm_residual_rmsnorm.hip.o')],mutation)
argv=[str(archive) if x==str(plans['hip']/'libvllm.a') else x for x in hip_link];argv[argv.index('-o')+1]=str(mutation/'test_ops_residual_rmsnorm');run('wrong-stream-link',argv,mutation)
for p,expected in inputs.items():assert seal(Path(p))==expected
outputs={str(p):seal(p) for p in [plans['cpu']/'tests/test_ops_residual_rmsnorm',plans['hip']/'tests/test_ops_residual_rmsnorm',mutation/'test_ops_residual_rmsnorm',mutated]}
report={'base':'94ac5d1742c540fc5cbc2084d0fb74fdfd5dab41','scope':'Only test translation units rebuilt; immutable production archives copied into private build directories. Scratch mutation replaces one real HIP provider object.','inputs':inputs,'outputs':outputs,'commands':commands,'all_inputs_unchanged':True,'max_parallel_compilers':4}
(root/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','outputs':outputs},indent=2))
