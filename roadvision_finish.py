"""Run from the RoadVision repository with its existing Python environment.
Repairs confirmed evaluation defects; runs new full evaluation, preserves logs/evidence.
Does not certify scientific validity or perform training.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import pathlib as P
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = P.Path.cwd()
EXP = ROOT / 'experiments' / 'phase13'
REPORT = ROOT / 'reports' / 'PHASE13B_CORRECTIONS'
LOGS = REPORT / 'logs'
OUT = ROOT / 'runs' / 'evaluation' / 'phase13b'
MODULES = ('dataset_audit','detection','geometry','depth_cavity','scheduler_temporal','quality_geo_maint')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def patch(p, fn, backup, changes):
    old = p.read_text(encoding='utf-8-sig')
    new, note = fn(old)
    ast.parse(new, filename=str(p))
    if old != new:
        dest = backup / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        p.write_text(new, encoding='utf-8')
    changes.append({'file':p.relative_to(ROOT).as_posix(),'changed':old != new,'note':note,'sha256':sha(p)})
    print(f'{p.name}: {note}', flush=True)

def fix_runner(s):
    old = 'os.path.join(project_root, "runs", "evaluation", "phase13", run_id)'
    new = 'os.path.join(project_root, "runs", "evaluation", "phase13b", run_id)'
    if old in s:
        if s.count(old)!=1: raise RuntimeError('Ambiguous output path')
        return s.replace(old,new), 'New results isolated in phase13b; original phase13 results untouched.'
    if new in s: return s, 'Phase13b output path already configured.'
    raise RuntimeError('Unexpected runner output path; inspect before proceeding.')

def fix_quality(s):
    tree=ast.parse(s)
    lines=s.splitlines(keepends=True)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='record_fresh_measurement']
    changed=0
    for n in sorted(calls,key=lambda x:x.lineno,reverse=True):
        keys={k.arg for k in n.keywords}
        needed={'raw_p75','iqr_norm_p75','ref_diagnostics'}
        if needed<=keys or len(n.args)>=10: continue
        context=''.join(lines[max(0,n.lineno-6):n.lineno-1])
        if 'Expired temporal cache' not in context or 'nc_exp' not in context or n.args or not isinstance(n.func.value,ast.Name) or n.func.value.id!='tcache':
            raise RuntimeError(f'Unexpected incomplete cache call at line {n.lineno}; refusing to invent real measurement values')
        if not {'track_id','frame_id','frame_idx','ts_sec','box_xyxy','is_valid','relative_score'}<=keys: raise RuntimeError('Unexpected synthetic fixture signature')
        if n.lineno!=n.end_lineno: raise RuntimeError('Unexpected multiline fixture; inspect manually')
        line=lines[n.lineno-1]; end=line.rfind(')')
        if end<0: raise RuntimeError('Cannot locate fixture call end')
        args={'raw_p75':'None','iqr_norm_p75':'None','ref_diagnostics':'{}'}
        extra=''.join(f', {k}={v}' for k,v in args.items() if k not in keys)
        indent=line[:len(line)-len(line.lstrip())]
        lines[n.lineno-1]=indent+'# Synthetic cache-expiration fixture; no physical depth measurements.\n'+line[:end]+extra+line[end:]
        changed+=1
    s=''.join(lines)
    sliced='for obs in obs_list[:300]:' in s
    s=s.replace('for obs in obs_list[:300]:','for obs in obs_list:')
    s=s.replace("exp_read.get('smoothed_relative_score')", "exp_read['relative_score']")
    s=s.replace('if exp_read["state"] == "EXPIRED" else', 'if exp_read["state"] == "EXPIRED" and exp_read["relative_score"] is None else')
    return s,f'Fixed {changed} incomplete synthetic cache call(s); removed first-300 truncation: {sliced}. Other quality logic requires evidence review.'

def fix_geometry(s):
    old='cav_out = extract_cavity_with_strategy(dmap, case["box"], strategy="elliptical_core", opt_cfg=opt_cfg, neighbor_boxes=case["neighbors"])'
    new='cav_out = extract_cavity_with_strategy(dmap, case["box"], strategy="elliptical_core", opt_cfg=opt_cfg, neighbor_boxes=case["neighbors"], precomputed_region=reg, precomputed_fit=fit_out)'
    if s.count(old)==1 and 'fit_out = fit_ransac_optimized(dmap, reg, cfg=opt_cfg)' in s:
        s=s.replace(old,new)
        # Move accuracy calculation outside timed block when exact original block exists.
        a='''        ref_crater = fit_out.get("ref_plane_crater")
        gt_err_rmse = (
            float(np.sqrt(np.mean((ref_crater - gt_crater) ** 2)))
            if (ref_crater is not None and ref_crater.shape == gt_crater.shape)
            else None
        )
'''
        if a in s:
            s=s.replace(a,'',1).replace('        dt_d = (time.perf_counter() - t0) * 1000.0\n','        dt_d = (time.perf_counter() - t0) * 1000.0\n'+a,1)
        return s,'Reuse precomputed region and fit; eliminate duplicate optimized RANSAC call. Timing scopes/accuracy still require review.'
    if 'precomputed_region=reg, precomputed_fit=fit_out' in s: return s,'Optimized fit reuse already present.'
    return s,'Original duplicate-call pattern absent; inspect actual benchmark before making speedup claims.'

def run(cmd, log, cwd=ROOT):
    print('\nRUN:',subprocess.list2cmdline([str(x) for x in cmd]),flush=True)
    with log.open('w',encoding='utf-8') as f:
        f.write('COMMAND: '+repr(cmd)+'\n');f.flush()
        p=subprocess.Popen(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in p.stdout:
            print(line,end='',flush=True);f.write(line);f.flush()
        code=p.wait();f.write(f'\nEXIT_CODE={code}\n')
    return code

def manifest_for(rid):
    p=OUT/rid/'experiment_manifest.json'
    if not p.is_file(): raise RuntimeError(f'No manifest: {p}')
    m=json.loads(p.read_text(encoding='utf-8'))
    if m.get('run_id')!=rid: raise RuntimeError('Wrong run ID in manifest')
    return m

def check_module(rid, module):
    m=manifest_for(rid)
    st=m['modules'][module]['status']
    p=OUT/rid/f'result_{module}.json'
    if st!='COMPLETED' or not p.is_file(): raise RuntimeError(f'{module}: status={st}, result exists={p.is_file()}')
    r=json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(r,dict) or r.get('status','COMPLETED')!='COMPLETED': raise RuntimeError(f'{module}: invalid or unsuccessful result')
    return r

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repair-only',action='store_true')
    ap.add_argument('--resume-run',help='Resume a FULL run created by this script, same run ID only')
    args=ap.parse_args()
    if not (EXP/'run_evaluation.py').is_file(): raise RuntimeError('Run this script from the RoadVision repository root.')
    LOGS.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup=REPORT/'repair_backups'/stamp
    changes=[]
    patch(EXP/'run_evaluation.py',fix_runner,backup,changes)
    patch(EXP/'quality_geo_maint_eval.py',fix_quality,backup,changes)
    patch(EXP/'geometry_ablation.py',fix_geometry,backup,changes)
    repair_record=LOGS/f'repair_{stamp}.json'
    repair_record.write_text(json.dumps(changes,indent=2),encoding='utf-8')
    if args.repair_only: return 0
    smoke='phase13b_quality_smoke_'+stamp
    base=[sys.executable,'-u','-m','experiments.phase13.run_evaluation']
    rc=run(base+['--mode','smoke','--run-id',smoke,'--experiments','quality_geo_maint'],LOGS/(smoke+'.log'))
    if rc!=0: raise RuntimeError(f'Targeted quality run failed (exit {rc}); see {LOGS/(smoke+".log")}')
    check_module(smoke,'quality_geo_maint')
    rid=args.resume_run or 'phase13b_full_'+stamp
    if not re.fullmatch(r'phase13b_full_[A-Za-z0-9_\-]+',rid): raise RuntimeError('Invalid full run ID')
    print('FULL RUN ID:',rid,flush=True)
    # Full-run results only; smoke outputs never copied into it.
    code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(EXP.glob('*.py'))}
    code_hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in EXP.glob('*.yaml')})
    provenance=LOGS/(rid+'_source.json')
    code_hashes['experiments/track_depth_cache.py']=sha(ROOT/'experiments'/'track_depth_cache.py')
    if args.resume_run:
        m=manifest_for(rid)
        if m.get('mode')!='full' or not provenance.is_file() or json.loads(provenance.read_text())!=code_hashes:
            raise RuntimeError('Resume refused: mode or recorded evaluation code/config differs. Use a fresh run.')
        env=json.loads((OUT/rid/'environment.json').read_text(encoding='utf-8'))
        for name,meta in env.get('checkpoint_hashes',{}).items():
            cp=ROOT/meta['path']
            if not cp.is_file() or sha(cp)!=meta['sha256']:
                raise RuntimeError(f'Resume refused: checkpoint changed: {name}')
    else:
        if (OUT/rid).exists(): raise RuntimeError('Run already exists')
        provenance.write_text(json.dumps(code_hashes,indent=2),encoding='utf-8')
    full_log=LOGS/(rid+'_'+stamp+'.log')
    rc=run(base+['--mode','full','--run-id',rid,'--experiments','all']+(['--resume'] if args.resume_run else []),full_log)
    if rc!=0: raise RuntimeError(f'Full evaluation failed (exit {rc}); completed modules are preserved. See {full_log}. Run ID: {rid}')
    for mod in MODULES: check_module(rid,mod)
    # Review scope is deliberately limited: no false final certification.
    status={'run_id':rid,'modules':manifest_for(rid)['modules'],'full_evaluation_exit_code':rc,'scientific_review':'PENDING','warnings':['Detection AP versus fixed-threshold counts, quality monotonicity, actual blank-video controls, measured multi-video scheduling and remaining audit issues must be verified from the source and result files. A completed harness is not proof of scientific validity.','Synthetic or exploratory evidence is not independent field validation.']}
    checks=[]
    checks.append(('python_tests', [sys.executable,'-m','unittest','discover','-s','tests'],ROOT))
    npm=shutil.which('npm.cmd') or shutil.which('npm')
    if npm and (ROOT/'roadvision-dashboard'/'package.json').is_file():
        for name in ('lint','build'): checks.append(('frontend_'+name,[npm,'run',name],ROOT/'roadvision-dashboard'))
    else: status['frontend']='NOT_RUN: npm or dashboard missing'
    logs=[repair_record,provenance,LOGS/(smoke+'.log'),full_log]
    for name,cmd,cwd in checks:
        log=LOGS/(rid+'_'+name+'_'+stamp+'.log');logs.append(log)
        status[name]={'exit_code':run(cmd,log,cwd)}
    status_file=LOGS/(rid+'_execution_status.json');status_file.write_text(json.dumps(status,indent=2),encoding='utf-8');logs.append(status_file)
    # Allowlist only run result formats; omit databases, raw media, weights, secrets, GPS files.
    suffixes={'.json','.csv','.yaml','.yml','.txt','.md','.log','.png','.svg','.pdf'}
    files=[p for p in (OUT/rid).rglob('*') if p.is_file() and p.suffix.lower() in suffixes and not any(t in p.name.lower() for t in ('gps','telemetry','credential','secret','.env','token'))]
    files+=logs+[p for p in EXP.glob('*.py')]+[p for p in EXP.glob('*.yaml')]
    # Preserve source paths; exclude all prior report backups.
    archive=ROOT/f'RoadVision_Phase13B_Run_Evidence_{rid}.zip'
    inventory={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            name=p.relative_to(ROOT).as_posix();data=p.read_bytes();inventory[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};z.writestr(name,data)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(inventory,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP integrity failure')
        for name,meta in inventory.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=meta['sha256']: raise RuntimeError(f'ZIP hash mismatch: {name}')
        print('ZIP MEMBERS:',len(z.namelist()))
    print('ZIP PATH:',archive.resolve());print('ZIP BYTES:',archive.stat().st_size);print('ZIP SHA256:',sha(archive));print('SCIENTIFIC REVIEW: PENDING. Inspect contents for private information before sharing.')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as e:
        print(f'STOP: {type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)
