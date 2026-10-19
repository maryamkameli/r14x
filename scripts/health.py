#!/usr/bin/env python3
"""Health check of the newest (or given) R1x4 training log. Prints one line per finding; exit 0 healthy, 1 warning, 2 error.

  python scripts/health.py [log]          # default: newest $R1X4_ROOT/logs/*.log
The overnight agent runs this every few minutes. Thresholds are deliberately loose; a WARN means look, not stop.
"""
import glob, math, os, re, sys, time

ROOT = os.environ.get('R1X4_ROOT', '/mnt/data/radke149/r1x4')
log = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(f'{ROOT}/logs/*.log'), key=os.path.getmtime, default=None)
if not log:
    print('ERROR no log found'); sys.exit(2)
txt = open(log, errors='replace').read()
steps = []
for m in re.finditer(r'step:(\d+) - (.*)', txt):
    d = {'step': int(m.group(1))}
    for k, v in re.findall(r'([\w/@.\-]+):(?:np\.float64\()?(-?[\d.eE+\-]+|nan|inf)\)?', m.group(2)):
        try:
            d[k] = float(v)
        except ValueError:
            pass
    steps.append(d)
worst, out = 0, []
def flag(level, msg):
    global worst
    worst = max(worst, level); out.append(('OK  ', 'WARN', 'ERROR')[level] + ' ' + msg)

age = time.time() - os.path.getmtime(log)
if re.search(r'Traceback|OutOfMemory|CUDA out of memory|NCCL error|RayTaskError|ActorDiedError', txt):
    tail = [l for l in txt.splitlines() if re.search(r'Error|error:', l)][-3:]
    flag(2, 'crash in log: ' + ' | '.join(x[-200:] for x in tail))
if 'training finished' in txt:
    flag(0, 'run finished')
elif age > 3600:
    flag(2, f'log silent for {age / 60:.0f} min (hang? check nvidia-smi / ray status)')
tr = [s for s in steps if s['step'] > 0 and 'train/student_acc' in s]
if not tr:
    flag(1 if age < 3600 else 2, f'no training step logged yet ({len(txt.splitlines())} log lines, {age / 60:.0f} min since last write)')
    print(f'log: {log}'); print('\n'.join(out)); sys.exit(worst)
last, first = tr[-1], tr[0]
g = lambda s, k, d=float('nan'): s.get(k, d)
flag(0, f"step {last['step']}: acc {g(last, 'train/student_acc'):.3f} len {g(last, 'train/student_len_mean'):.0f} "
        f"clip {g(last, 'train/student_clip_rate'):.2f} term {g(last, 'train/student_termination_rate'):.2f} "
        f"ent {g(last, 'train/student_entropy_pre_update', g(last, 'actor/entropy_pre_update')):.3f} gn {g(last, 'actor/grad_norm'):.3f} "
        f"lam {g(last, 'teacher/lambda', 1):.3f} step_s {g(last, 'timing_s/step'):.0f}")
for k in ('actor/grad_norm', 'actor/pg_loss', 'actor/entropy_loss'):
    if not math.isfinite(g(last, k, 0.0)):
        flag(2, f'{k} is {g(last, k)} (NaN/inf)')
if g(last, 'train/groups_dropped_label_mismatch', 0) > 0.1 * 48:
    flag(1, f"{g(last, 'train/groups_dropped_label_mismatch'):.0f} groups dropped for label mismatch (grader vs dataset)")
lam = g(last, 'teacher/lambda', 1.0)
if lam > 0 and 'train/adv_abs_mean' in last and abs(last['train/adv_abs_mean'] - 0.5) > 0.02 and 'teacher/lambda' not in last:
    flag(1, f"adv_abs_mean {last['train/adv_abs_mean']:.3f} != 0.5 in plain R1x4 (group assembly bug?)")
# dynamics relative to the first logged steps
w = tr[-5:]; w0 = tr[:3]
mean = lambda xs, k: sum(g(x, k, 0) for x in xs) / len(xs)
if len(tr) >= 8:
    e0, e1 = mean(w0, 'train/student_entropy_pre_update'), mean(w, 'train/student_entropy_pre_update')
    if e0 > 0 and e1 > 3 * e0:
        flag(1, f'student entropy x{e1 / e0:.1f} since the start ({e0:.3f} -> {e1:.3f}): watch for collapse into gibberish')
    t0, t1 = mean(w0, 'train/student_termination_rate'), mean(w, 'train/student_termination_rate')
    if t1 < t0 - 0.25:
        flag(1, f'termination rate {t0:.2f} -> {t1:.2f} (students stop finishing; length explosion?)')
    a0, a1 = mean(w0, 'train/student_acc'), mean(w, 'train/student_acc')
    if a1 < a0 - 0.15:
        flag(1, f'student train accuracy {a0:.3f} -> {a1:.3f}')
    if mean(w, 'train/student_clip_rate') > 0.8:
        flag(1, 'over 80% of student rollouts hit the 8192-token cap')
dg = [s['diag/cos'] for s in tr if 'diag/cos' in s]
if len(dg) >= 10 and all(c < 0 for c in dg[-10:]):
    flag(1, 'diag/cos < 0 for the last 10 diagnostics: teacher gradient opposes the student gradient (recipe: shorten T_a)')
if 'is/token_ess' in last and last['is/token_ess'] < 0.05:
    flag(1, f"IS token ESS {last['is/token_ess']:.3f}: ratios degenerate (check mu alignment / teacher choice)")
if 'is/rows_with_mu' in last and last['is/rows_with_mu'] == 0 and lam > 0:
    flag(1, 'variant IV in the teacher phase but no row has a mu-trace (wrong --mu_pool?)')
vals = [(s['step'], s['val/avg_score']) for s in steps if 'val/avg_score' in s]
if vals:
    flag(0, 'val/avg_score: ' + ', '.join(f'{st}:{v:.4f}' for st, v in vals[-6:]))
    best = max(v for _, v in vals)
    if len(vals) >= 3 and vals[-1][1] < best - 0.05:
        flag(1, f'held-out avg {vals[-1][1]:.4f} is >0.05 below the best {best:.4f}')
print(f'log: {log}')
print('\n'.join(out))
sys.exit(worst)
