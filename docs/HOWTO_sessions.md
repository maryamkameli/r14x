# Running long Claude sessions, structured requests, and agents on the RTX box

## 1. Long sessions on MSI without the 15-minute login-node CPU kill
Login nodes (ahl0x) kill any process past 15 min of CPU time; the Claude CLI itself accumulates that. Run it on a
compute node inside tmux, with the wrapper that already exists:
```bash
ssh agate.msi.umn.edu                                   # lands on a login node
/scratch.global/radke149/claude_compute.sh              # new session (tmux 'claude' -> srun msismall 4 cores, 4 days)
/scratch.global/radke149/claude_compute.sh --resume <session-id>   # continue an old conversation
# detach: Ctrl-b d      reattach later (any login node): tmux attach -t claude
```
tmux itself stays on the login node but uses almost no CPU; `claude` and everything it runs live on the compute node
(which can reach api.anthropic.com and run squeue/sbatch). If you log in to a different login node, the tmux
session is on the old one: `ssh ahl03` (or whichever) first, then `tmux attach -t claude`.

## 2. Writing requests in a structured way (with or without DSPy)
What costs time is ambiguity, not length. A request that states these fields up front needs no follow-up questions:
```markdown
GOAL:         one sentence, the outcome
INPUTS:       paths / datasets / models (exact names), what is already done
DEFINITIONS:  every term that could be read two ways ("p<1%" = teacher tokens with pre-update prob < 0.01; "core set" = ...)
CONSTRAINTS:  GPUs/partitions, deadlines, disk, what must NOT be touched
DELIVERABLES: files, where, format; what "done" means
DECISIONS:    choices you already made (so they are not asked), and "your call" for the rest
PRIORITY:     order if time runs out
```
Save it as `task.md` and start with `claude "$(cat task.md)"` (or paste it). DSPy can draft that spec from a
terse note, if you want it automated:
```python
import dspy  # pip install dspy
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-5-5"))      # needs ANTHROPIC_API_KEY
class TaskSpec(dspy.Signature):
    """Turn a terse research request into an unambiguous task spec for a coding agent."""
    request: str = dspy.InputField()
    context: str = dspy.InputField(desc="facts the agent will need: paths, models, prior runs")
    goal: str = dspy.OutputField(); definitions: list[str] = dspy.OutputField(desc="every ambiguous term, defined")
    constraints: list[str] = dspy.OutputField(); deliverables: list[str] = dspy.OutputField()
    open_questions: list[str] = dspy.OutputField(desc="what is still ambiguous; answer these before sending")
spec = dspy.Predict(TaskSpec)(request=open("note.txt").read(), context=open("context.md").read())
print(spec)   # answer spec.open_questions yourself, then send the spec to Claude
```
The useful part is `open_questions`: answering those before sending is what removes the round trips.

## 3. Driving the RTX box (cs-u-ding6000pro) with an agent
The box is reachable only from the UMN network/VPN, and MSI cannot ssh to it; files go box <- MSI with scp run
on the box. Two ways:

**A. Agent on the box (recommended: no laptop needs to stay online).**
```bash
ssh radke149@cs-u-ding6000pro.cs.umn.edu
tmux new -s claude
# Claude Code native install, kept off the small home disk:
mkdir -p /mnt/data/radke149/tools && cd /mnt/data/radke149/tools
curl -fsSL https://claude.ai/install.sh | bash          # then: claude  (log in once; /login)
cd /mnt/data/radke149/r1x4kit && claude "Read README.md and scripts/queue.txt, run scripts/run_queue.sh in tmux, watch the logs, fix failures."
```
Hermes Agent, pi or oh-my-pi install the same way (their own install commands) and run in another tmux window;
give each its API key through an environment variable in `~/.bashrc` (never in the repo). Reattach from anywhere on
the VPN: `ssh -t radke149@cs-u-ding6000pro.cs.umn.edu tmux attach -t claude`.

**B. Agent on the laptop, commands over ssh.** Set up key login and a persistent connection once:
```bash
ssh-keygen -t ed25519            # if you have no key
ssh-copy-id radke149@cs-u-ding6000pro.cs.umn.edu
cat >> ~/.ssh/config <<'EOF'
Host ding
  HostName cs-u-ding6000pro.cs.umn.edu
  User radke149
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
EOF
```
Then tell the laptop agent to run everything as `ssh ding '<command>'` and to launch long jobs inside tmux on the box
(`ssh ding "tmux new -d -s q 'bash /mnt/data/radke149/r1x4kit/scripts/run_queue.sh'"`), so a laptop sleep or
VPN drop does not kill the run. Mode A uses fewer tokens: the agent reads logs locally instead of over ssh round trips.
