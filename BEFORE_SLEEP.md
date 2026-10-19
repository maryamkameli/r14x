# Before you sleep (≈15 min, on the UMN VPN)

```bash
# 1. log in (first login asks you to change the default password)
ssh radke149@cs-u-ding6000pro.cs.umn.edu

# 2. kit + data onto the fast disk (pull FROM MSI; MSI cannot reach the box). Asks for your MSI password + Duo.
mkdir -p /mnt/data/radke149 && cd /mnt/data/radke149
scp radke149@agate.msi.umn.edu:/scratch.global/radke149/exports/r1x4kit_1p7b_4b_LATEST.zip .   # exact name: see the chat
unzip -q r1x4kit_1p7b_4b_*.zip && cd r1x4kit_1p7b_4b

# 3. an agent CLI, logged in once (interactive login now, so it can run headless at night)
curl -fsSL https://claude.ai/install.sh | bash && exec bash -l    # Claude Code; then:  claude   -> /login, then /exit
claude -p "reply with OK" --dangerously-skip-permissions          # must print OK
#   or Codex instead:  npm i -g @openai/codex && codex login && codex exec "reply with OK"   (then AGENT=codex below)

# 4. arm it: waits for 00:00, sets up the env, generates the IS teacher pool, starts the queue, keeps the agent watching
tmux new -s night 'AGENT=claude bash scripts/overnight_agent.sh'
#   detach: Ctrl-b d   (laptop can sleep; tmux keeps running on the box)
```

In the morning: `ssh` in, `tmux attach -t night`, and read `/mnt/data/radke149/r1x4/agent/STATUS.md` and `JOURNAL.md`.
Health at any time: `source scripts/rtx_env.sh && python scripts/health.py`.

What runs (scripts/queue_overnight.txt, in order, each on GPUs 0-3):
1. Qwen3-1.7B variant IV: p<1% + truncated per-token IS on Qwen3-4B traces + teacher weight annealed to 0 at
   step 60 of 300, then GRPO-4 (48 prompts/step).
2. The same annealing without IS (R1 traces), then variants III, II, I.

Timeline after 00:00: env setup + kernel tests ≈ 20-40 min, IS teacher pool ≈ 1-2 h, then training.
