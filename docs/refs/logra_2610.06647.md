URL: https://arxiv.org/pdf/2610.06647
Content-Type: text/markdown
Method: arxiv
Notes: Fetching PDF for full content...; PDF converted via markit

---

# LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches

**Authors:** Shaokun Zhang, Yifan Zhang, Jian Hu, Yueying Li, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong
**Published:** 2026-10-05
**Categories:** cs.CL
**arXiv:** 2610.06647

---

## Abstract

Reinforcement learning has greatly advanced the capabilities of large language models, but its memory demands remain a barrier to broader adoption. We introduce LoGRA, an approach to RL post-training that reduces memory by retaining useful learning signals in low-rank gradient sketches. These compact representations support both model updates and efficient policy synchronization. To prevent overly large updates from disrupting learning, we complement gradient compression with predicted-KL step control, which estimates policy changes before applying each update and adjusts its magnitude accordingly. With all techniques combined, LoGRA reduces average training memory usage by up to 45.7% across reasoning tasks without compromising performance. It also enables stable training of a 27B-parameter model for over 1,100 steps on a single eight-GPU node, where dense Adam runs out of memory, making previously memory-infeasible RL training practical. Code is available in the \href{https://github.com/skzhang1/labs-molt/tree/logra/examples/scripts/logra}{Molt library}.

---

## Full Paper

<!-- Page 1 -->

## LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches

**Shaokun Zhang, Yifan Zhang, Jian Hu, Yueying Li, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong** **NVIDIA**

### Abstract

**Reinforcement learning has greatly advanced the capabilities of large language models, but its memory** **demands remain a barrier to broader adoption. We introduce LoGRA, an approach to RL post-training** **that reduces memory by retaining useful learning signals in low-rank gradient sketches. These compact** **representations support both model updates and efficient policy synchronization. To prevent overly large** **updates from disrupting learning, we complement gradient compression with predicted-KL step control,** **which estimates policy changes before applying each update and adjusts its magnitude accordingly.** **With all techniques combined, LoGRA reduces average training memory usage by up to 45.7% across** **reasoning tasks without compromising performance. It also enables stable training of a 27B-parameter** **model for over 1,100 steps on a single eight-GPU node, where dense Adam runs out of memory, making** **previously memory-infeasible RL training practical. Code is available in the Molt library.**

### 1. Introduction

1.5B 7B 27B Dense Adam LoGRA
Dark: average · Pale: peak extension GPU capacity: 79.2 GiB

9.18 / 9.19
7.18 / 8.66
21.8% saving
17.29 / 22.26
45.7% saving
**OOM**

51.54 / 53.88
0 10

|20|30|50|60|70|80|
|---|---|---|---|---|---|
|Training memory (GiB / GPU)||||||

# arXiv:2610.06647v3 [cs.CL] 17 Oct 2026

Figure 1: **Lower training memory across model scales.** Colors denote model sizes; hatching distinguishes

Dense Adam from LoGRA. Dark bars show average memory and pale extensions reach peak memory (Table 1); labels report average / peak in GiB per training GPU. Arrows show average-memory savings over Dense Adam. The broken axis omits 34–48 GiB. The 27B Dense Adam bar marks OOM at GPU capacity, not a measured value.

Reinforcement learning (RL) has emerged as a key driver of advances in LLM, enabling models to improve through feedback on their own generated responses (Sheng et al., 2024; nem, 2025; Xu et al., 2026; Zhang et al., 2026b). However, its substantial memory requirements remain a major barrier to broader adoption. This challenge extends well beyond storing model parameters, as gradients and optimizer states can consume several times more memory than the weights themselves. For example, the BF16 weights of a 7B-parameter Qwen model (Bai et al., 2023) require approximately 14 GB, while Adam’s (Kingma & Ba, 2014) two FP32 moment

© 2026 NVIDIA. All rights reserved.

<!-- Page 2 -->

<u>LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches</u>

buffers add another 56 GB, even before accounting for gradients and activations. Consequently, hardware capable of running a model may still be insufficient to improve it through RL. Reducing this memory overhead is therefore critical to making RL post-training accessible to the broader research community.

To overcome this bottleneck, existing approaches span both system-level optimizations and memory-efficient training algorithms. At the system level, FSDP (Zhao et al., 2023) shards training states across GPUs, gradient checkpointing (Chen et al., 2016) reduces activation storage through recomputation, and CPU offloading moves selected training states to host memory (Rajbhandari et al., 2020). Parameter-efficient methods reduce the number of parameters that require gradients and optimizer states: LoRA (Hu et al., 2021), for example, freezes pretrained weights and learns small low-rank adapters. Gradient compression offers another route (Zhao et al., 2024; Su et al., 2025; Hao et al., 2024): they keep the original weight matrices trainable while using compact representations of gradients and optimizer states, thereby reducing training memory while still enabling effective updates to the full model.

Motivated by this perspective, we introduce LoGRA, a method that uses low-rank gradient sketches to reduce the memory and communication costs of RL post-training, thus lower the hardware barrier. Our intuition is that learning from rewards may require far less information than a full gradient can represent: with binary outcome rewards, each generated response receives only a single correctness signal, motivating a compact representation of the resulting learning signal (Schulman & Lab, 2025). LoGRA directly accumulates sketches during backpropagation, avoiding the construction and storage of full gradients for the selected weight matrices. These sketches produce updates that are merged into the model weights, with directions that can vary throughout training. The same compact representation is then reused to synchronize the rollout policy, extending the benefits of compression from training memory to communication.

LoGRA alone is insufficient to ensure stable updates: an overly large update can abruptly shift the policy and disrupt previously learned behavior. We therefore introduce predicted-KL step control, which estimates how much a proposed update would change the model’s next-token probabilities before applying it. This estimate guides the update scale to stay within a prescribed KL budget with simple optimizers.

We evaluate LoGRA across multiple reasoning tasks with models ranging from 1.5B to 27B parameters. LoGRA reduces average training memory by 21.8% at 1.5B and 45.7% at 7B without sacrificing task performance. These savings also make larger-scale training feasible: LoGRA sustains stable learning of a 27B model for over 1,100 steps on a single eight-GPU node, where the dense Adam baseline runs out of memory under the same hardware allocation. Ablations further clarify how projection design and KL budgets affect learning performance and training stability.

### 2. Method

An LLM contains large weight matrices, and storing a full gradient for each matrix adds substantial training memory. LoGRA replaces these full gradients with compact sketches and represents each weight update as the product of two thin matrices. We first describe this matrix representation, then explain its use in RL post-training and policy synchronization. We use predicted-KL step control to determine the update magnitude.

#### 2.1. Low-rank gradient sketches

##### Full-gradient updates.

Let *𝜃* denote the parameters of an LLM policy *𝜋<sub>𝜃</sub>*, and let *ℒ*(*𝜃*) be a differentiable RL training loss. Consider one weight matrix *𝑊 ∈* R *𝑑×𝑘* in the model. Its gradient *𝐺* has the same dimensions as *𝑊*; for example, an SGD update takes the form *𝐺* = *∇<sub>𝑊</sub> ℒ∈* R *𝑑×𝑘* *, 𝑊 ← 𝑊 − 𝜂𝐺,* (1)

where *𝜂* is the learning rate. When a training batch is split into microbatches, standard training accumulates their contributions in a full *𝑑 × 𝑘* gradient buffer before updating *𝑊*. This requires *𝑑𝑘* stored values in addition

<!-- Page 3 -->

to the weights and any optimizer state. Our goal is to perform the update using a much smaller gradient representation.

##### A compact representation.

Choose a rank *𝑟 ≪* min(*𝑑,𝑘*) and a random projection matrix *𝐴 ∈* R *𝑟×𝑘*. LoGRA represents the gradient by a sketch *𝑆* = *𝐺𝐴* *⊤* *∈* R *𝑑×𝑟*

*.* (2)
Multiplication by *𝐴* *⊤* compresses the *𝑘* columns of *𝐺* into *𝑟* columns of *𝑆*. The sketch therefore stores *𝑑𝑟* values instead of *𝑑𝑘*. The projection *𝐴* is sampled independently of the current batch and held fixed throughout *√* gradient accumulation; it is not learned. By default, its entries are independent random signs scaled by 1*/ 𝑟*.

Equation 2 defines the sketch mathematically, but does not require computing and then compressing a full gradient. Our implementation directly accumulates the projected gradient contributions from each microbatch into *𝑆* in fp32. It does so by projecting the inputs used in the weight-gradient calculation before multiplying them by the backward derivatives. Thus, the compressed weights do not require a full gradient buffer, while the model’s ordinary forward computation is preserved.

##### A low-rank weight update.

To update *𝑊*, we need a matrix with the same dimensions as *𝑊*. The sketch *𝑆* has only *𝑟* columns, so we multiply it by *𝐴* to obtain a *𝑑 × 𝑘* approximation to the full gradient:

*𝐺* ̂︀ = *𝑆𝐴 ∈* R*𝑑×𝑘, 𝑊 ← 𝑊 − 𝜂𝑆𝐴.* (3)

In other words, we use *𝑆𝐴* in place of *𝐺* in the standard gradient update. Although *𝑆𝐴* has the same dimensions as *𝐺*, it is represented by two small matrices: *𝑆* of size *𝑑 × 𝑟* and *𝐴* of size *𝑟 × 𝑘*. Because *𝑆* has only *𝑟* columns, every column of *𝑆𝐴* is a combination of these *𝑟* columns; this is why the update has rank at most *𝑟*. This compact representation approximates the full gradient rather than recovering it exactly. The resulting update is added directly to *𝑊*, which remains a full-sized weight matrix.

An optimizer can also adjust the sketch before it is used in the update. We call this adjusted sketch *𝑈*; it has the same dimensions as *𝑆*, and *𝑈* = *𝑆* for the basic SGD variant. The update then becomes

##### 𝑊 ← 𝑊 − 𝛼𝜂𝑈𝐴. (4)

Here *𝜂* sets the nominal step size, and *𝛼* is an additional multiplier: *𝛼* = 1 leaves the step unchanged, while a smaller value reduces it. The predicted-KL controller described below chooses *𝛼* without changing the shapes of *𝑈* and *𝐴*.

#### 2.2. Applying LoGRA to RL post-training

We apply the sketch to attention and MLP projection matrices, holding other parameters fixed. Algorithm 1 summarizes the procedure, which supports differentiable RL losses. Each matrix has its own *𝐴*, *𝑆*, and *𝑈*, while *𝛼* is shared. Updates are merged directly into the weights, with no persistent adapter branch. Refreshing *𝐴* each step allows successive updates to use different subspaces, so their sum need not have rank at most *𝑟*. Here *ℒ<sub>𝑚</sub>* denotes microbatch *𝑚*’s contribution to *ℒ*. The projected gradient in the algorithm is accumulated directly, without forming the full gradient matrix.

##### Memory and synchronization.

The gradient accumulator shrinks from *𝑑𝑘* to *𝑑𝑟* values: for *𝑑* = *𝑘* = 4096 and *𝑟* = 64, fp32 storage falls from 64MiB to 1MiB. Model weights, cached projections, activations, and other buffers still consume memory. Basic SGD requires no optimizer moments; stateful sketch optimizers add their own storage. For synchronization, the rollout engine regenerates *𝐴* from its seed and applies the received factor *𝛼𝜂𝑈*. The payload is *𝑂*(*𝑑𝑟*) values plus a seed and metadata, instead of a full *𝑑 × 𝑘* update.

<!-- Page 4 -->

**Algorithm 1** LoGRA with projection refreshing **Require:** Policy *𝜋<sub>𝜃</sub>*, selected matrices *𝑊*, rank *𝑟*, step size *𝜂* 1: Initialize rollout weights from *𝜃*. 2: **for** each training step **do** 3: Sample responses from *𝜋<sub>𝜃</sub>*; construct *ℒ*. *◁* **1. Prepare** 4: **for** each *𝑊* **do** 5: Sample *𝐴 ∈* R *𝑟×𝑘* from a fresh seed. 6: *𝑆 ←* 0 *∈* R *𝑑×𝑟* 7: **for** each microbatch *𝑚* (*𝜃* and *𝐴* fixed) **do** *◁* **2. Accumulate sketches** 8: **for** each *𝑊* **do** 9: *𝑆←𝑆* + (*∇<sub>𝑊</sub> ℒ<sub>𝑚</sub>*)*𝐴* *⊤* *◁* Direct sketch accumulation 10: For each *𝑊* : *𝑈←𝑆* (or use a sketch optimizer). *◁* **3. Choose the update** 11: *𝛼 ←* 1 *◁* Or use predicted KL; Section 2.3 12: **for** each *𝑊* **do** *◁* **4. Update and synchronize** 13: *𝑊 ← 𝑊 − 𝛼𝜂𝑈𝐴 ◁* Trainer 14: Send *𝛼𝜂𝑈* and the seed for *𝐴* to rollout. 15: Regenerate *𝐴*; apply the same update. *◁* Rollout 16: Wait for synchronization to finish.

#### 2.3. Step-size control with predicted KL

LoGRA specify how to change the weights, but the size of that change also matters. An update that is small in parameter space can still substantially change the policy’s output probabilities. Following the trust-region motivation of TRPO (Schulman et al., 2015), we therefore control the update magnitude using KL divergence between the current and proposed policies.

We estimate how much the proposed update would change the model’s next-token probabilities, then scale the update to fit a prescribed budget. Let *𝐷* denote the proposed update to all parameters; its component for each compressed weight matrix *𝑊* is *𝐷𝑊* = *−𝜂𝑈𝐴*, using the factors and nominal step size defined above. We apply *𝜃 ← 𝜃* + *𝛼𝐷*, where *𝛼* controls the step size without changing its direction.

We select a small subset of responses from the current training batch and examine the predictions used to generate their tokens. For each prediction, the model sees *𝑠* which is a prompt and the response generated so far. Let *𝒫* collect these contexts. At each context, we compare the current model’s probabilities over the vocabulary, *𝜋<sub>𝜃</sub>*(*·| 𝑠*), with those of the proposed updated model, *𝜋<sub>𝜃+𝛼𝐷</sub>*(*·| 𝑠*). Both are evaluated on the same text context and at the same temperature *𝑇 >* 0: a higher temperature gives a flatter distribution.

KL divergence measures how much these two probability distributions differ. We average this change over the examined predictions as derived in detail in the appendix:

<u>1</u> ∑︁2 *𝐷<sub>KL</sub>*(*𝜋<sub>𝜃</sub>*(*·| 𝑠*)*‖ 𝜋<sub>𝜃+𝛼𝐷</sub>*(*·| 𝑠*)) *≈ 𝛼 𝑞*(*𝐷*)*.* (5) *|𝒫|* *𝑠∈𝒫*

The side is the average KL between the current and proposed policies over the examined contexts. On the right, *𝑞*(*𝐷*) is our estimate of this average change for the unscaled proposal (*𝛼* = 1). The approximation predicts a quadratic dependence on step size: halving *𝛼* reduces the predicted KL by a factor of four.

We specify a KL budget *𝛿 >* 0, the allowed average predicted change in next-token probabilities for one training step. A smaller budget keeps the updated policy closer to the current one. Equation 5 gives a way to control this change: rescale the proposed update by *𝛼* so that *𝛼²𝑞*(*𝐷*) *≤ 𝛿*. Thus, we first estimate the proposal’s predicted KL *𝑞*(*𝐷*), then choose its scale to fit the budget.

<!-- Page 5 -->

To estimate *𝑞*(*𝐷*), we examine how the proposal would change the scores assigned to candidate next tokens. Let *𝑣* index vocabulary tokens, *𝑧<sub>𝑣</sub>*(*𝑠*) be the score (logit) of token *𝑣* at context *𝑠*, and *𝑧*(*𝑠*) collect these scores. The vector *𝑝*(*𝑠*) = softmax(*𝑧*(*𝑠*)*/𝑇*) contains their probabilities at temperature *𝑇*. Let *𝑧*˙<sub>𝑣</sub>(*𝑠*) be the rate at which score *𝑧<sub>𝑣</sub>*(*𝑠*) changes along *𝐷*, evaluated at the current weights. The predicted KL is

<u>1</u> ∑︁ *𝑞*(*𝐷*) = Var*𝑣∼𝑝<sub>(𝑠)</sub>*[ ˙*𝑧<sub>𝑣</sub>*(*𝑠*)]*.* (6) 2*|𝒫|𝑇²* *𝑠∈𝒫*

Derived could be found in the appendix. Here the variance measures how differently the token scores change, weighted by their current probabilities *𝑝*(*𝑠*), and *|𝒫|* is the number of examined contexts.

##### Rescale the update to fit the budget.

Once *𝑞*(*𝐷*) is available, the ratio *𝛿/𝑞*(*𝐷*) tells us how the budget compares with the proposed change. Because predicted KL scales with *𝛼²*, we take the square root of this ratio. With a chosen upper limit *𝛼<sub>max</sub> >* 0 on the update multiplier, we set {︁ √︀}︁ *𝛼* = min *𝛼<sub>max</sub>, 𝛿/𝑞*(*𝐷*)*,* (7)

for finite positive *𝑞*(*𝐷*). For example, a proposal with *𝑞*(*𝐷*) = 4*𝛿* is scaled to at most one half of its original size. The cap prevents excessive enlargement when the predicted KL is small. This rule controls the predicted change; the realized KL can differ because the estimate is a local approximation. A constant budget keeps *𝛿* fixed, while an annealed budget gradually lowers it to permit less policy change later in training.

### 3. Experiments

We conduct experiments to demonstrate that: (1) LoGRA substantially improves training efficiency without compromising performance; (2) predicted-KL enables stable single-node RL training for more than 1,000 steps; and (3) ablation analyses provide deeper understanding of our method.

#### 3.1. General Experimental Setup

We primarily evaluate Qwen models ranging from 1.5B to 27B parameters, including Qwen2.5-Math-1.5B/7B and Qwen3.8-27B, on mathematical reasoning tasks. We also investigate long-horizon training of the 27B model on the Reasoning-Gym Hard mixture. Unless otherwise specified, we use PPO with verifiable rewards, sampling one response per prompt and using a global running reward baseline for advantage estimation. This deliberately simple setup isolates the effects of our method from potential gains introduced by more sophisticated RL algorithms. By default, all experiments run on a single node equipped with eight H100 80GB GPUs and 128 CPU cores, with four GPUs allocated to FSDP training and the remaining four to independent generation engines.

#### 3.2. Main Experiments

We primarily compare LoGRA with dense-gradient training on DAPO-Math-7.5K. We evaluate on MATH-500 every five updates for the 1.5B and 7B models and every ten updates for the 27B model. Each batch contains 128 prompts, with one response per prompt, a 4,096-token context limit, and one policy update using a global reward baseline. LoGRA uses rank-256 RowAdam with mismatch subtraction and a predicted-KL budget annealed from 2 *×* 10<sup>−4</sup> to 2 *×* 10<sup>−5</sup>, whereas the dense baseline uses Adam with learning-rate annealing. We run three runs for 20,000 updates at 1.5B and 7B and 200 updates at 27B, retaining the 300-update annealing schedule for the latter.

##### Learning performance.

At 1.5B, LoGRA achieves peak Pass@1 of 67*.*87%, compared with 63*.*77% for Dense; peak Pass@4 increases from 80*.*33% to 82*.*47%. At 7B, the methods attain similar peak accuracy: 72*.*33% versus 72*.*48% Pass@1 and

85*.*13% versus 85*.*33% Pass@4 for LoGRA and Dense, respectively. LoGRA also trains the 27B model successfully,

<!-- Page 6 -->

||(a) Pass@1|||(b) Pass@4|
|---|---|---|---|---|
|70%|||84%||
|60%|||78%||
|50%|||72%||
||1.5B LoGRA|1.5B Dense|||
|40%|7B LoGRA|7B Dense|66%||
||27B LoGRA|27B Dense — OOM|||
||||60%||
|0|50|100 150|200 0|50|
||Optimizer update|||Optimizer update|

MATH-500 success rate

100 150 200

Figure 2: **Learning performance across model scales.** MATH-500 accuracy over the first 200 updates. Curves

and bands show means and *±* one sample standard deviation over three seeds.

##### (a) 1.5B (b) 7B (c) 27B

10.5 LoGRA: 21.9% less memory LoGRA: 45.6% less memory
Dense: OOM 32 80 GPU capacity ≈ 79.2 GiB

9.0
70 24

7.5
60 16 LoGRA ≈ 51.54 GiB

6.0 Dense ≈ 9.18 GiB Dense ≈ 31.81 GiB 50
Training memory (GiB / GPU) 0 50 100 150 200 0 50 100 150 200 0 50 100 150 200 Optimizer update Optimizer update Optimizer update

Figure 3: **Training memory across model scales.** Per-update peak allocated memory, averaged over three

seeds, with *±* one sample standard deviation. Arrows compare temporal means over updates 1–200 at 1.5B/7B; at 27B, the arrow shows headroom to device capacity.

reaching 71*.*52% Pass@1 and 81*.*87% Pass@4, whereas Dense exhausts GPU memory at its first Adam-state allocation. These results show that compressed updates preserve competitive learning performance in the evaluated configurations while making the largest model trainable under the same GPU budget.

##### Memory and throughput.

Figure 3 shows that LoGRA lowers average per-update memory at 1.5B and 7B. Over the complete runs, average

memory decreases from 9.18 to 7.18 GiB at 1.5B and from 31.82 to 17.29 GiB at 7B, reductions of 21.8% and 45.7%, respectively. Under the seed-averaged peak definition, LoGRA reduces peak memory from 9.19 to

8.66 GiB at 1.5B and from 31.85 to 22.26 GiB at 7B. At 27B, LoGRA uses 51.54 GiB on average and reaches a peak seed-averaged memory of 53.88 GiB; Dense fails before completing an update, precluding a measured memory-reduction ratio or throughput comparison. End-to-end update rates are similar where both methods complete training: 55.44 versus 54.28 updates/h at 1.5B and 25.73 versus 25.28 updates/h at 7B for LoGRA and Dense, respectively. The main systems benefit is reduced memory use and feasibility at 27B, with modest observed throughput gains at 1.5B and 7B.

<!-- Page 7 -->

Table 1: **Accuracy, memory, and throughput of the main experiments.** Memory and throughput cover full
 runs; accuracy peaks use updates 0–200. Peak memory is the maximum of the three-seed mean curve.
Average memory *↓* Peak memory *↓* Updates/h *↑* Peak Pass@1 *↑* Peak Pass@4 *↑* Model Method *𝑛* (GiB/GPU) (GiB/GPU) (%) (%)

|1.5B|Dense|3|9. 18 ± 0. 00|9. 19 ± 0. 00|54. 28 ± 0. 40|63. 77 ± 1. 35|80. 33 ± 1. 47|
|---|---|---|---|---|---|---|---|
|1.5B|LoGRA|3|7. 18 ± 0. 10|8. 66 ± 1. 62|55. 44 ± 0. 41|67. 87 ± 1. 23|82. 47 ± 1. 10|

|7B|Dense|3|31. 82 ± 0. 00|31. 85 ± 0. 00|25. 28 ± 0. 68|72. 48 ± 0. 35|85. 33 ± 0. 83|
|---|---|---|---|---|---|---|---|
|7B|LoGRA|3|17. 29 ± 0. 34|22. 26 ± 4. 98|25. 73 ± 0. 28|72. 33 ± 1. 33|85. 13 ± 1. 14|

|27B|Dense|0|||OOM|||
|---|---|---|---|---|---|---|---|
|27B|LoGRA|3|51. 54 ± 0. 17|53. 88 ± 0. 00|6. 99 ± 0. 02|71. 52 ± 0. 94|81. 87 ± 0. 81|

**(b) Response length**

|(a) Held-out task score||||||(c) Truncation rate|
|---|---|---|---|---|---|---|
|70||62.9%|10|8.31k → 4.71k|40|34.7% → 13.0%|
|60 fi|||8||30||
|50|||6||20||
|40|39.7% at start||4||10||
|0|400|800|0|400|0|400|
||Training step|||Training step||Training step|

er score (%)

(k tokens)

Mean response length Macro veri Truncated responses (%) 1200 800 1200 800 1200

Figure 4: **Sustained 27B training on a single eight-GPU node.** Qwen3.8-27B trained with LoGRA on the

Reasoning-Gym Hard mixture. **(a)** Macro-averaged verifier score across held-out tasks. **(b)** Mean evaluation response length. **(c)** Fraction of evaluation responses truncated at the 16,384-token context limit. All panels describe one training trajectory.

#### 3.3. Sustained 27B training on a single eight-GPU node

We demonstrate that LoGRA with predicted-KL enables stable 27B model training for over 1,000 steps on a single node. Specifically, we train Qwen3.8-27B on the Reasoning-Gym Hard mixture. For efficiency purpose, we estimate (q(D)) using a finite-difference probe based on direct KL evaluations, rather than the JVP-based estimator in Equation 6. The dataset contains 38,948 training prompts and 193 hard held-out problems across eight tasks. We use rank-256 sketches and anneal the predicted-KL budget. Every 20 steps, we evaluate the model using the macro-averaged verifier score and monitor response length and truncation rate.

The held-out macro score increases from 39*.*69% initially to 62*.*94%, with a peak of 65*.*52% at step 1,060 (Figure 4a). The final five observed evaluations range from 59*.*11% to 64*.*16%, showing no sustained collapse. Over the same trajectory, mean response length decreases from 8,312 to 4,705 tokens, while the truncation rate falls from 34*.*72% to 12*.*95% (Figure 4b,c). This trajectory demonstrates that the LoGRA recipe can sustain learning of a 27B model beyond 1,100 logged training steps on one eight-GPU node, without sustained deterioration in the observed evaluation scores or runaway response-length growth.

#### 3.4. Further Analysis

##### 3.4.1. Design Choices in Low-Rank Gradient Compression

We examine the effects of projection rank, basis refreshing, and projection distribution using Qwen2.5-Math-

1.5B on DAPO-Math-7.5k with ranks *𝑟 ∈* 4*,*16*,*64*,*256. Fixed Rademacher projections sample matrices once

<!-- Page 8 -->

##### (a) Training performance (b) Basis refreshing (c) Entry distribution

**68.50%**
0.4
**68.31%**7 7
0.3
68

**67.42%**
14 14 0.2 66 Layer Layer 21 21 0.1 Gradient cosine 64Fixed Rademacher Refreshed Rademacher MATH-500 pass@1 (%)Refreshed Gaussian 28 28 0.0 4 16 64 256 41664 41664 41664 41664 Projection rank *r* 256 256 256 256 Fixed Refreshed Rademacher Gaussian

Figure 5: **Components of gradient compression. (a)** Training performance as a function of projection rank;

**(b)** Gradient-reconstruction cosine for fixed and refreshed Rademacher bases. **(c)** The same diagnostic for refreshed Rademacher and Gaussian bases. Heatmap rows denote Transformer layers and columns denote ranks, with a shared [0*,*0*.*4] color scale; higher values indicate closer alignment with the dense gradient.
*√* with independent entries *±*1*/ 𝑟*, whereas refreshed Rademacher and Gaussian projections resample at every update. Following Flora’s motivation of avoiding confinement to a fixed low-rank subspace (Hao et al., 2024), we test whether refreshing benefits RL post-training. Each point represents one run, with MATH-500 pass@1 averaged over eight evaluations between steps 200 and 235. Figure **??**(a) shows that rank has the strongest effect: increasing it from 4 to 256 improves pass@1 by 3.02–3.95 percentage points across configurations. At rank 256, fixed Rademacher, refreshed Rademacher, and refreshed Gaussian achieve 68*.*50%, 68*.*31%, and

67*.*42%, respectively. Their ordering varies across ranks, showing no consistent benefit from refreshing, possibly because a fixed low-rank subspace captures sufficient task-relevant signal over this short training horizon. Figures 5(b) and 5(c) evaluate gradient reconstruction quality. We use the initialization and checkpoints at steps 160 and 280 from one fixed-basis rank-64 training trajectory. At each model state, we sample one response for each of 4,096 prompts and divide them into 32 minibatches of 128 responses. All projection conditions use the same responses and dense gradients. Fixed bases are reused across minibatches, whereas refreshed bases are redrawn for each minibatch. For each rank, we evaluate eight projection seeds and compute the Frobenius cosine similarity between the dense gradient *𝐺* and its reconstruction for all 196 attention and MLP weight matrices. Each heatmap cell is averaged over the seven matrices in each Transformer layer, minibatches, model states, and projection seeds. Figure 5(b) compares fixed and refreshed Rademacher projections, whereas
Figure 5(c) compares refreshed Rademacher and Gaussian projections. Reconstruction quality exhibits a strong
 dependence on rank: the mean cosine similarity increases from approximately 0*.*046 at rank 4 to 0*.*348 at rank
256. By contrast, the largest difference between corresponding layer–rank cells is only 0*.*0014 for fixed versus refreshed Rademacher projections and 0*.*0008 for refreshed Rademacher versus Gaussian projections. Although the decrease in mean cosine under Gaussian projections is detectable at ranks 64 and 256, its magnitude remains below 2*×*10<sup>−4</sup>. Overall, projection rank has a substantially greater effect on single-minibatch gradient reconstruction than either basis refreshing or projection distribution.
##### 3.4.2. Comparison with LoRA

LoRA learns low-rank weight adapters, whereas LoGRA compresses gradients and updates to the model weights. We compare their memory use, accuracy, and update time. We compare rank-256 configurations of Qwen2.5-Math-1.5B trained on DAPO-Math-7.5k. For this comparison, we use the first 300 training steps of each method. LoRA uses AdamW with a constant learning rate of 2 *×* 10<sup>−4</sup>; LoGRA uses a predicted-KL budget annealed from 2 *×* 10<sup>−4</sup> to 2 *×* 10<sup>−5</sup>.

<!-- Page 9 -->

Table 2: **LoRA and LoGRA on Qwen2.5-Math-1.5B.** Values are mean *±* sample standard deviation over three

seeds.

Average memory *↓* Peak memory *↓* Pass@1 *↑* Pass@4 *↑* Seconds/update *↓* Method (GiB) (GiB) (%) (%) (s)

LoRA 13*.*21 *±* 0*.*07 13*.*38 *±* 0*.*00 70*.*45 *±* 3*.*09 81*.*40 *±* 2*.*09 10*.*63 *±* 0*.*10 LoGRA 7*.*18 *±* 0*.*10 8*.*66 *±* 1*.*62 68*.*47 *±* 0*.*95 81*.*80 *±* 1*.*06 11*.*31 *±* 0*.*09

Average memory is the temporal mean of the three-seed mean per-update memory curve; peak memory is its maximum. The per-step measurements are peak allocated training memory, excluding generation memory. Seconds/update measures the policy-training call, including call overhead but excluding response generation and weight synchronization.

Table 2 shows an obvious memory advantage for LoGRA under the recorded recipes, with average and peak

memory of 7.18 and 8.66 GiB, compared with 13.21 and 13.38 GiB for LoRA. LoRA attains higher mean Pass@1 and a shorter policy-training call, while LoGRA attains a 0.40-percentage-point higher mean Pass@4. Thus, these results motivate gradient compression as an alternative when training memory is the limiting resource, but do not establish an accuracy or speed advantage.

##### 3.4.3. KL Budget Control

No predicted KL control Constant 1.5 × 10<sup>−3</sup> Annealed Constant 1.5 × 10<sup>−4</sup>

(a) Long-horizon accuracy (b) Response-length growth
70 1400 65 1200 60

55 1000 Performance degradation 50 800 MATH-500 pass@1 (%) 45 Mean response length (tokens) 600 0 200 400 600 800 0 200 400 600 800 Training step (logged) Training step (logged)

Figure 6: **Long-horizon accuracy and response length under different KL controls.** Thick solid lines show

two-seed means, thin lines individual runs, and shading their range. Curves use five-evaluation moving means. We then study whether predicted-KL step control improves long-horizon learning and how constant versus annealed KL budgets affect accuracy and response length. We use Qwen2.5-Math-1.5B on DAPO-Math-7.5k with rank-256 Rademacher projections refreshed every update, SGD, and a global running reward baseline. We compare a fixed relative update norm of 6 *×* 10<sup>−5</sup> without KL control, constant KL budgets of 1*.*5 *×* 10<sup>−3</sup> and 1*.*5 *×* 10<sup>−4</sup>, and a cosine-annealed budget from 2 *×* 10<sup>−4</sup> to 2 *×* 10<sup>−5</sup> over 1,500 steps. Each setting has two seeds; these KL-controlled settings do not subtract mismatch or use realised-KL feedback. Figure 6 plots MATH-500 pass@1 and response length over the first 800 logged steps, evaluating four responses per problem every five steps. Averaging raw evaluations over steps 700–800 within each run and then across seeds, pass@1 is 70*.*30% without KL control, 63*.*43% with the large constant budget, 71*.*03% with the small constant budget, and 71*.*47% with annealing. Corresponding response lengths are 774, 976, 858, and 832 tokens.

<!-- Page 10 -->

These results support the usefulness of predicted-KL step control with a conservative budget. The small constant budget improves tail accuracy over no control while maintaining stable learning. This is consistent with the trust-region motivation of TRPO (Schulman et al., 2015): limiting policy changes helps ensure that improvements predicted by a local objective translate into better performance, whereas overly large updates can undermine this approximation and destabilize training. In our experiments, the large constant budget exhibits performance degradation, while the small and annealed budgets remain stable, with annealing achieving the highest mean tail accuracy. The no-control baseline also remains stable, and the separately collected annealed runs differ in their initial budget, so these two-seed results suggest a benefit from well-calibrated KL control without establishing that it is necessary for stability or that annealing alone explains the gain.

### 4. Related Works

Large language models have emerged as a foundation for general-purpose problem solving (Jimenez et al., 2024; Shao et al., 2024; Xie et al., 2024; Zhang et al., 2024; Yuan et al., 2026), with reinforcement learning (RL) becoming a critical post-training stage for advancing their capabilities (Guo et al., 2025; Yu et al., 2026; Hu et al., 2025). However, scaling RL is more challenging than scaling SFT, as RL repeatedly alternates among response generation, reward evaluation, and policy optimization, requiring careful coordination of computation and memory across these stages.

Recent work has addressed these challenges through asynchronous execution and more efficient resource management. For example, asynchronous RL overlap rollout with policy training, reducing GPU idle time between stages (Xu et al., 2026; Zhang et al., 2026a). Other works improve resource allocation and scheduling to accommodate the distinct computational demands of generation and training (Mei et al., 2025). Memory-saving techniques further enable larger models to fit within available hardware: sharding distributes training states across GPUs (Zhao et al., 2023), activation checkpointing reduces memory consumption by recomputing intermediate activations(Chen et al., 2016), and CPU offloading transfers selected training states to host memory (Rajbhandari et al., 2020). Our work complements them by compressing the gradients and optimization updates themselves, thereby reducing both training memory and communication overhead during policy synchronization.

Another line of work exploits low-rank structure to reduce training memory. LoRA (Hu et al., 2021) freezes pretrained weights and optimizes low-rank adapters, reducing the memory required for gradients and optimizer states. GaLore (Su et al., 2025; Zhao et al., 2024) and Flora (Hao et al., 2024) instead keep the original weights trainable while compactly representing gradients and optimizer states. In contrast, LoGRA targets the RL training–generation loop: it accumulates low-rank sketches during backpropagation without materializing full gradients for selected weights and reuses the compressed updates for rollout-policy synchronization. It thus extends compression from training-state storage to gradient computation and policy communication. APOLLO (Zhu et al., 2025) uses random projections to estimate channel-wise adaptive scaling factors, reducing optimizer-state memory while retaining full-gradient updates. In comparison, LoGRA directly accumulates gradient sketches and constructs low-rank updates without materializing full gradients for selected weights.

### 5. Conclusion

We presented LoGRA, a memory-efficient approach to LLM RL post-training that uses low-rank gradient sketches throughout gradient accumulation, weight updates, and policy synchronization. Our results show that compact gradient representations can substantially reduce training memory without sacrificing reasoning performance. In particular, LoGRA enables sustained training of a 27B model for over 1,100 steps on a single eight-GPU node, where dense Adam exceeds the available memory. Predicted-KL step control complements compression by regulating policy changes, with conservative budgets supporting stable long-horizon learning. Together, these findings highlight gradient representation as a practical lever for lowering the hardware barrier to RL post-training.

<!-- Page 11 -->

### References

Nemo rl: A scalable and efficient post-training library. [https://github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL), 2025. GitHub repository. 1

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023. 1

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. *arXiv preprint arXiv:1604.06174*, 2016. 2, 10

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025. 10

Yongchang Hao, Yanshuai Cao, and Lili Mou. Flora: Low-rank adapters are secretly gradient compressors. *arXiv preprint arXiv:2402.03293*, 2024. 2, 8, 10

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021. 2, 10

Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Weixun Wang, Songlin Jiang, Haoran Wang, Hao Chen, Bin Chen, Wenkai Fang, et al. Openrlhf: A ray-based easy-to-use, scalable and high-performance rlhf framework. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing:* *System Demonstrations*, pp. 656–666, 2025. 10

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In *International Conference on Learning* *Representations*, volume 2024, pp. 54107–54157, 2024. 10

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*,

2014. 1
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. *arXiv preprint arXiv:2305.20050*, 2023. 15

Zhiyu Mei, Wei Fu, Kaiwei Li, Guangju Wang, Huanchen Zhang, and Yi Wu. Real: Efficient rlhf training of large language models with parameter reallocation. *Proceedings of Machine Learning and Systems*, 7, 2025. 10

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: international conference for high performance computing,* *networking, storage and analysis*, pp. 1–16. IEEE, 2020. 2, 10

John Schulman and Thinking Machines Lab. Lora without regret. *Thinking Machines Lab: Connectionism*, 2025. doi: 10.64434/tml.20250929. [https://thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/). 2

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In *International conference on machine learning*, pp. 1889–1897. Pmlr, 2015. 4, 10

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024. 10

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv: 2409.19256*, 2024. 1

<!-- Page 12 -->

DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, and Jiawei Zhao. Galore 2: Large-scale llm pre-training by gradient low-rank projection. *arXiv preprint arXiv:2504.20437*, 2025. 2, 10

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. *Advances in Neural Information Processing Systems*, 37:52040–52094, 2024. 10

Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, Jian Hu, Shizhe Diao, Zhenghui Jin, Yunheng Zou, Michael Demoret, et al. Polar: Agentic rl on any harness at scale. *arXiv preprint* *arXiv:2605.24220*, 2026. 1, 10

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *Advances in Neural* *Information Processing Systems*, 38:113222–113244, 2026. 10, 15

Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, Weiming Wu, Jiayang Sun, Jiamin Song, Kaiqian Cui, Bowen Wang, Haoyuan Wu, Yitong Li, et al. Osworld2. 0: Benchmarking computer use agents on long-horizon real-world tasks. *arXiv preprint arXiv:2606.29537*, 2026. 10

Hao Zhang, Mingjie Liu, Shaokun Zhang, Songyang Han, Jian Hu, Zhenghui Jin, Yuchi Zhang, Shizhe Diao, Ximing Lu, Binfeng Xu, et al. Prorl agent: Rollout-as-a-service for rl training of multi-turn llm agents. *arXiv* *preprint arXiv:2603.18815*, 2026a. 10

Shaokun Zhang, Jieyu Zhang, Jiale Liu, Linxin Song, Chi Wang, Ranjay Krishna, and Qingyun Wu. Offline training of language model agents with functions as learnable weights. *arXiv preprint arXiv:2402.11359*,

2024. 10
Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, and Guilin Liu. Nemotron-research-tool-n1: Exploring tool-using language models with reinforced reasoning. In *International Conference on Learning Representations*, volume 2026, pp. 91437–91453, 2026b. 1

Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. *arXiv preprint arXiv:2403.03507*, 2024. 2, 10

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. *arXiv preprint* *arXiv:2304.11277*, 2023. 2, 10

Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park, Vikas Chandra, Bo Long, David Z. Pan, Zhangyang Wang, and Jinwon Lee. Apollo: Sgd-like memory, adamw-level performance. In M. Zaharia, G. Joshi, and Y. Lin (eds.), *Proceedings of Machine Learning and Systems*, volume 7. MLSys, 2025. URL [https://proceedings.mlsys.org/paper_files/paper/2025/file/](https://proceedings.mlsys.org/paper_files/paper/2025/file/) 437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf. 10

<!-- Page 13 -->

### A. Theoretical Derivations

#### A.1. From policy KL to changes in token scores

Equation 5 describes how KL changes with the update scale; Equation 6 gives its coefficient in terms of changes in token scores. We derive them in this order. Throughout, the proposed parameter update *𝐷* and the nonempty set of contexts *𝒫* are fixed. Each context *𝑠* consists of a prompt and the response tokens preceding one next-token prediction. Both policies are evaluated on this same text.

##### Measure the change along an update.

Let *𝑎* be a scalar multiplier: *𝑎* = 0 leaves the weights unchanged, while *𝑎* = 1 applies the full proposal *𝐷*. For vocabulary token *𝑣*, write *𝑝<sub>𝑣</sub>*(*𝑎*) = *𝜋<sub>𝜃+𝑎𝐷</sub>*(*𝑣 | 𝑠*) and *𝑝<sub>𝑣</sub>* = *𝑝<sub>𝑣</sub>*(0). We use natural logarithms and the same temperature *𝑇 >* 0 for both policies. Their KL at context *𝑠* is ∑︁ <u>𝑝<sub>𝑣</sub></u> *𝐾<sub>𝑠</sub>*(*𝑎*) = *𝑝<sub>𝑣</sub>* log*.* (8) *𝑣* *𝑝<sub>𝑣</sub>*(*𝑎*)

A prime denotes a derivative with respect to the multiplier *𝑎*. Thus *𝑝<sup>′𝑣</sup>*(0) measures how quickly token *𝑣*’s probability initially changes along *𝐷*, whereas *𝐾<sub>𝑠′</sub>* (0) measures the initial slope of the KL. At *𝑎* = 0 the two policies coincide, so *𝐾<sub>𝑠</sub>*(0) = 0. Moreover, ∑︁ *𝐾<sub>𝑠′</sub>* (0) = *− 𝑝<sup>′𝑣</sup>*(0) = 0*,* (9) *𝑣* ∑︀ because <sub>𝑣</sub> *𝑝<sub>𝑣</sub>*(*𝑎*) = 1 for every *𝑎*. Individual token probabilities can change to first order, but the KL has zero slope at the point where the distributions agree.

##### Obtain Equation 5: the quadratic dependence on step size.

The second derivative *𝐾<sub>𝑠′′</sub>*(0) measures the local curvature: how quickly the KL’s slope changes as we move away from the current policy. Differentiating again gives

∑︁ <u>[𝑝<sup>′</sup> (0)]²</u> ∑︁ ∑︁ [︂ <u>d</u> ⃒⃒]︂<sub>2</sub> *𝐾<sub>𝑠′′</sub>*(0) = <u><sub>𝑣</sub></u> *− 𝑝<sup>′′</sup><sub>𝑣</sub>*(0) = *𝑝<sub>𝑣</sub>* log*𝑝<sub>𝑣</sub>*(*𝑎*)⃒⃒*.* (10) *𝑣* *𝑝<sub>𝑣</sub>* *𝑣 𝑣* d*𝑎* *𝑎*=0 ∑︀ The term <sub>𝑣</sub> *𝑝<sup>′′</sup><sub>𝑣</sub>*(0) vanishes because total probability remains one. For the local expansion, assume the logits have continuous, bounded third derivatives along *𝐷* near *𝑎* = 0. Taylor’s formula then writes the KL for a small multiplier *𝛼* as <u>𝛼² 𝛼²</u> *𝐾<sub>𝑠</sub>*(*𝛼*) = *𝐾<sub>𝑠</sub>*(0) + *𝛼𝐾<sub>𝑠′</sub>* (0) + *𝐾<sub>𝑠′′</sub>*(0) + *𝑂*(*𝛼³*) = *𝐾<sub>𝑠′′</sub>*(0) + *𝑂*(*𝛼³*)*.* (11) 2 2 Here *𝑂*(*𝛼³*) collects terms whose magnitude is bounded by a constant times *|𝛼|<sup>3</sup>* near zero, for fixed *𝐷*. Averaging over the examined predictions gives

<u>1</u> ∑︁2<u>1</u> ∑︁ *𝐾<sub>𝑠</sub>*(*𝛼*) = *𝛼 𝑞*(*𝐷*) + *𝑂*(*𝛼³*)*, 𝑞*(*𝐷*) = *𝐾<sub>𝑠</sub><sup>′′</sup>*(0)*.* (12) *|𝒫|* 2*|𝒫|* *𝑠∈𝒫 𝑠∈𝒫*

Dropping the higher-order remainder gives Equation 5. Thus *𝑞*(*𝐷*) is half the average KL curvature along the proposal: a larger value means that the same update scale changes the policy more. Within this approximation, halving the scale reduces KL by a factor of four.

##### Obtain Equation 6: express the curvature using token scores.

Let *𝑧<sub>𝑣</sub>*(*𝑎*; *𝑠*) be the logit, or score, of token *𝑣* at context *𝑠* under weights *𝜃* +*𝑎𝐷*. Define *𝑧*˙<sub>𝑣</sub>(*𝑠*) = d*𝑧<sub>𝑣</sub>*(*𝑎*; *𝑠*)*/*d*𝑎|<sub>𝑎</sub>*=0, its initial rate of change along the proposal. Using

<u>exp(𝑧<sub>𝑣</sub>(𝑎; 𝑠)/𝑇)</u> *𝑝<sub>𝑣</sub>*(*𝑎*) = ∑︀*,* <sub>𝑢</sub> exp(*𝑧<sub>𝑢</sub>*(*𝑎*; *𝑠*)*/𝑇*)

<!-- Page 14 -->

where *𝑢* also ranges over vocabulary tokens, differentiating the log probability yields ⃒ (︃ ∑︁ )︃ <u>d</u> ⃒⃒ log*𝑝<sub>𝑣</sub>*(*𝑎*) ⃒ = *𝑧*˙<sub>𝑣</sub>(*𝑠*) *− 𝑝<sub>𝑢</sub>𝑧*˙<sub>𝑢</sub>(*𝑠*)*.* (13) d*𝑎* *𝑎*=0 *𝑇* *𝑢* The subtraction reflects that probabilities depend on relative scores: a token gains probability when its score increases faster than the probability-weighted average score. Substituting into Equation 10 gives

(︃)︃<sub>2</sub>
<u>1</u> ∑︁ ∑︁ <u>1</u> *𝐾<sub>𝑠′′</sub>*(0) = *𝑝<sub>𝑣</sub> 𝑧*˙<sub>𝑣</sub>(*𝑠*) *− 𝑝<sub>𝑢</sub>𝑧*˙<sub>𝑢</sub>(*𝑠*) = Var*𝑣∼𝑝<sub>(𝑠)</sub>*[ ˙*𝑧<sub>𝑣</sub>*(*𝑠*)]*,* *𝑇²* *𝑣 𝑢* *𝑇²*

where *𝑝*(*𝑠*) collects the current token probabilities. This variance measures how differently the scores change, weighted by those probabilities. In particular, adding the same amount to every score leaves softmax probabilities unchanged; correspondingly, identical score-change rates give zero curvature. Substituting this expression into the definition of *𝑞*(*𝐷*) above gives exactly Equation 6:

<u>1</u> ∑︁ *𝑞*(*𝐷*) = Var*𝑣∼𝑝<sub>(𝑠)</sub>*[ ˙*𝑧<sub>𝑣</sub>*(*𝑠*)]*.* 2*|𝒫|𝑇²* *𝑠∈𝒫*

The average weights examined predictions equally. The probe evaluates the variance over the full vocabulary at each selected prediction; only the contexts are subsampled. It computes score-change rates before applying the update, using the factors *𝐷𝑊* = *−𝜂𝑈𝐴* for each selected weight matrix. Neither a stored dense gradient nor a dense curvature matrix is needed.

#### A.2. Scaling an update to a KL budget

The multiplier *𝛼* controls how much of the proposed update *𝐷* we apply: *𝛼* = 1 keeps it unchanged, while *𝛼* = 1*/*2 halves it. Let *𝑞*(*𝐷*) be the predicted KL for the full proposal and let *𝛿 >* 0 be the KL budget, the allowed average predicted change in the policy. We choose *𝛼* by comparing these two quantities. If the proposal exceeds the budget, we shrink it; if it falls below the budget, we enlarge it.

The key is that predicted KL scales with the *square* of the update multiplier, as derived in Appendix A.1. To √︀ match the budget, we therefore solve *𝛼²𝑞*(*𝐷*) = *𝛿*, giving *𝛼* = *𝛿/𝑞*(*𝐷*) for finite positive *𝑞*(*𝐷*). For example, if the proposal predicts four times the allowed KL, halving the update brings its predicted KL down to the budget. Conversely, if it predicts one quarter of the budget, doubling the update would reach the budget. We cap this enlargement at a chosen maximum multiplier *𝛼<sub>max</sub> >* 0, giving Equation 7: {︁ √︀}︁ *𝛼* = min *𝛼<sub>max</sub>, 𝛿/𝑞*(*𝐷*)*.* (14)

The cap prevents a very small KL prediction from producing an excessively large update. This rule changes the update’s size, not its direction.

### B. RowAdam

In the main experiments, we use RowAdam to make the comparison with dense Adam by retaining adaptive scaling based on past gradients. RowAdam is not identical to Adam: it adapts the scale of each row rather than each individual parameter, and does not maintain first-moment momentum.

Let *𝑆<sub>𝑡</sub>* be the normalized gradient sketch at update *𝑡*. For row *𝑖*, RowAdam maintains a running estimate *𝑣<sub>𝑡,𝑖</sub>* of its squared magnitude and uses it to scale the row:

∑︁*𝑟* <u>1</u> *𝑣<sub>𝑡,𝑖</sub>* = *𝛽₂𝑣<sub>𝑡−1,𝑖</sub>* + (1 *− 𝛽₂*) (*𝑆<sub>𝑡</sub>*)²<sub>𝑖𝑗</sub>*,* *𝑟* *𝑗*=1 (15) ̃︀ <u>(𝑆<sub>𝑡</sub>)<sub>𝑖,:</sub> 𝑣<sub>𝑡,𝑖</sub></u> (*𝑈<sub>𝑡</sub>*)<sub>𝑖,:</sub> = √︀*, 𝑣*̂︀<sub>𝑡,𝑖</sub> = <sup>𝑡</sup> *.* *𝑣*̂︀<sub>𝑡,𝑖</sub> + *𝜖<sub>𝑡</sub>* 1 *− 𝛽₂*

<!-- Page 15 -->

We initialize *𝑣<sub>0,𝑖</sub>* = 0; the correction in *𝑣*̂︀<sub>𝑡,𝑖</sub> accounts for this zero initialization. The parameter *𝛽₂* controls how much history is retained. Rows with consistently larger squared gradients receive smaller multipliers. After this row scaling, we rescale *𝑈*̃︀<sub>𝑡</sub> to match the original sketch’s Frobenius norm, yielding the factor *𝑈<sub>𝑡</sub>* used for the weight update. Thus RowAdam adjusts the relative contribution of different rows while preserving the <sup>−8</sup> √︀ sketch’s overall magnitude. We use *𝛽₂* = 0*.*95. The stabilizer *𝜖<sub>𝑡</sub>* is 10 times the mean of *𝑣*̂︀<sub>𝑡,𝑖</sub> across rows, with a positive numerical floor; zero sketches produce zero updates. The row statistics and their update counter persist across projection refreshes. Each selected matrix requires only *𝑑* second-moment values, rather than one value per weight.

### C. Data Details

#### C.1. Mathematical reasoning

We construct DAPO-Math-7.5K from the DAPO-Math-17k training split, using the corresponding release on Hugging Face (Yu et al., 2026). The preparation script streams the source in its dataset order, retains the first occurrence of each prompt, and stops after collecting 7,500 unique prompts. The deduplication key is the content of the first prompt message.

Evaluation uses all 500 problems in the test split of MATH-500 (Lightman et al., 2023). Each problem is converted to a user message, and its reference answer is stored as a string for the verifier. Training and evaluation use the model’s chat template. The training-subset deduplication described above operates within the training source; it is not a cross-dataset semantic decontamination procedure.

Evaluation generates four responses per problem at temperature 0*.*7 and top-*𝑝* 0*.*7. Pass@1 is the mean binary correctness over these 2,000 responses. The verifier extracts and checks the final mathematical answer; an unparseable answer receives no correctness credit. Training generates one response per prompt at temperature

1*.*0 and top-*𝑝* 1*.*0, with a 4,096-token total context limit.
#### C.2. Reasoning-Gym Hard mixture

Table 3: Reasoning-Gym Hard mixture.

Training problems

|Task|Evaluation problems||
|---|---|---|
|codeio|4,868|25|
|emoji_mystery|4,752|22|
|rectangle_count|5,000|25|
|word_ladder|4,994|25|
|pool_matrix|5,000|25|
|tsumego|5,000|25|
|sentence_reordering|4,334|21|
|manipulate_matrix|5,000|25|
|Total|38,948|193|

The long-horizon 27B experiment uses an eight-task Reasoning-Gym mixture with task-specific verifiers. We select tasks with mean verifier scores of approximately 0*.*25–0*.*60 in a preliminary evaluation of 89 tasks, excluding those whose failures mainly reflect response truncation or formatting errors. This provides a challenging training mixture with room for improvement.

For each task, we generate 5,000 candidate training problems and 25 evaluation problems at the default difficulty. We remove duplicate training questions and exclude training–evaluation overlaps; Table 3 reports the final counts. Each task’s verifier returns a score in ([0,1]). We report the equally weighted mean of the eight task-average scores. We report the equally weighted mean of the eight task-average scores. Evaluation uses one response per problem, temperature 0*.*7, top-*𝑝* 0*.*95, and a 16,384-token context limit.

<!-- Page 16 -->

### D. Hyperparameters

#### D.1. Main mathematical-reasoning experiments

Table 4 summarizes the settings for Table 1. Each run uses one node with eight H100 80GB GPUs and 128 CPU

cores: four FSDP training GPUs and four independent generation engines (tensor-parallel degree one). Both use BF16 computation, activation checkpointing, no CPU offloading, and full-weight policy synchronization.

Table 4: Main-experiment settings. Schedule horizons count actual optimizer updates.

|Setting|Dense|LoGRA|
|---|---|---|
|Training seeds|1, 2, 42|1, 2, 42|
|Prompts / responses per update|128 / 128|128 / 128|
|Microbatch per training GPU|1|1|
|Optimizer|Adam (zero weight decay)|RowAdam|
|Moment decay|( 𝛽₁ ,𝛽₂ ) = (0. 9, 0. 95) <sup>−8</sup>|𝛽₂ = 0. 95; no first moment <sup>−8</sup>|
|Epsilon|10, absolute|10, relative|
|Projection|Not applicable|Rank 256, Rademacher|
|Projection refresh|Not applicable <sup>−6</sup> <sup>−7</sup>|Every update <sup>−4</sup> <sup>−5</sup>|
|Step-size rule|LR 10 → 10|Predicted KL 2 × 10 → 2 × 10|
|Schedule|9-update warmup; cosine decay|Cosine budget decay|
|Mismatch subtraction|Not applicable|Enabled; budget floor 50%|
|Evaluation interval|5/5 / OOM|5/5/ 10|

##### KL budget.

For update *𝑢* = 1*,...,𝐻*, the main experiments use [︂ (︂)︂]︂ <u>𝛿<sub>max</sub> − 𝛿<sub>min</sub> 𝑢 − 1</u> *𝛿<sub>𝑢</sub>* = *𝛿<sub>min</sub>* + 1 + cos *𝜋,* (16) 2 *𝐻 −* 1

with *𝐻* = 300, *𝛿<sub>max</sub>* = 2 *×* 10<sup>−4</sup>, and *𝛿<sub>min</sub>* = 2 *×* 10<sup>−5</sup>. The 27B runs stop at 200 updates without shortening the schedule.

Mismatch subtraction accounts for differences between rollout-engine and trainer probabilities at the same nominal weights. We average rollout-minus-trainer log probabilities over generated tokens in completed, nontruncated responses, clamp the estimate below at zero, and smooth it with moving-average weight 0*.*3. Denoting this estimate by *𝜅<sub>𝑢</sub>*, the effective budget is

*𝛿<sub>𝑢</sub><sup>eff</sup>* = max(*𝛿<sub>𝑢</sub> − 𝜅<sub>𝑢</sub>,* 0*.*5*𝛿<sub>𝑢</sub>*)*.* (17)

Without a valid measurement, no subtraction is applied. This adjustment is a budgeting heuristic, not an exact decomposition of KL.

#### D.2. Long-horizon 27B training

The Reasoning-Gym run in Figure 4 uses seed 42, 128 prompts with one response each, microbatch size one, and a 16,384-token context limit. Training temperature and top-*𝑝* are both 1*.*0. It uses rank-256 refreshed projections, SGD sketch updates, a global running baseline, and a KL budget annealed from 2 *×* 10<sup>−4</sup> to 2 *×* 10<sup>−5</sup> over 1,000 merges then held constant. Four training GPUs and four generation engines share one node. Evaluation is scheduled every 20 logged steps;

#### D.3. Reconstruction heatmaps.

We use initialization and steps 160 and 280 of one fixed-basis rank-64 trajectory. At each state, all conditions share 4,096 responses, grouped into 32 batches of 128, and their dense gradients. We measure reconstruction

<!-- Page 17 -->

before optimizer transformations: ∑︀ ̂︀ *⊤* <u>𝑖,𝑗</u> <u>𝐺<sub>𝑖𝑗</sub>𝐺<sub>𝑖𝑗</sub></u> *𝐺* ̂︀ = (*𝐺𝐴*)*𝐴,* cos(*𝐺, 𝐺*̂︀) =*.* *‖𝐺‖𝐹 ‖𝐺*̂︀*‖<sub>𝐹</sub>*

Each heatmap cell equally averages the seven attention/MLP matrices in a layer, 32 batches, three model states, and eight random projection realizations; 28 layers contain 196 matrices in total. Fixed bases are reused across batches; refreshed bases are redrawn.

#### D.4. LoRA

Table 2 uses rank 256 and Qwen2.5-Math-1.5B, with three runs per method. LoRA uses AdamW with constant

learning rate 2 *×* 10<sup>−4</sup>, moment decay (0*.*9*,*0*.*95), epsilon 10<sup>−8</sup>, and zero weight decay. Both methods use 128 prompts with one response each, microbatch size one, and a 4,096-token context limit; LoGRA follows Table 4. LoRA statistics cover 300 logged steps, excluding records without an update from memory/time averages; LoGRA covers 300 actual updates. The different counters, optimizers, and schedules prevent isolating adapter parameterization alone.