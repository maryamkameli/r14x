URL: https://arxiv.org/pdf/2508.00222
Content-Type: text/markdown
Method: arxiv
Notes: Fetching PDF for full content...; PDF converted via markit

---

# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization

**Authors:** Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, Ge Li
**Published:** 2025-07-31
**Categories:** cs.AI, cs.CL, cs.LG
**arXiv:** 2508.00222

---

## Abstract

Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.

---

## Full Paper

<!-- Page 1 -->

#### <u>Preprint, July 2025</u>

## RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization

**Yihong Dong<sup>1,2</sup>, Xue Jiang<sup>1,2</sup>, Yongding Tao¹, Huanyu Liu¹, Kechi Zhang¹, Lili Mou<sup>3,4</sup>,** **Rongyu Cao², Yingwei Ma², Jue Chen², Binhua Li², Zhi Jin¹, Fei Huang², Yongbin Li², Ge Li¹** 1 School of Computer Science, Peking University 2 Tongyi Lab, Alibaba Group 3 Department of Computing Science, University of Alberta 4 Canada CIFAR AI Chair dongyh@stu.pku.edu.cn lige@pku.edu.cn

### Abstract

Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM’s immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM’s problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.¹

### 1 Introduction

The paradigm of Reinforcement Learning with Verifiable Reward (RLVR) has significantly pro- pelled the improvement of reasoning performance in Large Language Models (LLMs), particularly in solving complex tasks involving math and coding (OpenAI, 2024; Guo et al., 2025; KimiTeam,

2025). RLVR optimizes LLMs’ performance via a reinforcement learning (RL) process guided by
# arXiv:2508.00222v5 [cs.AI] 15 Apr 2026

verifiable reward computation, e.g., determining whether an output matches a ground-truth math answer or passes unit tests for coding. This method enables LLMs to scale their computation at test time by extending Chain-of-Thought (CoT) processes and spontaneously exhibit sophisticated cognitive behaviors such as reflection and exploration. Thus, RLVR is believed to be a promising way for LLMs to achieve continuous self-evolution toward more powerful AI (Guo et al., 2025).

Despite the empirical successes, some work (Havrilla et al., 2024; Shao et al., 2024; Yue et al., 2025a) points out that current RLVR cannot enable LLMs to acquire novel reasoning abilities, but rather simply utilize reasoning patterns already in the base model. As shown in Figure 1(a), although the pass@1 performance of RLVR-trained models surpasses that of the base model, its pass@128<sup>2</sup> is substantially lower. This trend suggests that the underlying capability distribution of the base

⁰Work done during Yihong Dong and Xue Jiang’s internship at Tongyi Lab. ¹Our code is available at [https://github.com/YihongDong/RL-PLUS](https://github.com/YihongDong/RL-PLUS). ²The pass@k calculates the proportion of problems the model can potentially solve within a finite (k) number of attempts metric is commonly used to gauge a model’s capability boundary.

<!-- Page 2 -->

**Problem A Problem B** Average Performance **GRPO < BaseCorrect** **IncorrectReasoning scope Reasoning scope**

0.8
0.7
Base Model GRPO pass@k0.6 RL-PLUS **Reasoning scope collapses Reasoning scope collapses**

0.5
0.4 1 2 4 8 16 32 64 128
k

(a) Collapse problem of capability boundaries in LLMs after RLVR training. (b) Benefits of our RL-PLUS.
Figure 1: (a) The commonly used RLVR methods can lead to the collapse problem of capability

boundaries in base LLMs. (b) RL-PLUS can overcome capability boundary collapse of LLMs in RLVR, consistently showing larger pass@k than base model.

model is broader and that existing RLVR can collapse the base model’s capability boundary, thus fundamentally limiting the acquisition of new reasoning pathways.

This limitation stems from an essential challenge when applying RLVR to LLMs: the potential solution space of LLMs is extremely immense with sparse reward that current RLVR techniques cannot effectively guide the model to explore new and unknown pathways, i.e., outward exploration. The challenge is particularly acute in long reasoning tasks where rewards are contingent upon the successful completion of an entire inferential chain. A single erroneous step can nullify the reward for the entire trajectory, thus failing to provide a positive signal for acquiring new knowledge. Consequently, the model is compelled to focus on inward exploitation, meaning that it refines and optimizes the knowledge and reasoning methods it already possesses, which results in a con- traction of the model’s exploratory range and a shrinking of its capabilities. This phenomenon not only prevents the model from acquiring new information or abilities that surpass its base model, but also significantly impedes any sustained enhancement of its overall performance.

The ancient educational principle that *“If one learns from others but does not think, one will be* *bewildered. If, on the other hand, one thinks but does not learn from others, one will be in peril”³* offers a crucial lens through which to view the limitations of current methodologies for enhancing LLM reasoning. Current RLVR can be viewed as the latter case, which excels at “thinking” through inward exploitation but demonstrates inadequate outward exploration due to its inherently on-policy strategy coupled with LLM’s immense action space and sparse reward, i.e., hard to continuous “learning” of new knowledge. Conversely, approaches like Supervised Fine-Tuning (SFT) represent the former case, focusing on imitating solutions but failing to internalize the underlying reasoning principles, leading to brittleness when encountering novel problems.

This motivates us to develop novel RLVR approaches with effective external learning, but there are two key challenges that need to be addressed. First, a distributional mismatch between the model’s policy and the external data source is inevitable. Standard importance sampling corrections for RL are inadequate, i.e., employing the proxy with on-policy introduces systematic bias, whereas direct using off-policy usually suffers from high variance and bias due to their significantly divergent distributions. Second, there is a challenge of efficiently extracting valuable information from this external data. Models are naturally inclined to favor high-probability tokens, thus reinforcing existing knowledge. However, the key to discovering novel reasoning often lies in exploring low-probability tokens that the model would otherwise ignore.

In this paper, we propose RL-PLUS, a novel hybrid-policy optimization approach designed to synergize internal exploitation with external data during RL process. Specifically, RL-PLUS has two core techniques. ❶ To resolve the issue of distributional mismatch, we employ Multiple Importance Sampling, which provides a lower bias and variance estimation of importance by combining information from multiple policies. ❷ To promote the discovery of new knowledge, we introduce an Exploration-Based Advantage Function, which reshapes the learning objective by prioritizing

³A principle from the philosopher and educator Confucius.

<!-- Page 3 -->

advantages for reasoning paths that are correct but are hard to explore (i.e., low probability) under the current policy. We also provide a theoretical analysis demonstrating that our approach achieves lower bias and variance compared with mainstream RLVR methods when leveraging external data.

Extensive experiments show the effectiveness and generalization of RL-PLUS. On six challenging math reasoning benchmarks, RL-PLUS achieves state-of-the-art (SOTA) performance, outperforming existing RLVR methods and improving upon SFT+GRPO by 5.2 average points. RL-PLUS also demonstrates superior generalization to six out-of-distribution (OOD) tasks. RL-PLUS exhibits clear and stable improvements across diverse model families, with the average relative improvements of GRPO up to 69.2%. Moreover, the analysis of Pass@k curves across multiple benchmarks indicates that RL-PLUS effectively transcends the inherent capability ceiling of the base model, thus addressing capability boundary collapse observed in prior RLVR approaches.

### 2 Background and Related Work

In this section, we first establish the theoretical preliminaries necessary to understand our work, and then provide a critical review of the most related work, identifying key limitations in existing methods and thereby motivating the design of our proposed RL-PLUS.

2.1 Preliminary Knowledge **LLM-based Reasoning as a Markov Decision Process.** We frame the task of generating a reasoning sequence (e.g., a solution to a math problem) as a Markov Decision Process (MDP) (Puterman, 2014). At each timestep *t*, the state *s<sub>t</sub>* consists of the initial prompt *q* concatenated with the sequence of previously generated tokens, *y<t*. The action *a<sub>t</sub>* is the selection of the next token *y<sub>t</sub>* from the vocabulary. The model, or policy *π<sub>θ</sub>*, maps a state to a distribution over actions. A reward *R*(*q,y*) is provided only upon completion of the entire sequence *y*. In the context of RLVR, this reward is typically sparse and binary. For example, a score is 1 if the final answer is correct and 0 otherwise. The objective is to learn a policy *π<sub>θ</sub>* that maximizes the expected cumulative reward *J*(*θ*) = E*y∼π<sub>θ</sub>*[*R*(*q,y*)]. **Policy Gradient Optimization.** Policy gradient methods are the standard for optimizing LLMs in on-policy RLVR settings. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) shows exceptional performance in various tasks, especially to enable effective scaling within the RLVR paradigm. Compared to Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017), GRPO leverages group-normalized rewards to estimate advantages, eliminating the need for a value model and thereby improving computational efficiency. The standard GRPO objective is:
  X *|y|* *J<sub>RL</sub>*(*θ*) = E<sub>(q,y)</sub>*∼D<sub>on</sub>*  min (*r<sub>i,t</sub>*(*θ*)*A<sub>i</sub>,*clip(*r<sub>i,t</sub>*(*θ*)*,*1 *− ϵ,*1 + *ϵ*)*A<sub>i</sub>*) *− βD<sub>KL</sub>*[*π<sub>θ</sub>∥π<sub>ref</sub>*] (1) *t*=1

<u>π<sub>θ</sub>(o<sub>i,t</sub> | q,oi,<t)</u> *r<sub>i,t</sub>*(*θ*) =*,* (2) *π<sub>θ</sub><sub>old</sub>*(*o<sub>i,t</sub> | q,oi,<t*)

<u>R<sub>i</sub> − mean({R₁,R₂,...,RG})</u> *A<sub>i</sub>* =*,* (3) std(*{R₁,R₂,...,RG}*)

where *r<sub>i,t</sub>*(*θ*) is the importance sampling ratio and *A<sub>i</sub>* is the estimated advantage for an on-policy trajectory *y<sub>i</sub>*. Recent work, such as Simple-rl (Zeng et al., 2025) and DAPO (Yu et al., 2025), has proposed either setting the KL coefficient *β* to a very small value or omitting the KL term in Equation 1 entirely. The rationale is that during the training of a model for long CoT reasoning, the model’s distribution is expected to diverge significantly from the initial policy, rendering this constraint unnecessary.

**Evaluating Reasoning Boundaries with pass@k.** To accurately assess a model’s true problem-solving capabilities, we utilize the pass@k metric (Chen et al., 2021). It measures the probability of obtaining at least one correct answer within *k* independent samples for a given problem. Un- like mean accuracy (i.e., pass@1), pass@k provides a more comprehensive view of the model’s

<!-- Page 4 -->

reasoning potential and is critical for evaluating whether a method expands the set of solvable problems (Yue et al., 2025a).

This on-policy RLVR paradigm, while powerful, leads to two fundamental challenges when the goal is to surpass a base model’s intrinsic capabilities: 1) an inability to effectively integrate novel, external knowledge due to the high variance and bias associated with off-policy data, and 2) the tendency for on-policy exploration to collapse into known, high-probability reasoning paths, thereby shrinking the model’s reasoning boundary. These challenges directly motivate our approach.

2.2 Related Work We position RL-PLUS by critically examining two primary lines of research: on-policy RLVR for reasoning and hybrid SFT-RL methods. **On-Policy RLVR and Its Intrinsic Limitations** Reinforcement learning has become a corner- stone for enhancing LLM reasoning (Yue et al., 2025b; Liu et al., 2025c; Wang et al., 2025). Seminal works have shown that RLVR can significantly improve performance on complex reasoning tasks by rewarding correct final answers (Guo et al., 2025; Zeng et al., 2025; Hu et al., 2025). Subsequent research has refined this paradigm; for instance, PRIME-Zero (Cui et al., 2025a) uses implicit process rewards, and Oat-Zero (Liu et al., 2025b) simplifies the advantage calculation in GRPO. However, a growing body of evidence reveals a critical flaw in these on-policy methods: they primar- ily optimize existing knowledge rather than discovering new reasoning capabilities. This leads to two well-documented issues. First is the Capability Boundary Collapse problem. While RLVR models often show superior pass@1 performance, their advantage diminishes as *k* increases in pass@k evaluations, with base models eventually surpassing them (Yue et al., 2025a). This strongly suggests that RLVR refines the probability of known correct paths but fails to expand the overall set of solvable problems. Second, these methods suffer from Entropy Collapse, where policy entropy sharply decreases during training, making the model overly deterministic and hindering further exploration (Cui et al., 2025b). This indicates that on-policy RLVR, by its nature, is prone to inward exploitation that reinforces existing biases and limits the model’s potential. **Hybrid SFT-RL Methods** To overcome knowledge limitations of pure RL, researchers explored hybrid methods that combine RL with SFT on external demonstration data (Cai et al., 2025). Early approaches employed sequential, multi-stage training (SFT then RL), as seen in models like Instruct-GPT (Ouyang et al., 2022). While conceptually simple, this often leads to catastrophic forgetting of the SFT-learned knowledge and suffers from computational inefficiency. More recent work has focused on unified or interleaved training frameworks. For example, ReLIFT (Ma et al., 2025) alternates between RL and online fine-tuning on difficult problems, while LUFFY (Yan et al., 2025) selectively imitates high-quality external trajectories using a mixed policy. In another example, TAPO (Wu et al., 2025) enhances RL by integrating external, high-level guidance in the form of “thought patterns” abstracted from prior data. Other methods, such as SASR (Chen et al., 2025) and SuperRL (Liu et al., 2025a), employ adaptive switches to dynamically balance SFT and RL objectives based on training state. While these methods are more sophisticated, they often rely on complex, potentially unstable heuristics for balancing the two learning signals. Moreover, simply adding an SFT loss to the RL objective, as explored in “GRPO w/ SFT Loss”, can degrade performance, highlighting the difficulty of effective integration. Even advanced frameworks like UFT (Wang et al., 2024b), which aim to unify SFT and RL to accelerate convergence, do not explicitly address how to stabilize off-policy updates while simultaneously directing exploration towards novel solutions. **Motivation** The foregoing analysis reveals persistent gaps in the related work. On-policy RLVR methods are constrained by the base model’s inherent knowledge, while existing hybrid SFT-RL methods lack a principled mechanism to both stabilize learning from external, off-policy data and explicitly incentivize exploration of low-probability but correct reasoning pathways. RL-PLUS is designed to directly address these deficiencies.

<!-- Page 5 -->

### 3 RL-PLUS

RL-PLUS overcomes the LLM’s capability boundaries collapse problem in RLVR by integrating externally-guided exploration with the exploitation of internal reasoning pathways.

3.1 Mitigating Distributional Mismatch with Multiple Importance Sampling A central challenge in learning from a static dataset *D<sub>e</sub>* = *{e<sub>i</sub>}<sup>N</sup><sub>i</sub>*=1is the distributional shift between the target policy *π<sub>θ</sub>* and the unknown behavior policy *π<sub>ω</sub>*. Standard importance sampling (IS) presents a dilemma for correcting this mismatch. On-policy IS estimator, which uses a proxy like *π<sub>θ</sub><sub>old</sub>* in the denominator, is systematically biased when applied to external data from *π<sub>ω</sub>* (**Lemma A.5**). Conversely, the theoretically correct off-policy estimator, using weights *r<sub>te</sub>*(*θ*) = *π* <u>π</u> <sub>ω</sub> <u><sub>θ</sub></u><u>(</u> ( <u>e</u> *e*
<u><sub>t</sub></u> <sub>t</sub> <u>|</u> *|* <u>e</u> *e* <u><t</u> *<t* <u><sup>)</sup></u><sub>)</sub>, suffers from support mismatch of *π<sub>θ</sub>* (**Lemma A.6**) and prohibitively high variance as the policies diverge (**Lemma A.7**), which destabilizes training. This issue is compounded by the fact that *π<sub>ω</sub>* is usually unknown, rendering direct weight computation infeasible.

To solve this, we introduce Multiple Importance Sampling to construct an estimator with lower variance and controllable bias. Instead of directly estimating *π<sub>ω</sub>*, we treat the generation of an external sample as arising from a mixture policy composed of the previous policy *π<sub>θ</sub><sub>old</sub>* and the external policy *π<sub>ω</sub>*. Therefore, the Multiple Importance Sampling of each token can be defined as:

<sup>m</sup> <u>2π<sub>θ</sub>(e<sub>i,t</sub>|q,ei,<t)</u> *r<sub>i,t</sub>*(*θ*) =*,* (4) *π<sub>ω</sub>*(*e<sub>i,t</sub>|q,ei,<t*) + *π<sub>θ</sub><sub>old</sub>*(*e<sub>i,t</sub>|q,ei,<t*) where *e<sub>i,t</sub>* is the *t*-th token in the external data trajectory *e<sub>i</sub>*. It replaces the aforementioned *explosive* *bias from poor proxy or support mismatch* with a controlled, bounded distortion error (**Remarks A.8** **and A.9**), making the overall MIS estimator robust for stable learning from external data. The formal denominator acts as a crucial *variance guardrail*. The presence of *π<sub>θ</sub><sub>old</sub>*, which is intentionally kept close to *π<sub>θ</sub>*, prevents the ratio from exploding even if *π<sub>ω</sub>* is highly dissimilar, ensuring the estimator’s variance remains bounded. **Theorem 3.1** (Variance Robustness of MIS)**.** *So long as there is at least one policy in the behavior* *pool {π<sub>β</sub><sub>k</sub>} (e.g., π<sub>β</sub><sub>k</sub><sup>∗</sup>) that is a good approximation of the target policy π<sub>θ</sub> (i.e., π<sub>β</sub><sub>k</sub><sup>∗</sup> ≈ π<sub>θ</sub>), the* *variance of the MIS estimator will be low. The estimator is insensitive to other arbitrarily ”bad”* *behavior policies in the pool. (See Proof in Appendix A.4)*

A key challenge remains: the behavior policy *π<sub>ω</sub>* is unknown. We require a robust method to estimate it. Instead of naively using a proxy, we derive an estimator for *π<sub>ω</sub>* from a principled Bayesian perspective. We frame the estimation as a decision problem where we must balance our belief in our best available model, *π<sub>θ</sub><sub>old</sub>*, against a state of maximal uncertainty, represented by a non-informative uniform policy *U*. This allows us to hedge against model error, leading to the following Bayes-optimal estimator. **Theorem 3.2** (Bayes-Optimal Policy Estimator)**.** *Let the model space for the unknown behavior* *policy π<sub>ω</sub> be composed of two candidate models: (1) The specific proxy policy, π<sub>θ</sub><sub>old</sub>, representing our available, specific information. (2) A non-informative uniform policy, U*R (*τ*)*, representing maximal uncertainty. Let the trajectory space T have a finite volume V* = *T* *dτ, such that* *U*(*τ*) = 1*/V. Under the Principle of Indifference, we assign equal prior probabilities to these models, i.e., P*(*π<sub>ω</sub>* = *π<sub>θ</sub><sub>old</sub>*) = *P* (*π<sub>ω</sub>* = *U*) = 1*/*2*. Then, the estimator π*ˆ<sub>ω</sub> *that minimizes the Bayes* *risk (expected L2 error) is the Bayesian model average: π*ˆ<sub>ω</sub><sup>∗</sup> (*τ*) = *₂π<u><sup>1</sup></u> <sub>θ</sub><sub>old</sub>*(*τ*) + <u><sup>1</sup></u>*₂U*(*τ*) *(See Proof in* *Appendix A.5)*

3.2 Efficient Exploration with Exploration-Based Advantage Function Merely incorporating external data stably is insufficient; we must also guide the model to focus on its most valuable information, especially the ”new knowledge” that the model is unlikely to discover on its own. Models tend to favor high-probability tokens, whereas novel knowledge is often embedded in correct reasoning paths that the model considers to have low probability. To this end, we design an Exploration-Based Advantage Function, *A<sup>c</sup><sub>i,t</sub>*, which prioritizes encourag- ing the model to explore reasoning steps that are correct but hard to explore, which can be defined as:

<!-- Page 6 -->

<sup>c</sup> <u>R<sub>i</sub> − mean({R₁,R₂,...,RG})</u> *A<sub>i,t</sub>* = *· C<sub>i,t</sub>* (5) std(*{R₁,R₂,...,RG}*)

The first term is the standardized reward for all trajectories, including both internal exploration and external data, and the second term is the weight to encourage exploration. Inspired by focal loss (Lin et al., 2017), we define the weight *C<sub>i,t</sub>* as:

*C<sub>i,t</sub>* = (1 *−* detach(*π<sub>θ</sub>*(*e<sub>i,t</sub>|q,ei,<t*)))<sup>γ</sup>*,* (6)

where *π<sub>θ</sub>*(*e<sub>i,t</sub>|q,ei,<t*) represents the model’s exploration probability in the correct token *e<sub>i,t</sub>* from the external data. When it is hard to explore (i.e., *π<sub>θ</sub>* is small), the weight *C<sub>i,t</sub>* becomes large, ampli- fying the advantage signal for that timestep and compelling the model to attend to this overlooked region. *γ* is a hyperparameter to control *C<sub>i,t</sub>*. The ‘detach’ function is a standard operation in Torch that prevents gradients from backpropagating through the probability calculation, which enhances training stability.

3.3 The Composite RL-PLUS Objective To synergize internal exploitation *D<sub>o</sub>* with external data *D<sub>e</sub>*, we formulate the final training objective of RL-PLUS as a composite function *J*RL-PLUS(*θ*), which is defined as:
*J*RL-PLUS(*θ*) = E<sub>(o</sub><sub>i</sub>*,A<sub>i</sub>*)*∼D<sub>o</sub>* <u>[r<sub>i,t</sub>(θ)A<sub>i</sub></u>] + E<sub>(e</sub><sub>i</sub>*,A<sup>c</sup><sub>i,t</sub>*)*∼D<sub>e</sub> r<sub>i,t</sub><sup>m</sup>* (*θ*)*A<sup>c</sup><sub>i,t</sub>* (7) | {z} | {z} Internal Exploitation (Thinking) External data for Exploration (Learning)

where the first term represents the standard policy gradient objective, which is responsible for stabi- lizing and improving upon the model’s existing reasoning capabilities. The second term constitutes the core of our contribution, which drives the policy to external exploration. It leverages our two primary innovations: 1) Multiple Importance Sampling *r<sub>i,t</sub><sup>m</sup>* (*θ*), which provides a low-variance, robust mechanism for integrating external data, and 2) Exploration-Based Advantage Function *A<sup>c</sup><sub>i,t</sub>*, which re-weights the learning signal to prioritize novel yet high-value reasoning paths.

Moreover, we omit the clipping mechanism (e.g., clip(*r<sub>t</sub>*(*θ*)*,*1*− ϵ,*1+*ϵ*)), which would suppress the gradient signals corresponding to highly informative, low-probability events, i.e., the “new knowledge” we aim to acquire. By removing this constraint, RL-PLUS is empowered to take larger, more assertive optimization steps when it encounters valuable information in the external data, thus accel- erating the assimilation of novel knowledge and more effectively expanding its capability boundaries in RLVR.

### 4 Experimental Results

In this section, we conduct extensive experiments to demonstrate the effectiveness and generalization of RL-PLUS. Detailed experimental setups can be found in Appendix.

**Performence of RL-PLUS.** As shown in Table 1, RL-PLUS comprehensively outperforms existing RLVR methods across all evaluated applications, achieving SOTA performance. A comparison with several straightforward baselines clearly demonstrates the benefits of RL-PLUS. SFT can be viewed as a means of learning from external knowledge, while GRPO enables the model to explore solutions on its own through reinforcement learning. The combined “SFT+GRPO” approach yields synergistic gains, illustrating the value of integrating both external knowledge and self-exploration. However, the “GRPO w/ SFT Loss” baseline, which simply adds an SFT loss to the RL training, shows a decline in performance. This suggests that effectively merging these two learning paradigms is a non-trivial challenge. RL-PLUS significantly improves upon “SFT+GRPO” by an average of +5.2 points, showcasing a more potent strategy for this integration. Furthermore, when compared to concurrent methods like LUFFY and ReLIFT, which also incorporate external examples into their training process in some form, RL-PLUS also achieves superior performance, which indicates that RL-PLUS offers a more effective way for learning from external knowledge.

<!-- Page 7 -->

Table 1: Performance of RL-PLUS against other baselines, where the best-performing result for

each benchmark is highlighted in **bold** and the base model is Qwen2.5-Math-7B for all methods.

|Method||AIME 24|AIME 25|AMC|MATH-500|Minerva|Olympiad|Avg.|
|---|---|---|---|---|---|---|---|---|
|Qwen2.5-Math-7B||11.5|4.9|31.3|43.6|7.4|15.6|19.0|
|Qwen2.5-Math-7B-Instruct||12.5|10.2|48.5|80.4|32.7|41.0|37.6|
|SimpleRL|(Zeng et al., 2025)|27.0|6.8|54.9|76.0|25.0|34.7|37.4|
|OpenReasoner|(Hu et al., 2025)|16.5|15.0|52.1|82.4|33.1|47.1|41.0|
|PRIME|(Cui et al., 2025a)|17.0|12.8|54.0|81.4|39.0|40.3|40.7|
|Oat (Liu et al., 2025b)||33.4|11.9|61.2|78.0|34.6|43.4|43.7|
|DAPO|(Yu et al., 2025)|23.4|15.5|66.3|86.0|40.1|49.6|46.8|
|TAPO|(Wu et al., 2025)|33.3|18.6|77.5|83.4|38.2|46.2|49.5|
|LUFFY|(Yan et al., 2025)|29.4|23.1|65.6|87.6|37.5|57.2|50.1|
|ReLIFT|(Ma et al., 2025)|28.4|21.8|64.3|86.8|40.1|54.8|49.4|
|SFT||22.2|22.3|52.8|82.6|40.8|43.7|44.1|
|GRPO|(Shao et al., 2024)|25.1|15.3|62.0|84.4|39.3|46.8|45.5|
|GRPO w/ SFT Loss||19.5|16.4|49.7|80.4|34.9|39.4|40.1|
|SFT+GRPO||25.8|23.1|62.7|87.2|39.7|50.4|48.2|
|RL-PLUS||33.4|25.9|68.1|90.2|43.8|58.8|53.4|

Table 2: Out-of-Distribution performance on programming tasks (i.e., HumanEval, LiveCodeBench,

LeetCode) and science QA (i.e., ARC-c, GPQA-diamond, MMLU-Pro).

|Method|HumanEval|LeetCode|LiveCodeBench|ARC-c|GPQA-diamond|MMLU-Pro|Avg.|
|---|---|---|---|---|---|---|---|
|Base Model|42.1|22.8|14.9|18.2|13.1|30.2|23.6|
|SFT|55.5|8.3|8.1|75.2|24.7|42.7|35.8|
|GRPO|63.4|21.1|15.3|81.7|40.4|47.5|44.9|
|SFT+GRPO|59.8|8.34|9.7|72.4|24.2|37.7|35.4|
|RL-PLUS|68.3|27.8|19.2|82.3|40.4|54.7|48.8|

**Performance on OOD Tasks.** The results on OOD tasks are presented in Table 2, which show that RL-PLUS achieves substantial improvements over all baselines, including the mainstream method SFT+GRPO. It surpasses the next best baseline by an average of +3.9 points. This indicates that RL-PLUS not only enhances capabilities within a specific domain but also develops more fundamental reasoning abilities that generalize to other domains. In the domain of science QA, RL-PLUS consistently outperforms both GRPO and SFT+GRPO across all benchmarks. More notably, under a significant domain shift to programming tasks, our approach maintains its strong performance and advantage. In contrast, the performance of SFT and SFT+GRPO deteriorates significantly in this area. Considering this alongside the in-domain results from Table 1, a clear pattern emerges: while SFT-based methods provide a strong boost for in-domain tasks, they fail to generalize and perform worse than RL-based methods in OOD scenarios. RL-PLUS resolves this trade-off. By effectively merging the external knowledge acquisition of SFT with the robust generalization of RL, it achieves superior performance in both in-domain and out-of-distribution settings, outclassing methods reliant on either paradigm alone.

**Training Dynamics.** In Figure 2, we present the training dynamics of our proposed method and baselines on various benchmarks. As illustrated, RL-PLUS consistently outperforms the alterna- tives in terms of test accuracy and rewards throughout the training process. Notably, RL-PLUS continues to show a clear upward trend in performance even after the baselines have plateaued. We further analyze the changes in actor entropy during training. We observe that directly incorporating external data during rollouts (the green line in Figure 2) leads to an “entropy explosion”, causing the model’s outputs to become chaotic. In contrast, the entropy of the baseline models collapses to nearly zero over the course of training, indicating a loss of exploratory capability. The entropy of RL-PLUS, however, does not diminish to zero, which suggests that our trained model retains a considerable capacity for exploration. Prior research (Cui et al., 2025b) has established that policy performance is achieved at the cost of policy entropy, and the depletion of entropy marks the upper limit of performance. This implies that RL-PLUS still possesses potential for further improvement. Additionally, the response length can reflect the test-time scaling performance of a method. The

<!-- Page 8 -->

Average Test Score Critic Rewards Mean Actor Entropy Loss Response Length

1.75
0.55 5000
0.8
0.50 1.50
0.7 1.25
0.45 4000
0.40 0.6 1.00
Value0.35 Value0.5 Value0.75Value3000

0.30 0.4 0.50
2000

0.25 0.3
0.25
0.20
0.2 0.00 1000
0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600 Step Step Step Step GRPO Externel Data with / Externel Data with / old RL-PLUS

Figure 2: Training dynamics of RL-PLUS and other baselines.

steadily increasing response length of RL-PLUS is the indicator of a healthy and robust training state. In contrast, while directly incorporating external data also leads to long response lengths, its low accuracy and high policy entropy suggest that this length stems from unproductive exploration rather than meaningful reasoning.

Table 3: The performance of RL-PLUS based on Different LLMs.

|Model|AIME 24|AIME 25|AMC|MATH-500|Minerva|Olympiad|Avg.|
|---|---|---|---|---|---|---|---|
|LLaMA-3.1-8B|4.7|0.4|18.5|46.4|19.8|13.2|17.2|
|SFT|2.6|0.9|29.8|50.0|21.3|16.9|20.2|
|GRPO|3.5|0.5|19.5|45.0|20.2|14.2|17.2|
|RL-PLUS|11.7|2.1|35.5|64.4|29.4|31.2|29.1|
|Deepseek-Math-7B|1.1|0.3|14.5|40.4|18.8|10.7|14.3|
|SFT|3.8|0.3|23.3|51.2|21.3|19.8|19.9|
|GRPO|2.5|0.2|17.3|47.0|20.9|14.5|17.1|
|RL-PLUS|4.1|0.4|25.0|54.8|21.7|21.4|21.3|
|Qwen2.5-Math-1.5B|7.2|3.6|26.4|28.0|9.6|21.2|16.0|
|SFT|11.7|13.2|37.8|70.6|26.8|31.3|31.9|
|GRPO|11.8|7.7|40.2|61.8|26.8|32.0|30.1|
|RL-PLUS|20.4|13.6|50.0|80.4|33.1|45.2|40.5|
|Qwen2.5-Math-7B|11.5|4.9|31.3|43.6|7.4|15.6|19.0|
|SFT|22.2|22.3|52.8|82.6|40.8|43.7|44.1|
|GRPO|25.1|15.3|62.0|84.4|39.3|46.8|45.5|
|RL-PLUS|33.4|25.9|68.1|90.2|43.8|58.8|53.4|

**Application on Various LLMs.** To validate the applicability of RL-PLUS on various LLMs, we conduct experiments on several mainstream open-source LLMs, including LLaMA-3.1-8B, Deepseek-Math-7B, and the 1.5B and 7B versions of Qwen2.5-Math. The detailed results are presented in Table 3. The results indicate that RL-PLUS achieves comprehensively superior performance, regardless of the base model. Notably, on Qwen2.5-Math-7B model, RL-PLUS elevates the average score to 53.4, significantly outperforming the base model of 9.0 and other methods such as SFT of 44.1 and GRPO of 45.5. Furthermore, on LLaMA-3.1-8B, where methods like GRPO struggled to yield improvements, RL-PLUS successfully trained the model to achieve an absolute gain of 11.9 points. These findings provide evidence that RL-PLUS can consistently enhance LLMs of varying architectures and scales, significantly boosting their reasoning capabilities.

**Acquiring Reasoning Abilities Beyond Base Model.** The fundamental goal of incorporating an external policy into the RL-PLUS method is to expand the model’s capability boundary by con- tinuously introducing knowledge. Following the experimental setup of (Yue et al., 2025a), we test whether RL-PLUS acquires superior reasoning abilities relative to the base model. Figure 3 displays the pass@k performance curves for different methods across multiple tasks. A clear trend is observ- able where the performance curve of the GRPO method gradually converges with that of the base model as k increases. In some instances, GRPO’s performance even drops below the base model at larger k-values, a finding consistent with that of (Yue et al., 2025a). In contrast, our approach maintains a consistent performance advantage over both the base model and GRPO as k-values increase.

<!-- Page 9 -->

AIME AMC MATH-500 Olympiad

1.0
Base Model 1.00

0.7 GRPO 0.8 RL-PLUS
0.9 0.95
0.6
0.90 0.7
0.8
0.5
0.85 0.6
pass@k0.4 pass@k0.7 pass@k pass@k

0.80
0.5
0.3 0.6 0.75
0.4
0.2 0.70
0.5
1 2 4 8 16 32 64 128 1 2 4 8 16 32 64 128 1 2 4 8 16 32 64 128 1 2 4 8 16 32 64 128 k k k k

Figure 3: Pass@k curves of RL-PLUS compared with baselines across multiple benchmarks.

This sustained outperformance provides strong evidence that RL-PLUS effectively breaks through the capability boundary of the base model, rather than merely optimizing performance within its inherent ability range. On the AMC and MATH-500 tasks, the accuracy of RL-PLUS eventually plateaus because its performance is approaching the maximum possible score of 1.0.

Table 4: Ablation Study of RL-PLUS.

|Method|AIME 24|AIME25|AMC|MATH-500|Minerva|Olympiad|Avg.|
|---|---|---|---|---|---|---|---|
|Variants with External Data||||||||
|π<sub>θ</sub>/π<sub>θ</sub><sub>old</sub>|19.6|14.8|55.1|81.0|33.5|46.2|41.7|
|π<sub>θ</sub>/π<sub>θ</sub><sub>ω</sub>|25.8|16.3|59.9|83.8|32.4|49.3|44.6|
|π<sub>θ</sub>/π<sub>θ</sub><sub>ω</sub> with Our Policy Estimation|26.1|19.2|62.3|86.8|38.6|52.0|47.5|
|RL-PLUS|33.4|25.9|68.1|90.2|43.8|58.8|53.4|
|- Exploration-Based Advantage Function|28.3|24.1|67.8|88.8|40.4|56.0|50.9|
|- Multiple Importance Sampling|25.1|15.3|62.0|84.4|39.3|46.8|45.5|

**Ablation Study.** To analyze the sources of RL-PLUS’s effectiveness, we conduct a series of ablation studies, with the results presented in Table 4. We first ablate the two core components of our approach: Multiple Importance Sampling and Exploration-Based Advantage Function. The experimental results show that removing Exploration-Based Advantage Function causes the model’s average performance to decrease from 53.4 to 50.9, which demonstrates the importance of efficient exploration for reinforcement learning. Furthermore, removing Multiple Importance Sampling leads to a more significant performance degradation, with the average score dropping substantially to 45.5, highlighting the significance of incorporating external knowledge. Additionally, we compare our method against three naive approaches for integrating external knowledge. The first variant approximates the external policy *π<sub>θ</sub><sub>ω</sub>* using the old policy *π<sub>θ</sub><sub>old</sub>*. The second variant, an approach also seen in LUFFY (Yan et al., 2025), approximates the external policy’s probability as 1, treating it as a perfect oracle. When using our policy estimation as the external policy, i.e., the third variant, the performance improves by 2.9 points, demonstrating the effectiveness of our policy estimation. Due to the improper integration methods, these variants all show a significant performance gap compared to RL-PLUS.

### 5 Training Stability of RL-PLUS

To validate the training stability of RL-PLUS, we extended the number of training steps on the Qwen2.5-Math-1.5B model to **over 10 times** the original setup. As shown in Figure 4, the model’s key metrics demonstrate excellent stability and continuous performance improvement as training progresses. Specifically, the Average Test Score and Critic Rewards Mean both show a steady upward trend, while the Actor Entropy Loss rapidly converges and stabilizes in a healthy, non-zero range. This reveals an ideal balance: the model’s policy, while becoming more effective (i.e., exploitation), also maintains the necessary policy stochasticity for exploration, thus avoiding prema- ture convergence to a local optimum. These results strongly demonstrate that the RL-PLUS framework possesses outstanding training stability and has the potential for further performance gains through extended training.

<!-- Page 10 -->

Average Test Score Actor Entropy Loss Response Length Critic Rewards Mean

1.0
0.4 1800 0.6
0.8
1600 0.5

0.3
0.6 0.4
1400 Value0.2Value Value Value0.3

0.4
1200

0.1 0.2
0.2
1000 0.1

0.0 0.0 0 1000 2000 3000 4000 5000 6000 0 1000 2000 3000 4000 5000 6000 0 1000 2000 3000 4000 5000 6000 0 1000 2000 3000 4000 5000 6000 Step Step Step Step
RL-PLUS (Qwen2.5-Math-1.5B)

Figure 4: Training Stability of RL-PLUS.

### 6 Conclusion

In this paper, we proposed RL-PLUS, a novel hybrid-policy optimization approach designed to counter the “capability boundary collapse” observed in LLMs trained with RLVR. RL-PLUS ad- dresses this problem by synergizing external data with internal exploitation through two core components: Multiple Importance Sampling to resolve distributional mismatch from external data, and Exploration-Based Advantage Function to incentivize the discovery of correct yet low-probability reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of RL-PLUS. Notably, Pass@k curves and training dynamics demonstrate that our method breaks through the reasoning capability boundary of base model, leading to further performance improvements.

### References

Hongyi James Cai, Junlin Wang, Xiaoyin Chen, and Bhuwan Dhingra. How much backtracking is enough? exploring the interplay of sft and rl in enhancing llm reasoning. *arXiv preprint* *arXiv:2505.24273*, 2025.

Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, and Xiao Wang. Step-wise adaptive integration of supervised fine-tuning and reinforcement learning for task-specific llms. *arXiv preprint arXiv:2505.13026*, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *CoRR*, abs/2110.14168, 2021.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. *arXiv preprint* *arXiv:2502.01456*, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. *arXiv preprint arXiv:2505.22617*, 2025b.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming– the rise of code intelligence. *arXiv preprint arXiv:2401.14196*, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv* *preprint arXiv:2501.12948*, 2025.

<!-- Page 11 -->

Alexander Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, et al. Teaching large language models to reason with reinforcement learning. *arXiv preprint* *arXiv:2403.04642*, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In *ACL (1)*, pp. 3828–3850. Association for Computational Lin- guistics, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In *NeurIPS Datasets and Benchmarks*, 2021.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. *arXiv preprint arXiv:2503.24290*, 2025.

Hugging Face. Openr1-math-220k, February 2025. URL [https://huggingface.co/](https://huggingface.co/) datasets/open-r1/OpenR1-Math-220k.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. *arXiv preprint arXiv:2403.07974*, 2024.

KimiTeam. Kimi k1.5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*,

2025.
Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In *NeurIPS*, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. https: //huggingface.co/datasets/Numinamath, 2024. Hugging Face repository, 13:9.

Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In *ICCV*, pp. 2999–3007. IEEE Computer Society, 2017.

Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, and Dongmei Zhang. Superrl: Reinforcement learning with supervision to boost language model reasoning. *arXiv preprint arXiv:2506.01096*, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Tianyu Pang, Chao Du, and Min Lin. There may not be aha moment in r1-zero-like training — a pilot study. [https://oatllm.notion.site/](https://oatllm.notion.site/) oat-zero, 2025b. Notion Blog.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. *arXiv preprint* *arXiv:2503.20783*, 2025c.

Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Bin Cui, et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. *arXiv preprint arXiv:2506.07527*, 2025.

OpenAI. Openai o1 system card and model report. 2024. Technical report, OpenAI o1 series, available online.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to fol- low instructions with human feedback. *Advances in neural information processing systems*, 35: 27730–27744, 2022.

<!-- Page 12 -->

Martin L Puterman. *Markov decision processes: discrete stochastic dynamic programming*. John Wiley & Sons, 2014.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Di- rani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In *First Conference on Language Modeling*, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li,

Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *CoRR*, abs/2402.03300, 2024.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint* *arXiv: 2409.19256*, 2024.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. *arXiv preprint arXiv:2504.20571*, 2025.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. *Advances in Neural Information Processing Systems*, 37:95266–95290, 2024a.

Zhichao Wang, Bin Bi, Zixu Zhu, Xiangbo Mao, Jun Wang, and Shiyu Wang. Uft: Unifying fine-tuning of sft and rlhf/dpo/una through a generalized implicit reward function. *arXiv preprint* *arXiv:2410.21438*, 2024b.

Jinyang Wu, Chonghua Liao, Mingkuan Feng, Shuai Zhang, Zhengqi Wen, Pengpeng Shao, Huazhe Xu, and Jianhua Tao. Thought-augmented policy optimization: Bridging external guidance and internal capabilities. *arXiv preprint arXiv:2505.15692*, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. *arXiv preprint arXiv:2504.14945*, 2025.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jian- hong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. *arXiv preprint arXiv:2409.12122*, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? *CoRR*, abs/2504.13837, 2025a.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. *arXiv preprint arXiv:2504.05118*, 2025b.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl- zoo: Investigating and taming zero reinforcement learning for open base models in the wild. *arXiv* *preprint arXiv:2503.18892*, 2025.

<!-- Page 13 -->

### A Theoretical Analysis of Multiple Importance Sampling

We provide a rigorous theoretical analysis of the Multiple Importance Sampling (MIS) estimator for policy optimization. First, we dissect the bias and variance issues inherent to standard Importance Sampling (IS) when using data from a single behavior policy. Subsequently, we prove that the MIS estimator is unbiased and analyze its superior variance properties. We show that MIS is robust to the inclusion of suboptimal behavior policies, establishing it as a powerful tool for integrating diverse data sources in policy optimization.

A.1 Preliminaries and Core Assumptions Our analysis is based on the following standard settings and assumptions. Let the objective function be *J*(*θ*) = E*τ ∼π<sub>θ</sub>*[*R*(*τ*)], where *τ* represents a complete trajectory, *R*(*τ*) is its corresponding cumulative return, and *π<sub>θ</sub>* is the target policy we aim to optimize. **Assumption A.1** (Joint Support Coverage)**.** *The support of the target policy π<sub>θ</sub> is covered by the* *union of the supports of all behavior policies {π<sub>β</sub><sub>k</sub>}<sup>K</sup><sub>k</sub>*=1*. Formally,*
[ *K* *supp*(*π<sub>θ</sub>*) *⊆ supp*(*π<sub>β</sub><sub>k</sub>*) *k*=1 *This assumption ensures that any trajectory possible under π<sub>θ</sub> can be sampled with a non-zero* *probability by at least one behavior policy.* **Assumption A.2** (Bounded Rewards)**.** *The trajectory returns are bounded, i.e., for all trajectories* *τ, there exists a constant R<sub>max</sub> such that |R*(*τ*)*|≤ R<sub>max</sub> < ∞. This ensures that all expectations* *and variances are well-defined.*

A.2 Analysis of Bias and Variance in Single-Strategy Importance Sampling When learning from data generated by a single external behavior policy*π<sub>ω</sub>*, the standard IS estimator can suffer from bias and variance problems. We analyze three primary failure modes.
A.2.1 Importance Sampling Estimators We formally define the estimators central to our analysis. We consider a dataset of *N* trajectories. **Definition A.3** (Standard Importance Sampling (IS) Estimator)**.** When all data is sampled from a single behavior policy *π<sub>ω</sub>* (i.e., *K* = 1*,π<sub>β</sub><sub>1</sub>* = *π<sub>ω</sub>*), the standard IS estimator for *J*(*θ*) is:
X *N* <u>1 π<sub>θ</sub>(τ<sub>i</sub>)</u> *Ĵ<sub>IS</sub>*(*θ*) = *R*(*τ<sub>i</sub>*)*,* where *τ<sub>i</sub> ∼ π<sub>ω</sub>* *N* *i*=1 *π<sub>ω</sub>*(*τ<sub>i</sub>*)

**Definition A.4** (Proxy IS Estimator)**.** A biased variant of the IS estimator that uses a proxy policy *π<sub>θ</sub><sub>old</sub>* in the denominator, while the data is sampled from a different policy *π<sub>ω</sub>*:

X *N* <u>1 π<sub>θ</sub>(τ<sub>i</sub>)</u> *Ĵ*proxy(*θ*) = *R*(*τ<sub>i</sub>*)*,* where *τ<sub>i</sub> ∼ π<sub>ω</sub>* *N* *i*=1 *π<sub>θ</sub><sub>old</sub>*(*τ<sub>i</sub>*)

A.2.2 Bias from a Proxy In practice, to mitigate the high variance that occurs when the data-generating policy *π<sub>ω</sub>* is far from the target policy *π<sub>θ</sub>*, one might be tempted to use a different policy, *π<sub>θ</sub><sub>old</sub>*, as the denominator for the importance ratio. This ”proxy” policy is chosen to be closer to *π<sub>θ</sub>* (e.g., a previous iterate of the policy). However, this introduces a systematic bias, as it violates the fundamental principle of importance sampling. **Lemma A.5** (Bias of the IS Estimator with a Proxy)**.** *Assume trajectory data τ<sub>i</sub> is sampled from* *an external policy π<sub>ω</sub>, i.e., τ<sub>i</sub> ∼ π<sub>ω</sub>. If we construct an estimator using a proxy policy π<sub>θ</sub><sub>old</sub> in the* *denominator of the importance weight:*
X *N* <u>π<sub>θ</sub>(τ<sub>i</sub>)</u> *Ĵproxy*(*θ*) = *R*(*τ<sub>i</sub>*) *N* *i*=1 *π<sub>θ</sub><sub>old</sub>*(*τ<sub>i</sub>*)

<!-- Page 14 -->

*then this estimator is **biased** for the true objective J*(*θ*) *whenever the proxy policy π<sub>θ</sub><sub>old</sub> is not iden-* *tical to the true sampling policy π<sub>ω</sub>. The bias is given by:* Z *△* <u>π<sub>ω</sub>(τ)</u> *B*(*θ,ω,θ<sub>old</sub>*)=E<sub>π</sub><sub>ω</sub>[*Ĵproxy*(*θ*)] *− J*(*θ*) = *π<sub>θ</sub>*(*τ*)*R*(*τ*) *−* 1 *dτ* (8) *π<sub>θ</sub><sub>old</sub>*(*τ*)

*Proof.* We compute the expectation of the proxy estimator *Ĵ*proxy(*θ*) under the true data distribution *π<sub>ω</sub>*. The expectation is taken with respect to *τ ∼ π<sub>ω</sub>*. <u>π<sub>θ</sub>(τ)</u> E<sub>π</sub><sub>ω</sub>[*Ĵ*proxy(*θ*)] = E*τ ∼π<sub>ω</sub> R*(*τ*) *π<sub>θ</sub><sub>old</sub>*(*τ*) Z <u>π<sub>θ</sub>(τ)</u> = *π<sub>ω</sub>*(*τ*) *R*(*τ*)*dτ* *π<sub>θ</sub><sub>old</sub>*(*τ*) This is the expected value that the estimator will yield. Crucially, because the sampling distribution *π<sub>ω</sub>*(*τ*) in the integral does not cancel with the denominator R *π<sub>θ</sub><sub>old</sub>*(*τ*), this expression cannot be simplified to the true objective *J*(*θ*) = *π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ*.

The bias of this estimator is its expectation minus the true objective:

*B*(*θ,ω,θ<sub>old</sub>*) = E<sub>π</sub><sub>ω</sub>[*Ĵ*proxy(*θ*)] *− J*(*θ*) Z Z <u>π<sub>θ</sub>(τ)</u> = *π<sub>ω</sub>*(*τ*) *R*(*τ*)*dτ − π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* *π<sub>θ</sub><sub>old</sub>*(*τ*) Z <u>π<sub>θ</sub>(τ)</u> = *π<sub>ω</sub>*(*τ*) *R*(*τ*) *− π<sub>θ</sub>*(*τ*)*R*(*τ*) *dτ* *π<sub>θ</sub><sub>old</sub>*(*τ*) Z <u>π<sub>ω</sub>(τ)</u> = *π<sub>θ</sub>*(*τ*)*R*(*τ*) *−* 1 *dτ* *π<sub>θ</sub><sub>old</sub>*(*τ*) The final expression for the bias is zero if and only if *π<sub>ω</sub>*(*τ*) = *π<sub>θ</sub><sub>old</sub>*(*τ*) for all relevant trajectories. If the external data policy *π<sub>ω</sub>* differs significantly from the proxy policy *π<sub>θ</sub><sub>old</sub>*, this ratio will deviate substantially from 1, leading to a large, systematic bias.

A.2.3 Bias from Support Mismatch Even when using the correct data-generating policy *π<sub>ω</sub>* in the denominator, the standard IS estimator is biased if the support of *π<sub>ω</sub>* does not fully cover the support of the target policy *π<sub>θ</sub>*. **Lemma A.6** (Bias of the Standard IS Estimator from Support Mismatch)**.** *When using data sampled* *from an external policy π<sub>ω</sub> to estimate the objective J*(*θ*)*, if the support condition supp*(*π<sub>θ</sub>*) *̸⊆*
<u>1</u> P<sub>N</sub> <u>π<sub>θ</sub>(τ<sub>i</sub></u> *supp*(*π<sub>ω</sub>*) *is not met, the standard IS estimator ĴIS*(*θ*) =<u><sup>)</sup></u> *N i*=1 *π<sub>ω</sub>*(*τ<sub>i</sub>* <sub>)</sub> *R*(*τ<sub>i</sub>*) *(where τ<sub>i</sub> ∼ π<sub>ω</sub>) is* *biased. The bias relative to the true objective is:* Z *△* *B*(*θ,ω*)=E<sub>π</sub><sub>ω</sub>[*ĴIS*(*θ*)] *− J*(*θ*) = *− π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* (9) *τ ∈supp*(*π<sub>θ</sub>*)*\supp*(*π<sub>ω</sub>*)

*Proof.* The expectation of the IS estimator is calculated as follows: <u>π<sub>θ</sub>(τ)</u> E<sub>π</sub><sub>ω</sub>[*Ĵ<sub>IS</sub>*(*θ*)] = E*τ ∼π<sub>ω</sub> R*(*τ*) *π<sub>ω</sub>*(*τ*) Z <u>π<sub>θ</sub>(τ)</u> = *π<sub>ω</sub>*(*τ*) *R*(*τ*)*dτ* *τ ∈*supp(*π<sub>ω</sub>*)*π<sub>ω</sub>*(*τ*) Z = *π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* *τ ∈*supp(*π<sub>ω</sub>*)*∩*supp(*π<sub>θ</sub>*) The true objective *J*(*θ*) can be decomposed over the same domains: Z *J*(*θ*) = *π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* *τ ∈*supp(*π<sub>θ</sub>*) Z Z = *π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* + *π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* *τ ∈*supp(*π<sub>θ</sub>*)*∩*supp(*π<sub>ω</sub>*) *τ ∈*supp(*π<sub>θ</sub>*)*\*supp(*π<sub>ω</sub>*)

<!-- Page 15 -->

The bias is the difference between these two quantities. This term represents the expected return from trajectories possible under*π<sub>θ</sub>* but not under*π<sub>ω</sub>*, and it is zero if and only if the support condition holds.

A.2.4 Variance Divergence of the Importance Ratio **Lemma A.7** (Variance of the IS Ratio)**.** *Even if the support condition is satisfied, the variance of* *the importance ratio r<sup>ω</sup>*(*τ*) =
*π* <u>π</u> <sub>ω</sub> <u><sub>θ</sub></u><u>(</u> ( <u>τ</u> *τ* <u>)</u> ) *can become extremely large when the target policy π<sub>θ</sub> and* *behavior policy π<sub>ω</sub> are dissimilar. Precisely, the variance is equal to the **Chi-squared divergence*** *between the two policies:* Var<sub>π</sub><sub>ω</sub>(*r<sup>ω</sup>*) = *χ²*(*π<sub>θ</sub>,π<sub>ω</sub>*)

*Proof.* The variance of the ratio is Var<sub>π</sub><sub>ω</sub>(*r<sup>ω</sup>*) = E<sub>π</sub><sub>ω</sub>[(*r<sup>ω</sup>*)²] *−* (E<sub>π</sub><sub>ω</sub>[*r<sup>ω</sup>*])². Under the support coverage condition, the expectation of the ratio is E<sub>π</sub><sub>ω</sub>[*r<sup>ω</sup>*] = 1. We compute the second moment:

Z2Z <sup>ω</sup> <u>π<sub>θ</sub>(τ) π<sub>θ</sub>(τ)²</u> E<sub>π</sub><sub>ω</sub>[(*r*)²] = *π<sub>ω</sub>*(*τ*) *dτ* = *dτ.* *π<sub>ω</sub>*(*τ*) *π<sub>ω</sub>*(*τ*)

R <u>(π<sub>θ</sub>(τ)−π<sub>ω</sub>(τ))²</u> R <u>π<sub>θ</sub>(τ)²</u> R R By noting that *χ²*(*π<sub>θ</sub>,π<sub>ω</sub>*) = *π<sub>ω</sub>*(*τ*) *dτ* = *π<sub>ω</sub>*(*τ*) *dτ −* 2 *π<sub>θ</sub>*(*τ*)*dτ* + *π<sub>ω</sub>*(*τ*)*dτ* = E<sub>π</sub><sub>ω</sub>[(*r<sup>ω</sup>*)²] *−* 2 + 1 = E<sub>π</sub><sub>ω</sub>[(*r<sup>ω</sup>*)²] *−* 1, we have:

#### E<sub>π</sub><sub>ω</sub>[(r<sup>ω</sup>)²] = χ²(π<sub>θ</sub>,π<sub>ω</sub>) + 1.

#### Therefore, the variance is:

Var<sub>π</sub><sub>ω</sub>(*r<sup>ω</sup>*) = (*χ²*(*π<sub>θ</sub>,π<sub>ω</sub>*) + 1) *−* 1<sup>2</sup> = *χ²*(*π<sub>θ</sub>,π<sub>ω</sub>*)*.*

Both the *χ²*-divergence and the more commonly known KL-divergence (*D<sub>KL</sub>*(*π<sub>θ</sub>∥π<sub>ω</sub>*)) are measures of dissimilarity between distributions (both are instances of f-divergences). A large value in one typically implies a large value in the other. Therefore, as the policies diverge, there are often regions where *π<sub>θ</sub>*(*τ*) *≫ π<sub>ω</sub>*(*τ*). In these regions, the ratio *r<sup>ω</sup>*(*τ*) becomes extremely large, causing the variance to explode.

A.3 Bias Advantage of the MIS Estimator The standard MIS estimator is proven to be unbiased. In practice, a common and highly practical scenario involves using external data collected from the behavior policy, *π<sub>ω</sub>*, which may be far from the target policy *π<sub>θ</sub>*. To stabilize estimates, one can introduce a proxy policy, *π<sub>θ</sub><sub>old</sub>* (e.g., a previous iterate of *π<sub>θ</sub>*), into the denominator of the importance weight. This creates a powerful estimator that deliberately accepts a small, controlled bias in exchange for a substantial reduction in variance. We now formally analyze the bias advantage of this practical MIS estimator compared to the aforementioned approaches. *Remark* A.8 (Controlled Bias vs. Explosive Bias of Proxy IS)*.* This estimator is motivated by variance reduction. While biased, its bias is far more controlled than that of the proxy estimator from Lemma A.5, which uses only *π<sub>θ</sub><sub>old</sub>* in the denominator. A comparison of their bias-inducing factors is revealing:
<u>π<sub>ω</sub>(τ)−π<sub>θ</sub><sub>old</sub> (τ)</u>

- **Proxy IS Factor:** *f*proxy(*τ*) =
*π<sub>θ</sub><sub>old</sub>* (*τ*)

<u>π<sub>ω</sub>(τ)−π<sub>θ</sub><sub>old</sub> (τ)</u>

- **Practical MIS Factor:** *f<sub>MIS</sub>*(*τ*) =
*π<sub>ω</sub>*(*τ*)+*π<sub>θ</sub><sub>old</sub>* (*τ*)

When *π<sub>θ</sub><sub>old</sub>*(*τ*) *→* 0 for a trajectory that is plausible under *π<sub>ω</sub>*, the proxy IS factor can become arbitrarily large, leading to an uncontrolled, potentially infinite bias. In contrast, the practical MIS factor is a normalized difference and is strictly bounded within (*−*1*,*1). The presence of the true sampling distribution *π<sub>ω</sub>*(*τ*) in the denominator acts as a crucial **guardrail**, preventing the weights from exploding and ensuring the bias remains bounded.

<!-- Page 16 -->

*Remark* A.9 (Overcoming Support Mismatch)*.* The practical MIS estimator also offers a robust solution to the critical problem of support mismatch (Lemma A.6), where supp(*π<sub>θ</sub>*) *̸⊆* supp(*π<sub>ω</sub>*). The practical MIS estimator mitigates this by relying on the weaker joint support assumption, supp(*π<sub>θ</sub>*) *⊆* supp(*π<sub>ω</sub>*) *∪*supp(*π<sub>θ</sub><sub>old</sub>*). By including *π<sub>θ</sub><sub>old</sub>*, it explicitly covers the full support of *π<sub>θ</sub>* and eliminates the truncation error. In its place, it introduces a **distortion error**, given by the bounded bias term derived above. In essence, this estimator replaces a potentially infinite and unrecoverable truncation error, i.e, Z

*B*support= *− π<sub>θ</sub>*(*τ*)*R*(*τ*)*dτ* *τ ∈*supp(*π<sub>θ</sub>*)*\*supp(*π<sub>β</sub><sub>1</sub>*), with a manageable and bounded distortion error, making it a far more robust choice for real-world applications.

A.4 Variance Advantage and Robustness of the MIS Estimator The core advantage of MIS lies in its variance control and robustness, and we formally analyze below. **Theorem A.10** (Variance Robustness of MIS)**.** *So long as there is at least one policy in the behavior* *pool {π<sub>β</sub><sub>k</sub>} (e.g., π<sub>β</sub><sub>k</sub><sup>∗</sup>) that is a good approximation of the target policy π<sub>θ</sub> (i.e., π<sub>β</sub><sub>k</sub><sup>∗</sup> ≈ π<sub>θ</sub>), the* *variance of the MIS estimator will be low. The estimator is insensitive to other arbitrarily ”bad”* *behavior policies in the pool.* *Proof.* We qualitatively analyze the behavior of the MIS weight *w*(*τ*) =<u>P</u>
<u>π</u> *α* <u><sub>θ</sub></u> <sub>j</sub> <u>(</u> *π* <u>τ</u> <sub>β</sub> <u>)</u> <sub>(τ)</sub>, whose mag- *j* <sub>j</sub>

#### nitude directly drives the variance.

**Dilemma of Standard IS:** Assume we only use a ”bad” policy *π<sub>β</sub><sub>m</sub>*, for which the probability density approaches zero in some region *S<sub>bad</sub>* (*π<sub>β</sub><sub>m</sub>*(*τ*) *→* 0), while the target policy has non-negligible density there (*π<sub>θ</sub>*(*τ*) *> ϵ*). In this case, the standard IS ratio *π* <u>π</u> <sub>β</sub> <u><sub>θ</sub></u><u>(τ</u> (*τ* <u>)</u> ) would diverge in *S<sub>bad</sub>*, causing <sub>m</sub> the variance to explode.

**Advantage of MIS:** Now, we add a ”good” policy P*π<sub>β</sub><sub>k</sub><sup>∗</sup>* to the pool, satisfying *π<sub>β</sub><sub>k</sub><sup>∗</sup> ≈ π<sub>θ</sub>*. The denominator of the MIS weight is a mixture density:*jα<sub>j</sub>π<sub>β</sub><sub>j</sub>* (*τ*). Even in the problematic region *S<sub>bad</sub>*, the denominator contains at least one term, *α<sub>k</sub><sup>∗</sup>π<sub>β</sub><sub>k∗</sub>* (*τ*) *≈ α<sub>k</sub><sup>∗</sup>π<sub>θ</sub>*(*τ*), which is positive and non-negligible. The MIS weight is therefore effectively bounded:

*π<sub>θ</sub>*(*τ*) *π<sub>θ</sub>*(*τ*) *π<sub>θ</sub>*(*τ*) 1 *N* *w*(*τ*) = P *≈ ≤ ≈* = *α<sub>k</sub><sup>∗</sup>π<sub>β</sub><sub>k∗</sub>* (*τ*) +*j̸*=*k<sup>∗</sup> α<sub>j</sub>π<sub>β</sub><sub>j</sub>* (*τ*) *α<sub>k</sub><sup>∗</sup>π<sub>θ</sub>*(*τ*) +*... α<sub>k</sub><sup>∗</sup>π<sub>β</sub><sub>k∗</sub>* (*τ*) *α<sub>k</sub><sup>∗</sup> n<sub>k</sub><sup>∗</sup>*

The weight is bounded from above by a constant that does not depend on the ratio of policies. The summation in the denominator acts as a **”variance guardrail”**, preventing the sampling deficiencies of any single policy from destabilizing the entire estimate.

*Remark* A.11 (Practical Implications)*.* The robustness of MIS is especially critical when combining internal data (from an old policy *π<sub>old</sub>*) and external data (from *π<sub>ω</sub>*). The policy *π<sub>old</sub>* ensures that the KL-divergence from the current policy *π<sub>θ</sub>* is kept within a controllable range. This ensures that there is always a ”good” policy in the pool. Therefore, even if the external policy *π<sub>ω</sub>* is far from *π<sub>θ</sub>*, the MIS estimator can stabilize the variance through the presence of *π<sub>old</sub>*. MIS achieves a ”soft”, unbiased form of variance control by mixing policy densities in the denominator. This adaptive weighting mechanism makes MIS a theoretically sound and highly effective choice for integrating heterogeneous data sources in policy optimization.

A.5 Optimal Bayesian Estimation of the Behavior Policy under Model Uncertainty We need a method to construct a robust estimator for *π<sub>ω</sub>* that acknowledges our uncertainty. We propose a principled approach based on Bayesian decision theory to derive an optimal estimator for *π<sub>ω</sub>* that explicitly balances our belief in the proxy model *π<sub>θ</sub><sub>old</sub>* with a model of maximal uncertainty. We frame the task of selecting an estimator *π*ˆ<sub>ω</sub> as a Bayesian decision problem.
- **State of Nature**: The true, unknown behavior policy *π<sub>ω</sub>*.

<!-- Page 17 -->

- **Action**: Our choice of an estimator *π*ˆ<sub>ω</sub> for *π<sub>ω</sub>*.
- **Model Space** *M*: The set of candidate models for *π<sub>ω</sub>*. Given our limited knowledge, we define a minimal, discrete model space that captures the dichotomy between our specific knowledge and our uncertainty.
- **Loss Function** *L*(ˆ*π<sub>ω</sub>,π<sub>ω</sub>*): A function that quantifies the error of our estimator. A standard
R choice is the squared L2-error, *L*(ˆ*π<sub>ω</sub>,π<sub>ω</sub>*) = (ˆ*π<sub>ω</sub>*(*τ*) *− π<sub>ω</sub>*(*τ*))² *dτ*.

Our goal is to find the estimator *π*ˆ<sub>ω</sub> that minimizes the **Bayes risk**, which is the expected loss with respect to our prior beliefs about the state of nature. **Theorem A.12** (Bayes-Optimal Policy Estimator)**.** *Let the model space for the unknown behavior* *policy π<sub>ω</sub> be composed of two candidate models:*

- *The specific proxy policy, π<sub>θ</sub><sub>old</sub>, representing our available, specific information.*
- *A non-informative uniform policy, U*(*τ*)*, representing maximal uncertainty.*
R *Let the trajectory space T have a finite volume V* = *T* *dτ, such that U*(*τ*) = 1*/V. Under the* ***Principle of Indifference**, we assign equal prior probabilities to these models, i.e., P*(*π<sub>ω</sub>* = *π<sub>θ</sub><sub>old</sub>*) = *P* (*π<sub>ω</sub>* = *U*) = 1*/*2*. Then, the estimator π*ˆ<sub>ω</sub> *that minimizes the Bayes risk (expected L2 error) is the* *Bayesian model average:*

<sup>∗</sup> <u>1 1</u> *π*ˆ<sub>ω</sub>(*τ*) = *π<sub>θ</sub><sub>old</sub>*(*τ*) + *U*(*τ*) 2 2

*Proof.* The Bayes risk of an estimator *π*ˆ<sub>ω</sub> is the expectation of the loss function over the prior distribution of *π<sub>ω</sub>*:

*R*(ˆ*π<sub>ω</sub>*) = E<sub>π</sub><sub>ω</sub> [*L*(ˆ*π<sub>ω</sub>,π<sub>ω</sub>*)] X = *L*(ˆ*π<sub>ω</sub>,π<sup>′</sup>*)*P* (*π<sub>ω</sub>* = *π<sup>′</sup>*) *π<sup>′</sup>∈{π<sub>θ</sub><sub>old</sub>,U}* Z Z <u>1 1</u> = (ˆ*π<sub>ω</sub>*(*τ*) *− π<sub>θ</sub><sub>old</sub>*(*τ*))² *dτ* + (ˆ*π<sub>ω</sub>*(*τ*) *−U*(*τ*))² *dτ* 2 2

To find the optimal estimator *π*ˆ<sub>ω</sub><sup>∗</sup> that minimizes this risk, we can use the calculus of variations or simply note that the integrand is a sum of squared errors, which is minimized point-wise. For any given trajectory *τ*, we seek to minimize:

*f*(ˆ*π<sub>ω</sub>*(*τ*)) = (ˆ*π<sub>ω</sub>*(*τ*) *− π<sub>θ</sub><sub>old</sub>*(*τ*))² + (ˆ*π<sub>ω</sub>*(*τ*) *−U*(*τ*))²

This is a simple quadratic function of the scalar value *π*ˆ<sub>ω</sub>(*τ*). We find the minimum by taking the derivative with respect to *π*ˆ<sub>ω</sub>(*τ*) and setting it to zero:

<u>∂f</u> = 2 (ˆ*π<sub>ω</sub>*(*τ*) *− π<sub>θ</sub><sub>old</sub>*(*τ*)) + 2 (ˆ*π<sub>ω</sub>*(*τ*) *−U*(*τ*)) = 0 *∂π*ˆ<sub>ω</sub>(*τ*) 2ˆ*π<sub>ω</sub>*(*τ*) *− π<sub>θ</sub><sub>old</sub>*(*τ*) *−U*(*τ*) = 0 <u>1</u> *π*ˆ<sub>ω</sub>(*τ*) = (*π<sub>θ</sub><sub>old</sub>*(*τ*) + *U*(*τ*)) 2 This result gives the point-wise minimizer. Integrating over all *τ* confirms that the optimal estimator function is: <sup>∗</sup> <u>1 1</u> *π*ˆ<sub>ω</sub>(*τ*) = *π<sub>θ</sub><sub>old</sub>*(*τ*) + *U*(*τ*) 2 2 This estimator is known as the **Bayes estimator** under quadratic loss for this prior. It is optimal in the sense that no other estimator has a lower expected error, given our stated beliefs about the possible models for R *π<sub>ω</sub>*. It is straightforward to verify that R R *π*ˆ<sub>ω</sub><sup>∗</sup> (*τ*) is a valid probability density function, as *π*ˆ<sub>ω</sub><sup>∗</sup> (*τ*)*dτ* = 1 *π<sub>θ</sub><sub>old</sub>*(*τ*)*dτ* + 1 *U*(*τ*)*dτ* = <sup>1</sup> (1) + <sup>1</sup> (1) = 1.

**Assumption A.13** (Unit-Volume Trajectory Space)**.** *For analytical tractability, we assume the tra-*R *jectory space T is normalized to have unit volume, i.e.,* *T* *dτ* = 1*. Under this assumption, the* *maximum-entropy (uniform) distribution is U*(*τ*) = 1 *for all τ ∈T .*

<!-- Page 18 -->

*Remark* A.14 (Robustness and Connection to Regularization)*.* Theorem A.12 provides a rigorous justification for what is, in essence, a form of regularization. The resulting estimator *π*ˆ<sub>ω</sub><sup>∗</sup> is a mixture model that hedges against the deficiencies of *π<sub>θ</sub><sub>old</sub>*. The uniform component *U*(*τ*) acts as a **“safety** **net”** or a **“defensive distribution”**. By ensuring that *π*ˆ<sub>ω</sub><sup>∗</sup> (*τ*) *≥* 2 <u>1</u> *V* *>* 0 for all *τ*, it guarantees that the importance sampling ratio’s denominator is strictly positive and bounded away from zero. This prevents the variance of the importance weights from exploding, a critical property for stable off-policy learning.

The assumption *P* = 1*/*2 reflects a state of maximal ambiguity between the specific information we have (*π<sub>θ</sub><sub>old</sub>*) and the general uncertainty we face (*U*). It is the most conservative and robust choice when we cannot quantify our confidence in *π<sub>θ</sub><sub>old</sub>*. Thus, forming the estimator as their mean is the theoretically optimal strategy to navigate this uncertainty.

### B Theoretical Analysis of the Exploration-Based Advantage

We provide a theoretical justification for the proposed Exploration-Based Advantage function. We prove that it adaptively focuses the policy gradient updates on high-value, hard-to-explore actions.

B.1 Gradient Analysis We now analyze the effect of this advantage function on the policy gradient. **Lemma B.1** (Gradient Contribution of a Single Timestep)**.** *The gradient update for the policy pa-* *rameters θ induced by the action e<sub>i,t</sub> from a correct, high-reward trajectory i is given by:*
∆*θ<sub>i,t</sub> ∝∇<sub>θ</sub>* log*π<sub>θ</sub>*(*e<sub>i,t</sub>|q,ei,<t*) *· A<sub>i</sub> ·* (1 *− π<sub>θ</sub>*(*e<sub>i,t</sub>|q,ei,<t*))<sup>γ</sup>

*Proof.* The gradient update for the policy objective at timestep *t* is proportional to *∇<sub>θ</sub>* log*π<sub>θ</sub>*(*e<sub>i,t</sub>|...*) *· A<sup>c</sup><sub>i,t</sub>*. Substituting the definition of *A<sup>c</sup><sub>i,t</sub>*, we have:

∆*θ<sub>i,t</sub> ∝∇<sub>θ</sub>* log*π<sub>θ</sub>*(*e<sub>i,t</sub>|...*) *· A<sub>i</sub> · C<sub>i,t</sub>*(*θ*)

By the definition of *C<sub>i,t</sub>*(*θ*) and the properties of the detach operator, the term *C<sub>i,t</sub>*(*θ*) is treated as a scalar weight during backpropagation. Substituting its definition yields the result.

This lemma establishes the precise form of the gradient update. We now prove our main result: that this form adaptively focuses learning. **Theorem B.2** (Adaptive Gradient Focusing)**.** *Given a high-reward trajectory where A<sub>i</sub> >* 0*, the* *gradient magnitude of the update induced by A<sup>c</sup><sub>i,t</sub> is inversely related to the policy’s confidence* *π<sub>θ</sub>*(*e<sub>i,t</sub>|...*)*. The update is amplified for ”hard” (low-probability) actions and suppressed for* *”easy” (high-probability) actions.*

*Proof.* We analyze the asymptotic behavior of the scaling factor on the gradient, based on Lemma

B.1. Let *p<sub>t</sub>* = *π<sub>θ</sub>*(*e<sub>i,t</sub>|...*) denote the policy’s probability for the correct action at time step *t*. The gradient is scaled by the factor *A<sub>i</sub> ·* (1 *− p<sub>t</sub>*)<sup>γ</sup>. We consider two cases for the value of *p<sub>t</sub>*. **Case 1: Hard-to-Explore Correct Action.** In this case, the policy assigns a low probability to the correct action, i.e., *p<sub>t</sub> →* 0. The exploration weight becomes:
lim *C<sub>i,t</sub>*(*θ*) = lim (1 *− p<sub>t</sub>*)<sup>γ</sup> = 1 *p<sub>t</sub>→*0 *p<sub>t</sub>→*0

The resulting gradient update, ∆*θ<sub>i,t</sub> ∝∇<sub>θ</sub>* log*p<sub>t</sub> · A<sub>i</sub>*, retains its full magnitude. The learning signal from this valuable, unexplored action is preserved.

**Case 2: Easy-to-Explore Correct Action.** In this case, the policy is already confident about the correct action, i.e., *p<sub>t</sub> →* 1. The exploration weight becomes:

lim *C<sub>i,t</sub>*(*θ*) = lim (1 *− p<sub>t</sub>*)<sup>γ</sup> = 0 *p<sub>t</sub>→*1 *p<sub>t</sub>→*1

The resulting gradient update vanishes: ∆*θ<sub>i,t</sub> →* 0. The model effectively ignores updates from examples it has already mastered.

<!-- Page 19 -->

**Conclusion.** It demonstrates that the optimization process is focused on the gradients from actions where the policy is incorrect or uncertain, thereby prioritizing the learning of new knowledge. This proves that the advantage function leads to adaptive gradient focusing.

AMC Test Score Olympiad Bench Test Score AIME Test Score Minerva Test Score

0.6 0.45
0.6 0.30 0.40
0.5
0.25
0.5 0.35
0.4
0.20
Score0.4 Score Score Score0.30

0.3 0.15
0.25
0.3
0.10
0.2 0.20
0.2
0.05
0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600 Step Step Step Step GRPO Externel Data with / Externel Data with / old RL-PLUS

Figure 5: Detailed Training dynamics of RL-PLUS and other baselines.

### C Effect of hyperparameter γ

Our systematic investigation into the hyperparameter *γ* in RL-PLUS, illustrated in Figure 6, reveals two key findings. First, the model demonstrates considerable robustness, as its performance fluctu- ates only small across the tested range of *γ*. Second, RL-PLUS consistently surpasses the GRPO baseline across all math reasoning benchmarks, irrespective of the specific value of *γ*. Further analysis highlights a distinct trend: the model uniformly achieves its peak performance when *γ*=0.5. This optimal value holds not only for the Average test score but also across all individual benchmarks, including AMC, Olympiad, AIME, Minerva, and Math. This suggests that an intermediate value for *γ* strikes an effective balance in the model’s learning process. While the model is not highly sensi- tive to this parameter, the clear peak establishes *γ*=0.5 as a strong default, and there is still potential room for improvement with fine-grained tuning of *γ*.

**AMC Test Score Olympiad Bench Test Score AIME Test Score** GRPO GRPO GRPO 35 70 67.8 67.8 68.1 60 58.5 33.3

65.8 56.0 56.6 56.0
30 28.9 28.6 60 28.3 50 50 25 40 40 20 Score (%) Score (%)30 Score (%) 30 15 20 20 10 10 10 5

0 0 0 =0 =0.2 =0.5 =1 =0 =0.2 =0.5 =1 =0 =0.2 =0.5 =1 **Minerva Test Score Math Test Score Average Test Score** 50GRPO 100GRPO GRPO

43.8 88.8 88.9 90.2 88.5 60
56.3 56.6
58.8
40.4 40.6 39.8 55.5
40 80 50

30 60 40

Score (%) Score (%) Score (%)30 20 40 20 10 20 10

=0 =0.2 =0.5 =1 =0 =0.2 =0.5 =1 =0 =0.2 =0.5 =1

Figure 6: Effect of hyperparameter *γ* in RL-PLUS.

<!-- Page 20 -->

**Prompt: [System Prompt]** This is the problem: Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.

**Correct Answer**: 809 **GRPO** **Response**: To determine the number of positive integers (n) less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play, we find the number of multiples of 5 up to 2024.

The largest multiple of 5 less than or equal to 2024 is (2020 = 5 \times 404), so there are 405 multiples of 5 from 0 to 2024 (including 0). Therefore, the number of such (n) is (405).

The final answer is (**\boxed{405}**).

**SFT+GRPO** **Response**: To determine the number of positive integers (n) less than or equal to 2024 for which Bob can always win the game regardless of Alice's play, we need to analyze the losing positions. A position is a losing position if every move from it leads to a winning position for the opponent.

Through recursive analysis, we find that the losing positions are those where (n \equiv 0) or (n \equiv 2 \pmod{3}). This means the numbers that are congruent to 0 or 2 modulo 3 are losing positions for the player about to move.

To count the numbers from 1 to 2024 that are not congruent to 1 modulo 3, we calculate the number of numbers in each congruence class modulo 3:

1. Numbers congruent to 0 modulo 3 between 1 and 2024: (\left\lfloor \frac{2024}{3} \right\rfloor = 674)
2. Numbers congruent to 2 modulo 3 between 1 and 2024: (\left\lfloor \frac{2024 - 2}{3} \right\rfloor + 1 = 674 +1= 675) Adding these together gives the total number of losing positions: [674 + 675 = 1349] Thus, the number of positive integers
(n) less than or equal to 2024 for which Bob can always win is (**\boxed{1349}**).
**RL-PLUS** **Response**: To determine the number of positive integers (n) less than or equal to 2024 for which Bob has a strategy to guarantee a win regardless of Alice's play, we analyze the game where Alice and Bob take turns removing either 1 or 4 tokens from a stack of (n) tokens. The key is to identify the losing positions for Alice,

i.e., positions where Bob can always win. We use the concept of winning and losing positions. A position is a losing position if the player whose turn it is will lose if the opponent plays optimally. We find that the losing positions for Alice are those where (n \equiv 0) or (2 \mod 5). This is because: -If (n \equiv 0 \mod 5), any move (subtracting 1 or 4) results in a position (n - 1) or (n - 4), both of which are winning positions. -If (n \equiv 2 \mod 5), any move results in a position (n - 1) or (n - 4), both of which are winning positions. Thus, the losing positions for Alice are (n \equiv 0) or (2 \mod 5). We count the number of such (n) up to 2024.
1. **Counting (n \equiv 0 \mod 5)**:
- The sequence is (5, 10, 15, \ldots,
2020).
- The number of terms is (\left\lfloor
\frac{2024}{5} \right\rfloor = 404).

2. **Counting (n \equiv 2 \mod 5)**:
- The sequence is (2, 7, 12, \ldots,
2022).
- The number of terms is (\left\lfloor
\frac{2024-2}{5} \right\rfloor + 1 =

405). Adding these counts together, the total number of positive integers (n \leq
2024) for which Bob has a winning strategy is (404 + 405 = 809). Thus, the final answer is (**\boxed{809}**).
### D Case Study

rect conclusion. It completely misinterprets the

Figure 7: A case of RL-PLUS compared with baselines GRPO and SFT+GRPO.

Figure 7 presents a typical case study that visually contrasts the performance of RL-PLUS with the

baseline methods, GRPO and SFT+GRPO. In this case, RL-PLUS demonstrates a significant advantage in both logical rigor and computational precision. Specifically, GRPO, while touching upon a part of the core issue by identifying ‘multiples of 5’ as part of the losing positions, demonstrates an incomplete understanding. It fails to identify the other critical condition, thus arriving at an incor-SFT+GRPO’s approach is fundamentally flawed. game-theoretic model of the problem, erroneously applying an irrelevant ’modulo 3’ logic, causing its reasoning to be incorrect from the outset. The performance of RL-PLUS is exemplary. It be-

<!-- Page 21 -->

gins by accurately identifying the problem as a game of finding P-positions (second-player winning positions). Subsequently, through deductive reasoning, it successfully derives the complete pattern for the set of losing positions: when *n ≡* 0 or 2 *mod* 5. Finally, it proceeds with a clear, step-by- step calculation for both conditions, sums them accurately, and arrives at the correct answer. This case provides compelling evidence that RL-PLUS possesses a more profound and comprehensive multi-step reasoning capability.

### E Experimental Setup

**Training Details.** All experiments are conducted on 8 NVIDIA A100 80G GPUs. By default, we use the Qwen2.5-Math-7B model (Yang et al., 2024) as the base model in our experiments. For our training, we use a subset of OpenR1-Math-220k (Hugging Face, 2025), which contains 45,000 prompts with correct reasoning trajectories annotated by Deepseek-R1, and change the rope theta of Qwen2.5-Math-7B from 10000 to 40000 and extend the window size to 16384, following previous work (Yan et al., 2025). In implementing the RL algorithm, we leverage the VeRL framework (Sheng et al., 2024). We set the batch size to 128, the mini-batch size to 64, and the maximum training epoch to 2. For each problem, we use 8 rollout trajectories, with a maximum response length of 8192 tokens. For our approach, one of the model-generated rollouts is replaced with a correct reasoning trajectory from the training dataset. It is important to note that we ensure all other RL algorithms maintain the same parameter settings as RL-PLUS to guarantee a fair comparison. For the hyperparameter *γ*, we set it to 0.5 in all experiments by default. To validate the applicability of RL-PLUS on various base LLMs, we additionally extend RL-PLUS to other base models, including LLaMA-3.1-8B-instruct, Deepseek-Math-7B-instruct, and Qwen2.5-Math-1.5B.

**Evaluation.** In line with established practices, we evaluate the performance of RL-PLUS on a comprehensive suite of standard mathematical reasoning benchmarks, including GSM8K (Cobbe et al., 2021), MATH500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024), as well as on competition-level benchmarks such as AIME 2024 (Li et al., 2024) and AMC 2023 (Li et al., 2024). Additionally, although our training focuses on math, we extend our evaluation to out-of-domain (OOD) tasks to assess the robustness and generalization capabilities of our approach. The OOD datasets include ARC-c (Clark et al., 2018)(Open-Domain Reasoning), GPQA-diamond (Rein et al., 2024) (Science Graduate Knowledge), MMLU-Pro (Wang et al., 2024a) (Reasoning-focused Questions from Academic Exams and Textbooks), as well as three code generation datasets: HumanEval (Chen et al., 2021), LeetCode (Guo et al., 2024), and LiveCodeBench (Jain et al., 2024). During evaluation, we set the sampling temperature to 0.6 and report the average pass@1 score over 5 runs by default.

**Baselines.** We compare our approach with two categories of baselines, all trained upon the same base model. The first category comprises eight recently proposed RLVR methods, including: 1) **SimpleRL** (Zeng et al., 2025) and 2) **OpenReasoner-Zero** (Hu et al., 2025) are two open-source RL implementations that train starting from the base model using rule-based rewards. SimpleRL em- ploys a token-level, length-rectified GRPO algorithm, while OpenReasoner-Zero utilizes the PPO algorithm. 3) **PRIME** (Cui et al., 2025a) introduces an implicit process reward based on outcome labels during RL. 4) **Oat-Zero** (Liu et al., 2025b) modifies the GRPO algorithm by removing the standard deviation from the advantage computation and eliminating token-level normalization in the policy loss calculation. 5) **DAPO** (Yu et al., 2025) optimizes GRPO algorithm by introducing four operations: Clip-Higher, Dynamic Sampling, a Token-Level Policy Gradient Loss, and Over-long Reward Shaping. 6) **LUFFY** (Yan et al., 2025) leverages off-policy reasoning trajectories to augment GRPO. 7) **TAPO** (Wu et al., 2025) integrates reasoning templates into GRPO sampling process to enhance the model’s internal reasoning capabilities. 8) **ReLIFT** (Ma et al., 2025) performs RL and SFT alternately during training. The second category consists of four straightforward baselines: 1) **SFT**, supervised fine-tuning using external reasoning trajectory data. 2) **GRPO** (Shao et al., 2024), training with GRPO algorithm on question-answer pairs. 3) **SFT+GRPO**, a common RL cold-start approach that performs SFT before RL training. 4) **GRPO w/ SFT Loss**, jointly optimizes the GRPO objective and SFT loss during training.