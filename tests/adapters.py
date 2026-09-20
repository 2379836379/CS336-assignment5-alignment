from __future__ import annotations

import os
import json
import random
import re
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    if len(prompt_strs) != len(output_strs): raise ValueError("length mismatch")
    seqs=[]; starts=[]; ends=[]
    for p, o in zip(prompt_strs, output_strs):
        pt=tokenizer.encode(p, add_special_tokens=True); ot=tokenizer.encode(o, add_special_tokens=False)
        seqs.append(pt+ot); starts.append(len(pt)); ends.append(len(pt)+len(ot))
    m=max((len(x) for x in seqs), default=1); pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    arr=[x+[pad]*(m-len(x)) for x in seqs]
    input_ids=torch.tensor([x[:-1] for x in arr],dtype=torch.long); labels=torch.tensor([x[1:] for x in arr],dtype=torch.long)
    mask=torch.zeros_like(labels,dtype=torch.bool)
    for i,(a,b) in enumerate(zip(starts,ends)): mask[i,max(0,a-1):max(0,b-1)]=True
    return {"input_ids":input_ids,"labels":labels,"response_mask":mask}


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    logits=model(input_ids=input_ids).logits; lp=torch.log_softmax(logits,dim=-1)
    out={"log_probs":lp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)}
    if return_token_entropy: out["token_entropy"]=-(lp.exp()*lp).sum(-1)
    return out


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    scores=[reward_fn(r,g) for r,g in zip(rollout_responses,repeated_ground_truths)]
    rewards=torch.tensor([float(x["reward"]) for x in scores],dtype=torch.float32); n=max(1,len(scores))
    return rewards,{"reward_mean":float(rewards.mean()) if len(rewards) else 0.0,"format_reward_mean":sum(float(x.get("format_reward",0)) for x in scores)/n,"answer_reward_mean":sum(float(x.get("answer_reward",0)) for x in scores)/n}


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    if group_size<=0 or raw_rewards.numel()%group_size: raise ValueError("invalid group_size")
    g=raw_rewards.reshape(-1,group_size)
    if baseline=="mean": a=g-g.mean(1,keepdim=True)
    elif baseline=="none": a=g.clone()
    else: raise ValueError("invalid baseline")
    if advantage_normalizer=="std": a=a/(g.std(1,keepdim=True,unbiased=True)+advantage_eps)
    elif advantage_normalizer=="mean": a=a/(g.mean(1,keepdim=True)+advantage_eps)
    elif advantage_normalizer!="none": raise ValueError("invalid normalizer")
    return a.reshape(-1),{"raw_reward_mean":float(raw_rewards.mean()),"advantage_mean":float(a.mean())}


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    a=raw_rewards_or_advantages.reshape(-1,1).to(policy_log_probs); meta={}
    if importance_reweighting_method=="none": loss=-a*policy_log_probs
    else:
        if old_log_probs is None: raise ValueError("old_log_probs required")
        lr=policy_log_probs-old_log_probs
        if importance_reweighting_method=="noclip": loss=-a*torch.exp(lr)
        elif importance_reweighting_method=="grpo":
            if cliprange is None: raise ValueError("cliprange required")
            r=torch.exp(lr); c=r.clamp(1-cliprange,1+cliprange); loss=-torch.minimum(r*a,c*a)
        elif importance_reweighting_method=="gspo":
            if cliprange is None or response_mask is None: raise ValueError("GSPO args required")
            m=response_mask.to(lr); r=torch.exp((lr*m).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)); c=r.clamp(1-cliprange,1+cliprange); loss=-torch.minimum(r*a,c*a).expand_as(policy_log_probs)
        else: raise ValueError("invalid importance method")
    return loss,meta


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    x=per_token_policy_gradient_loss*mask.to(per_token_policy_gradient_loss)
    if loss_normalization=="sequence": return (x.sum(1)/mask.sum(1).clamp_min(1)).mean()
    if loss_normalization=="constant":
        if normalization_constant is None: raise ValueError("normalization_constant required")
        return x.sum()/normalization_constant
    raise ValueError("invalid normalization")


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    t=run_tokenize_prompt_and_output(repeated_prompts,rollout_responses,tokenizer); r,rm=run_compute_rollout_rewards(reward_fn,rollout_responses,repeated_ground_truths); a,am=run_compute_group_normalized_rewards(r,group_size,baseline,advantage_eps,advantage_normalizer)
    n=len(repeated_prompts)
    if n%gradient_accumulation_steps: raise ValueError("batch not divisible")
    ms=n//gradient_accumulation_steps; optimizer.zero_grad(set_to_none=True); losses=[]
    for st in range(0,n,ms):
        sl=slice(st,st+ms); lp=run_get_response_log_probs(model,t["input_ids"][sl],t["labels"][sl],False)["log_probs"]; old=old_log_probs[sl] if old_log_probs is not None else None
        pg,_=run_compute_policy_gradient_loss(a[sl],lp,importance_reweighting_method,old,cliprange,t["response_mask"][sl]); loss=run_aggregate_loss_across_microbatch(pg,t["response_mask"][sl],loss_normalization,normalization_constant); losses.append(loss.detach()); (loss if loss_normalization=="constant" else loss/gradient_accumulation_steps).backward()
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm) if max_grad_norm is not None else None; optimizer.step(); optimizer.zero_grad(set_to_none=True)
    md={**rm,**am};
    if gn is not None: md["grad_norm"]=gn.detach()
    return (torch.stack(losses).sum() if loss_normalization=="constant" else torch.stack(losses).mean()),md


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    with open(dataset_path) as f: records=[json.loads(x) for x in f if x.strip()]
    if shuffle: random.shuffle(records)
    ids=[]
    for z in records:
        text="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+z["prompt"]+"\n\n### Response:\n"+z["response"]; ids.extend(tokenizer.encode(text,add_special_tokens=True)); ids.append(tokenizer.eos_token_id)
    items=[{"input_ids":torch.tensor(ids[i:i+seq_length],dtype=torch.long),"labels":torch.tensor(ids[i+1:i+seq_length+1],dtype=torch.long)} for i in range(0,len(ids)-seq_length,seq_length)]
    class Packed(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self,i): return items[i]
    return Packed()


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    ix=list(range(len(dataset))); random.shuffle(ix) if shuffle else None
    batches=[{k:torch.stack([dataset[i][k] for i in ix[s:s+batch_size]]) for k in dataset[ix[s]]} for s in range(0,len(ix),batch_size)]
    class Batches:
        def __len__(self): return len(batches)
        def __iter__(self): return iter(batches)
    return Batches()


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    found=re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?\b",model_output,re.I)
    if found: return found[-1].upper()
    for i,o in enumerate(mmlu_example.get("options",[])[:4]):
        if o and re.search(r"(?<!\w)" + re.escape(o.lower()) + r"(?!\w)", model_output.lower()): return "ABCD"[i]
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    x=re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",model_output); return x[-1].replace(",","") if x else None


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    def score(model,response):
        ids=torch.tensor([tokenizer.encode(prompt,add_special_tokens=True)+tokenizer.encode(response,add_special_tokens=False)])
        plen=len(tokenizer.encode(prompt,add_special_tokens=True)); lp=torch.log_softmax(model(input_ids=ids).logits[:,:-1],-1).gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
        return lp[:,plen-1:].sum(1) / max(1, len(tokenizer.encode(response,add_special_tokens=False)))
    pc,pr=score(lm,response_chosen),score(lm,response_rejected)
    with torch.no_grad(): rc,rr=score(lm_ref,response_chosen),score(lm_ref,response_rejected)
    return -torch.nn.functional.logsigmoid(2.083*beta*((pc-pr)-(rc-rr))).mean()
