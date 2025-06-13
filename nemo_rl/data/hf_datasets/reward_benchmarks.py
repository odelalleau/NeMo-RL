# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


# GenRM prompt template
GENRM_PROMPT_TEMPLATE = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query) and one or two responses from the Assistant, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
If there are two responses, you need to also give a ranking score based on the [Ranking Scoring Guidelines].
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

[Helpfulness Scoring Guidelines]

When evaluating Helpfulness, consider the following factors:

- Correctness/Completeness: Is the response accurate and complete?
- Coherence/Clarity: Is the response clear, coherent, and easy to understand?
- Instruction following: Does the response follow the instructions and fulfill the user's request?
- Relevance: Is the response relevant to the user's query/input?
- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? Does it show creativity but not hallucinations?

**Score 5: Extremely Helpful**

- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.
- It accurately acts on the user's request, without unnecessary information.
- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.

**Score 4: Mostly Helpful**

- The response is mostly helpful and mainly aligned with what the user was looking for.
- There is still some room for improvement, but the response is generally useful.

**Score 3: Partially Helpful**

- The response is partially helpful but misses the overall goal of the user's query/input in some way.
- The response did not fully satisfy what the user was looking for.

**Score 2: Borderline Unhelpful**

- The response is borderline unhelpful and mostly does not capture what the user was looking for.
- However, it is still usable and helpful in a small way.

**Score 1: Not Helpful**

- The response is not useful or helpful at all.
- The response completely missed the essence of what the user wanted.

[Ranking Scoring Guidelines]

Ranking score is used to rank the two responses based on their helpfulness. Even if you give the same individual helpfulness score for both responses, you need to differentiate them strictly.
The ranking score is a number between 1 and 6, where:
1 = Response 1 is much better than Response 2
2 = Response 1 is better than Response 2
3 = Response 1 is slightly better than Response 2
4 = Response 2 is slightly better than Response 1
5 = Response 2 is better than Response 1
6 = Response 2 is much better than Response 1

#### Conversation Context ####
{context}

#### Responses to be Scored ####
{responses}

#### Output Format Requirements ####
First give your analysis on each responses in the format of:
[The Begin of Analysis on Response i]
Analysis on the i-th response
[The End of Analysis on Response i]

Then give the scores of each response in order, separate by comma in the boxed, adhering this format:
[The Begin of Individual Scores]
\\boxed{{x, y}} if there exists 2 responses
[The End of Individual Scores]

If there are two responses, give the relative ranking score in the format of:
[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]
You don't need to give a ranking score if only one response is provided."""


def format_genrm_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the GenRM prompt with context and responses."""
    if response2 is None:
        responses = f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]"
    else:
        responses = (
            f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n"
            f"[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n"
        )
    
    return GENRM_PROMPT_TEMPLATE.format(
        context=context,
        responses=responses
    )





def format_rmbench_example(data: dict[str, Any]) -> dict[str, Any]:
    ################## TODO ##################
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    response1 = data.get("chosen", "")
    response2 = data.get("rejected", "")
    
    # Format as conversation context
    context = f"User: {prompt_text}"
    
    prompt = format_genrm_prompt(context, response1, response2)
    
    # RM-Bench typically has binary preference (chosen/rejected)
    # Map to GenRM's 1-6 scale (2 = chosen is better, 5 = rejected is better)
    label = data.get("label", 1)
    preference = 2 if label == 1 else 5
    
    return {
        "prompt": prompt,
        "num_responses": 2,
        "label_1": None,  # RM-Bench doesn't have individual scores
        "label_2": None,
        "preference": preference,
        "ground_truth": label,
    }


def format_rewardbench_example(data: dict[str, Any]) -> dict[str, Any]:
    ################## TODO ##################
    """Format RewardBench data for GenRM evaluation."""
    # RewardBench format varies by subset
    prompt_text = data.get("prompt", "")
    chosen = data.get("chosen", "")
    rejected = data.get("rejected", "")
    
    # Format as conversation context
    context = f"User: {prompt_text}"
    
    prompt = format_genrm_prompt(context, chosen, rejected)
    
    # RewardBench uses chosen/rejected format
    # Map to GenRM's 1-6 scale (2 = chosen is better)
    preference = 2  # Chosen (response 1) is better
    
    return {
        "prompt": prompt,
        "num_responses": 2,
        "label_1": None,  # RewardBench doesn't have individual scores
        "label_2": None,
        "preference": preference,
        "ground_truth": "chosen",
    }

def format_judgebench_example(data: dict[str, Any]) -> dict[str, Any]:
    """Format JudgeBench data for GenRM evaluation."""
    # Extract conversation context and responses based on actual JudgeBench format
    context = data.get("question", "")
    response1 = data.get("response_A", "")
    response2 = data.get("response_B", "")
    prompt = format_genrm_prompt(context, response1, response2)
    
    # Parse the label field (e.g., "B>A" means B is better than A)
    label = data.get("label", "")
    preference = None
    if label == "A>B":
        preference = 0  # Response 1 (A) is better
    else: # label == "B>A"
        preference = 1  # Response 2 (B) is better
    
    # JudgeBench doesn't have individual scores, just preference labels
    result = {
        "prompt": prompt,
        "num_responses": 2,
        "label_1": None,  # JudgeBench doesn't provide individual scores
        "label_2": None,
        "preference": preference,
        "ground_truth": label,  # Store original label as ground truth
    }
    
    return result


class JudgeBenchDataset:
    """JudgeBench dataset for GenRM evaluation."""
    def __init__(self):
        ds = load_dataset("ScalerLab/JudgeBench", split="gpt")
        self.formatted_ds = ds.map(format_judgebench_example, load_from_cache_file=False)
        self.task_spec = TaskDataSpec(task_name="JudgeBench")



class RMBenchDataset:
    """RM-Bench dataset for GenRM evaluation."""
    
    def __init__(self):
        ###################TODO ###################
        pass


class RewardBenchDataset:
    """RewardBench dataset for GenRM evaluation."""
    
    def __init__(self):
        ###################TODO ###################
        pass