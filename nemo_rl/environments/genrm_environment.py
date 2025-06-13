# nemo_rl/environments/genrm_environment.py
import re
import logging
from typing import Any, Optional, TypedDict
import numpy as np
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

class GenRMEnvironmentMetadata(TypedDict):
    num_responses: int
    helpfulness_1: Optional[int]
    helpfulness_2: Optional[int]
    preference_ranking: Optional[int]


@ray.remote
class GenRMEnvironment(EnvironmentInterface):
    """Generative Reward Model environment for HelpSteer3 dataset."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -100)  # Large penalty for format violations
        logging.basicConfig(level=logging.INFO)
    
    def extract_answer(self, string: str) -> Optional[str]:
        """Extract Answer String from \\boxed expression."""
        # Try to find the last occurrence of \boxed
        idx = string.rfind("\\boxed")
        if idx < 0:
            # Try \fbox as alternative
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        # Find the matching braces
        i = idx + 6  # Skip past "\boxed"
        if i >= len(string) or string[i] != '{':
            return None
            
        # Count braces to find the matching closing brace
        brace_count = 0
        start_idx = i
        while i < len(string):
            if string[i] == '{':
                brace_count += 1
            elif string[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Extract content between braces
                    return string[start_idx + 1:i].strip()
            i += 1

        return None

    def parse_score_value(self, score_str: str) -> Optional[int]:
        """Parse a score value from a string, handling various formats."""
        if not score_str:
            return None
            
        score_str = score_str.strip()
        
        # Try direct conversion first
        try:
            return int(score_str)
        except ValueError:
            pass
        
        # Try to extract the first number found
        match = re.search(r'(\d+)', score_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
                
        return None

    def distance_abs(self, predicted: Any, ground_truth: Any) -> int:
        """Calculate absolute distance between predicted and ground truth."""
        try:
            # Handle None values
            if predicted is None or ground_truth is None:
                return 100
            
            # Convert to integers if they're strings
            if isinstance(predicted, str):
                predicted = self.parse_score_value(predicted)
                if predicted is None:
                    return 100
                    
            if isinstance(ground_truth, str):
                ground_truth = self.parse_score_value(ground_truth)
                if ground_truth is None:
                    return 100
            
            # Calculate distance
            return abs(int(predicted) - int(ground_truth))
        except Exception as e:
            logging.error(f"Error calculating distance: {e}, predicted: {predicted}, ground_truth: {ground_truth}")
            return 100

    def validate_format_and_extract(self, assistant_response: str, num_responses: int) -> tuple[bool, Optional[list], Optional[str], str]:
        """
        Validate the strict format and extract scores.
        Returns: (is_valid_format, individual_scores, preference_ranking, error_message)
        """
        if num_responses == 1:
            # Expected format: "[The Begin of Individual Scores]\n\\boxed{x} \n[The End of Individual Scores]\n"
            pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
            match = re.search(pattern, assistant_response, re.DOTALL)
            
            if not match:
                return False, None, None, "Format violation: Expected '[The Begin of Individual Scores]\\n\\\\boxed{x} \\n[The End of Individual Scores]\\n' for single response"
            
            score_content = match.group(1).strip()
            # For single response, should be just one score
            if ',' in score_content:
                return False, None, None, "Format violation: Single response should contain only one score, found comma"
            
            return True, [score_content], None, ""
            
        elif num_responses == 2:
            # Expected format: "[The Begin of Individual Scores]\n\\boxed{x, y} \n[The End of Individual Scores]\n[The Begin of Ranking Score]\n\\boxed{z} \n[The End of Ranking Score]"
            
            # Check individual scores section
            individual_pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
            individual_match = re.search(individual_pattern, assistant_response, re.DOTALL)
            
            if not individual_match:
                return False, None, None, "Format violation: Missing or incorrect '[The Begin of Individual Scores]\\n\\\\boxed{x, y} \\n[The End of Individual Scores]' section"
            
            # Check ranking score section
            ranking_pattern = r'\[The Begin of Ranking Score\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Ranking Score\]'
            ranking_match = re.search(ranking_pattern, assistant_response, re.DOTALL)
            
            if not ranking_match:
                return False, None, None, "Format violation: Missing or incorrect '[The Begin of Ranking Score]\\n\\\\boxed{z} \\n[The End of Ranking Score]' section"
            
            # Extract individual scores
            scores_content = individual_match.group(1).strip()
            individual_scores = [score.strip() for score in scores_content.split(',')]
            
            # For two responses, should have exactly two scores
            if len(individual_scores) != 2:
                return False, None, None, f"Format violation: Expected exactly 2 individual scores, found {len(individual_scores)}"
            
            # Extract preference ranking
            preference_content = ranking_match.group(1).strip()
            # Preference ranking should be a single value
            if ',' in preference_content:
                return False, None, None, "Format violation: Preference ranking should be a single value, found comma"
            
            return True, individual_scores, preference_content, ""
        
        else:
            return False, None, None, f"Unsupported number of responses: {num_responses}"
    
    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[GenRMEnvironmentMetadata],) -> EnvironmentReturn:
        """Evaluate GenRM predictions and return rewards."""
        
        rewards = []
        observations = []
        
        for conversation, meta in zip(message_log_batch, metadata):
            # Extract assistant's response
            assistant_response = ""
            for msg in conversation:
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break
            
            
            
            is_valid_format, individual_scores, preference_ranking, error_message = self.validate_format_and_extract(
                assistant_response, meta["num_responses"]
            )
            # Validate format and extract scores
            print("assistant_response: ", assistant_response)
            print("extracted results: ")
            if individual_scores is not None:
                print(individual_scores)
            if preference_ranking is not None:
                print(preference_ranking)
            print()
            
            
            if not is_valid_format:
                # Apply format penalty
                reward = self.format_penalty
                rewards.append(float(reward))
                observations.append({
                    "role": "environment",
                    "content": f"Format violation penalty applied. Error: {error_message}. Reward: {reward}"
                })
                
            else:
                reward = self.format_penalty
                try:
                    if meta["num_responses"] == 1:
                        if meta["helpfulness_1"] is not None and individual_scores:
                            reward = -self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                    
                    elif meta["num_responses"] == 2:
                        # Calculate distance for both individual scores
                        if meta["helpfulness_1"] is not None and individual_scores and len(individual_scores) == 2 and meta["preference_ranking"] is not None and preference_ranking:
                            reward = - self.distance_abs(individual_scores[0], meta["helpfulness_1"])  - self.distance_abs(individual_scores[1], meta["helpfulness_2"]) - self.distance_abs(preference_ranking, meta["preference_ranking"])
                
                except Exception as e:
                    logging.error(f"Error processing response: {e}")

                rewards.append(float(reward))
                observations.append({
                    "role": "environment",
                    "content": f"Format correct. GenRM evaluation complete. Reward: {reward}"
                })
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminateds = torch.ones_like(rewards_tensor, dtype=torch.bool)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminateds,
        )
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Post processing and metrics calculation."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
        
        if len(rewards) == 0:
            return batch, {}
        
        # Convert rewards to numpy for easier computation
        rewards_np = rewards.numpy() if hasattr(rewards, 'numpy') else np.array(rewards)
        
        # Calculate metrics
        mean_reward = float(np.mean(rewards_np))
        
        # Calculate format violation rate (rewards equal to format_penalty)
        format_violation_rate = float(np.mean(rewards_np == self.format_penalty))
        
        # For non-format-penalty rewards, calculate accuracy metrics
        accuracy_rewards = rewards_np[rewards_np != self.format_penalty]
        if len(accuracy_rewards) > 0:
            mean_accuracy_reward = float(np.mean(accuracy_rewards))
            perfect_pred = float(np.mean(accuracy_rewards == 0))  # Distance 0 = perfect
            good_pred = float(np.mean(accuracy_rewards >= -10))  # Distance <= 10 = good
        else:
            mean_accuracy_reward = 0.0
            perfect_pred = 0.0
            good_pred = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "mean_distance": -mean_reward if mean_reward != self.format_penalty else 0,
            "format_violation_rate": format_violation_rate,
            "mean_accuracy_reward": mean_accuracy_reward,
            "perfect_pred_rate": perfect_pred,
            "good_pred_rate": good_pred,
            "num_samples": num_samples,
            "format_correct_samples": len(accuracy_rewards),
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass