"""
Main experiment runner for AISAI.
Runs 100 trials per prompt per model and logs results to Google Sheets.
"""

import time
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

from llm_client import LLMClient, get_all_models
from sheets_logger import SheetsLogger
from prompts import PROMPTS, PROMPT_DESCRIPTIONS


class AISAIExperiment:
    """Main experiment runner for AI Self-Awareness Index."""

    def __init__(self, log_to_sheets: bool = True, trials_per_prompt: int = 100):
        """
        Initialize experiment runner.

        Args:
            log_to_sheets: Whether to log results to Google Sheets
            trials_per_prompt: Number of trials to run per prompt per model
        """
        self.client = LLMClient()
        self.logger = SheetsLogger() if log_to_sheets else None
        self.trials_per_prompt = trials_per_prompt

        print("\n" + "="*60)
        print("AISAI: AI Self-Awareness Index Experiment")
        print("="*60)
        print(f"Trials per prompt: {trials_per_prompt}")
        print(f"Logging to sheets: {log_to_sheets}")
        print("="*60 + "\n")

    def run_single_trial(
        self,
        provider: str,
        model: str,
        prompt_type: str,
        prompt: str,
        trial_number: int,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run a single trial.

        Args:
            provider: LLM provider (openai, anthropic, gemini)
            model: Model name
            prompt_type: Type of prompt (A_baseline, B_against_ai, C_against_self)
            prompt: The actual prompt text
            trial_number: Trial number
            temperature: Sampling temperature

        Returns:
            Trial result dictionary
        """
        result = self.client.call_model(provider, model, prompt, temperature)

        # Add metadata
        result["trial_number"] = trial_number
        result["prompt_type"] = prompt_type

        # Log to sheets if enabled
        if self.logger:
            self.logger.log_trial(result)

        return result

    def run_model_experiment(
        self,
        provider: str,
        model: str,
        prompts_to_run: List[str] = None,
        delay_between_calls: float = 1.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run full experiment for a single model across all prompts.

        Args:
            provider: LLM provider
            model: Model name
            prompts_to_run: List of prompt types to run (default: all)
            delay_between_calls: Delay in seconds between API calls

        Returns:
            Dictionary mapping prompt types to lists of results
        """
        if prompts_to_run is None:
            prompts_to_run = list(PROMPTS.keys())

        print(f"\n{'='*60}")
        print(f"Running experiment for: {provider}/{model}")
        print(f"{'='*60}\n")

        all_results = {}

        for prompt_type in prompts_to_run:
            prompt = PROMPTS[prompt_type]
            description = PROMPT_DESCRIPTIONS[prompt_type]

            print(f"\n{'-'*60}")
            print(f"Prompt Type: {prompt_type}")
            print(f"Description: {description}")
            print(f"{'-'*60}")

            results = []

            # Run trials with progress bar
            for trial in tqdm(range(1, self.trials_per_prompt + 1), desc=f"{prompt_type}"):
                try:
                    result = self.run_single_trial(
                        provider=provider,
                        model=model,
                        prompt_type=prompt_type,
                        prompt=prompt,
                        trial_number=trial
                    )
                    results.append(result)

                    # Delay between calls to avoid rate limits
                    if trial < self.trials_per_prompt:
                        time.sleep(delay_between_calls)

                except Exception as e:
                    print(f"\n✗ Error on trial {trial}: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "trial_number": trial,
                        "prompt_type": prompt_type,
                        "provider": provider,
                        "model": model
                    })

            all_results[prompt_type] = results

            # Print summary for this prompt
            self._print_prompt_summary(results, prompt_type)

        return all_results

    def run_all_models(
        self,
        providers: List[str] = None,
        prompts_to_run: List[str] = None,
        delay_between_calls: float = 1.0
    ):
        """
        Run experiment for all available models.

        Args:
            providers: List of providers to test (default: all available)
            prompts_to_run: List of prompt types to run (default: all)
            delay_between_calls: Delay in seconds between API calls
        """
        all_models = get_all_models()

        # Filter by provider if specified
        if providers:
            all_models = [m for m in all_models if m["provider"] in providers]

        print(f"\nTotal models to test: {len(all_models)}")
        print(f"Total trials: {len(all_models)} models × {self.trials_per_prompt} trials × {len(prompts_to_run or PROMPTS)} prompts")
        print(f"= {len(all_models) * self.trials_per_prompt * len(prompts_to_run or PROMPTS)} API calls\n")

        for model_info in all_models:
            provider = model_info["provider"]
            model = model_info["model"]

            try:
                results = self.run_model_experiment(
                    provider=provider,
                    model=model,
                    prompts_to_run=prompts_to_run,
                    delay_between_calls=delay_between_calls
                )

                # Calculate and log summary statistics
                self._calculate_and_log_summary(provider, model, results)

            except Exception as e:
                print(f"\n✗ Failed to run experiment for {provider}/{model}: {e}")

    def _print_prompt_summary(self, results: List[Dict[str, Any]], prompt_type: str):
        """Print summary statistics for a prompt's results."""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        guesses = []
        for r in successful:
            parsed = r.get("parsed_response", {})
            if parsed and isinstance(parsed, dict):
                guess = parsed.get("guess")
                if guess is not None:
                    try:
                        guesses.append(float(guess))
                    except (ValueError, TypeError):
                        pass

        print(f"\n📊 Summary for {prompt_type}:")
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Failed: {len(failed)}/{len(results)}")

        if guesses:
            import statistics
            print(f"  Mean guess: {statistics.mean(guesses):.2f}")
            print(f"  Median guess: {statistics.median(guesses):.2f}")
            print(f"  Std dev: {statistics.stdev(guesses):.2f}" if len(guesses) > 1 else "  Std dev: N/A")
            print(f"  Range: [{min(guesses):.1f}, {max(guesses):.1f}]")
        else:
            print("  No valid guesses extracted")

    def _calculate_and_log_summary(self, provider: str, model: str, all_results: Dict[str, List[Dict[str, Any]]]):
        """Calculate summary statistics and log to sheets."""
        import statistics

        for prompt_type, results in all_results.items():
            successful = [r for r in results if r.get("success", False)]
            failed = [r for r in results if not r.get("success", False)]

            guesses = []
            for r in successful:
                parsed = r.get("parsed_response", {})
                if parsed and isinstance(parsed, dict):
                    guess = parsed.get("guess")
                    if guess is not None:
                        try:
                            guesses.append(float(guess))
                        except (ValueError, TypeError):
                            pass

            summary = {
                "prompt_type": prompt_type,
                "provider": provider,
                "model": model,
                "total_trials": len(results),
                "successful_trials": len(successful),
                "failed_trials": len(failed),
                "mean_guess": round(statistics.mean(guesses), 2) if guesses else None,
                "median_guess": round(statistics.median(guesses), 2) if guesses else None,
                "std_guess": round(statistics.stdev(guesses), 2) if len(guesses) > 1 else None,
                "min_guess": round(min(guesses), 2) if guesses else None,
                "max_guess": round(max(guesses), 2) if guesses else None
            }

            if self.logger:
                self.logger.update_summary(summary)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AISAI experiment")

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google", "all"],
        default="all",
        help="Which provider to test"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to test (e.g., gpt-4, claude-3-opus-20240229)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials per prompt (default: 100)"
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=["A_baseline", "B_against_ai", "C_against_self", "all"],
        default=["all"],
        help="Which prompts to run"
    )

    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="Don't log to Google Sheets"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Setup prompts
    prompts_to_run = None if "all" in args.prompts else args.prompts

    # Setup providers
    providers = None if args.provider == "all" else [args.provider]

    # Initialize experiment
    experiment = AISAIExperiment(
        log_to_sheets=not args.no_sheets,
        trials_per_prompt=args.trials
    )

    # Run experiment
    if args.model:
        # Run specific model
        provider = None
        for p in ["openai", "anthropic", "google"]:
            if args.provider == p or args.provider == "all":
                provider = p
                break

        if not provider:
            print("Error: Must specify provider when using --model")
            return

        experiment.run_model_experiment(
            provider=provider,
            model=args.model,
            prompts_to_run=prompts_to_run,
            delay_between_calls=args.delay
        )
    else:
        # Run all models
        experiment.run_all_models(
            providers=providers,
            prompts_to_run=prompts_to_run,
            delay_between_calls=args.delay
        )

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
