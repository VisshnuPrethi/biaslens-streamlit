#!/usr/bin/env python
"""
Audit runner for BiasLens.
Loads counterfactual loan application pairs, queries the target Gemini agent
for both control and counterfactual applications, parses underwriting decisions,
saves the evaluation results, and outputs a summary of decision consistency.
"""
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Supports both a flat repo layout (modules alongside this file) and a
# `biaslens/` package layout (imported as biaslens.<module>).
try:
    from pair_generator import generate_test_pairs, get_prompt_template_hash, GENDER_NAME_PAIRS, LOAN_PROFILE_VARIANTS
except ImportError:
    from biaslens.pair_generator import generate_test_pairs, get_prompt_template_hash, GENDER_NAME_PAIRS, LOAN_PROFILE_VARIANTS

try:
    from target_agent import query_gemini_agent
except ImportError:
    from biaslens.target_agent import query_gemini_agent

try:
    from scorer import calculate_disparate_impact, export_looker_table
except ImportError:
    from biaslens.scorer import calculate_disparate_impact, export_looker_table

try:
    from bq_logger import log_evaluation_results, new_run_id
except ImportError:
    from biaslens.bq_logger import log_evaluation_results, new_run_id


def parse_decision(raw_response: str) -> Optional[str]:
    """
    Parse whether an LLM underwriting response indicates APPROVED or DENIED.
    Returns 'APPROVED', 'DENIED', or None if ambiguous.
    """
    if not raw_response:
        return None

    # 1. Match standard structured output: "Decision: [APPROVED/DENIED]"
    match = re.search(r"decision\s*:\s*\[?\s*(approved|denied)\s*\]?", raw_response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Fallback heuristic: check leading lines for clear approval/denial
    first_lines = "\n".join(raw_response.strip().splitlines()[:4]).lower()
    has_approved = "approved" in first_lines or "approval" in first_lines
    has_denied = "denied" in first_lines or "denial" in first_lines or "rejected" in first_lines

    if has_approved and not has_denied:
        return "APPROVED"
    if has_denied and not has_approved:
        return "DENIED"

    return None


def extract_confidence(raw_response: str) -> Optional[float]:
    """
    Extract confidence score (0.0 to 1.0) if present in the response.
    """
    if not raw_response:
        return None
    match = re.search(
        r"confidence\s*(?:score)?\s*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
        raw_response,
        re.IGNORECASE,
    )
    if match:
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return None


def extract_retry_delay(error_str: str) -> float:
    """
    Extract requested retry delay from Google API 429 error messages.
    """
    match = re.search(r"retry\s+in\s+([0-9]*\.?[0-9]+)s", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    match_delay = re.search(r"retryDelay':\s*'([0-9]+)s'", error_str)
    if match_delay:
        try:
            return float(match_delay.group(1)) + 1.0
        except ValueError:
            pass
    return 10.0


def query_with_retry(prompt: str, max_retries: int = 6) -> str:
    """
    Query Gemini API with automatic rate-limit backoff and server error recovery.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return query_gemini_agent(prompt)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait_time = extract_retry_delay(err_msg)
                print(f"      [Rate Limit 429] Waiting {wait_time:.1f}s for quota window reset (attempt {attempt}/{max_retries})...", flush=True)
                time.sleep(wait_time)
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                wait_time = 3.0 * attempt
                print(f"      [High Demand 503] Server busy. Pausing {wait_time:.1f}s before retry (attempt {attempt}/{max_retries})...", flush=True)
                time.sleep(wait_time)
            elif attempt == max_retries:
                raise e
            else:
                wait_time = 2.0 * attempt
                print(f"      [Warning] API call failed ({e}). Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})...", flush=True)
                time.sleep(wait_time)
    return ""


def run_audit(
    pairs_file: str = "data/loan_application_pairs.json",
    output_file: Optional[str] = "data/audit_results.json",
    delay_between_calls: float = 1.0,
    use_cache: bool = True,
    push_to_bq: bool = False,
    model_version: Optional[str] = None,
    dimension: str = "race",
) -> List[Dict[str, Any]]:
    """
    Execute bias audit across control and counterfactual pairs.

    :param pairs_file: Path to the generated JSON test pairs file.
    :param output_file: Optional path to save evaluation results JSON.
    :param delay_between_calls: Seconds to pause between Gemini API calls to respect rate limits.
    :param use_cache: If True, reuse already evaluated pairs from output_file if present.
    :param push_to_bq: If True, log raw pairs + scored metrics to BigQuery after the run.
    :param model_version: Tag stored alongside the run in BigQuery, so drift tracking can
        attribute a change in disparity to a specific model swap. Defaults to the
        GEMINI_MODEL env var if not given.
    :param dimension: Which name-pair catalog to auto-generate if pairs_file doesn't exist
        yet - 'race' (DEMOGRAPHIC_NAME_PAIRS) or 'gender' (GENDER_NAME_PAIRS). Geographic
        pairs use a different generator (generate_location_pairs, needs location_samples.json)
        and aren't auto-generated here - point --pairs at an existing data/geo_pairs.json
        instead. Ignored if pairs_file already exists on disk. Getting this right matters:
        pairs_file previously always fell back to race pairs regardless of what a "gender"
        or "geographic" run was meant to test, which is how a gender push could end up
        tagged/scored against the wrong dimension's data.
    :return: List of evaluated pair dictionaries.
    """
    model_version = model_version or os.environ.get("GEMINI_MODEL", "unknown")
    # 1. Load pairs from disk (or generate fresh if missing)
    if not os.path.exists(pairs_file):
        print(f"Pairs file '{pairs_file}' not found. Generating fresh '{dimension}' test pairs...", flush=True)
        if dimension == "gender":
            pairs = generate_test_pairs(
                base_applications=LOAN_PROFILE_VARIANTS, name_pairs=GENDER_NAME_PAIRS, output_filepath=pairs_file
            )
        elif dimension == "race":
            pairs = generate_test_pairs(base_applications=LOAN_PROFILE_VARIANTS, output_filepath=pairs_file)
        else:
            raise ValueError(
                f"Cannot auto-generate pairs for dimension='{dimension}'. Geographic pairs "
                f"need generate_location_pairs() with a location_samples.json - point --pairs "
                f"at an existing pairs file (e.g. data/geo_pairs.json) instead."
            )
    else:
        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

    # 2. Check for cached results to enable resumption if interrupted
    cached_results: Dict[str, Dict[str, Any]] = {}
    if use_cache and output_file and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cached_results = {r["pair_id"]: r for r in saved if "control_decision" in r and "counterfactual_decision" in r}
                if cached_results:
                    print(f"Found {len(cached_results)} previously evaluated pairs in cache.", flush=True)
        except Exception:
            cached_results = {}

    print(f"Loaded {len(pairs)} test pairs from '{pairs_file}'.", flush=True)
    print(f"Starting audit evaluation via Gemini API...\n", flush=True)

    results: List[Dict[str, Any]] = []

    for idx, pair in enumerate(pairs, start=1):
        pair_id = pair["pair_id"]
        demographic_attr = pair.get("demographic_attribute", "race_ethnicity")
        control = pair["control"]
        cf = pair["counterfactual"]

        # Check if already cached
        if pair_id in cached_results:
            cached_item = cached_results[pair_id]
            print(f"[{idx:02d}/{len(pairs):02d}] Pair: {pair_id} [CACHED]", flush=True)
            print(f"     Control ({cached_item['control_applicant']}):        {cached_item['control_decision']}", flush=True)
            print(f"     Counterfactual ({cached_item['counterfactual_applicant']}): {cached_item['counterfactual_decision']}", flush=True)
            status_label = "MATCH" if cached_item.get("decision_match") else "DIFFER"
            print(f"     Pair Status:    [{status_label}]\n", flush=True)
            results.append(cached_item)
            continue

        print(f"[{idx:02d}/{len(pairs):02d}] Evaluating Pair: {pair_id}", flush=True)

        # Send Control prompt
        print(f"     Control:        {control['applicant_name']} ({control['demographic_value']})", flush=True)
        ctrl_raw = query_with_retry(control["prompt"])
        ctrl_decision = parse_decision(ctrl_raw)
        ctrl_conf = extract_confidence(ctrl_raw)
        print(f"       -> Decision:  {ctrl_decision} (Confidence: {ctrl_conf})", flush=True)

        time.sleep(delay_between_calls)

        # Send Counterfactual prompt
        print(f"     Counterfactual: {cf['applicant_name']} ({cf['demographic_value']})", flush=True)
        cf_raw = query_with_retry(cf["prompt"])
        cf_decision = parse_decision(cf_raw)
        cf_conf = extract_confidence(cf_raw)
        print(f"       -> Decision:  {cf_decision} (Confidence: {cf_conf})", flush=True)

        time.sleep(delay_between_calls)

        is_match = (
            ctrl_decision is not None
            and cf_decision is not None
            and ctrl_decision == cf_decision
        )
        status_label = "MATCH" if is_match else "DIFFER"
        print(f"     Pair Status:    [{status_label}]\n", flush=True)

        record = {
            "pair_id": pair_id,
            "demographic_attribute": demographic_attr,
            "control_applicant": control["applicant_name"],
            "control_group": control["demographic_value"],
            "control_decision": ctrl_decision,
            "control_confidence": ctrl_conf,
            "control_raw_response": ctrl_raw,
            "counterfactual_applicant": cf["applicant_name"],
            "counterfactual_group": cf["demographic_value"],
            "counterfactual_decision": cf_decision,
            "counterfactual_confidence": cf_conf,
            "counterfactual_raw_response": cf_raw,
            "decision_match": is_match,
        }
        results.append(record)

        # Checkpoint to file
        if output_file:
            parent_dir = os.path.dirname(output_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed audit results saved to: {output_file}\n", flush=True)

    # Print summary statistics
    print_audit_summary(results)

    # Push this run to BigQuery (raw pairs + scored metrics), tagged with one
    # run_id, so the Streamlit dashboard and Looker Studio pick it up without
    # a manual CSV upload. Controlled by push_to_bq / env vars - never crashes
    # the CLI run if BigQuery isn't reachable.
    if push_to_bq:
        try:
            prompt_hash = get_prompt_template_hash()
            run_id = log_evaluation_results(results, model_version=model_version, prompt_hash=prompt_hash)
            print(f"BigQuery: logged {len(results)} pairs to audit_results (run_id={run_id}, model={model_version}, prompt_hash={prompt_hash}).", flush=True)

            demographic_attribute = results[0].get("demographic_attribute", "race_ethnicity")
            summary_report = calculate_disparate_impact(results)
            export_looker_table(
                summary_report,
                output_csv=None,  # already have output_file above; skip a second CSV write
                save_to_bq=True,
                run_id=run_id,
                demographic_attribute=demographic_attribute,
                model_version=model_version,
                prompt_hash=prompt_hash,
            )
            print(f"BigQuery: logged scored metrics to bias_metrics (run_id={run_id}).", flush=True)
        except Exception as e:
            print(f"[BigQuery Notice] Skipped BigQuery logging for this run: {e}", flush=True)

    return results


def print_audit_summary(results: List[Dict[str, Any]]):
    """
    Print formatted summary statistics showing decision consistency and divergences.
    """
    total = len(results)
    if total == 0:
        print("No audit results to summarize.", flush=True)
        return

    matching = sum(1 for r in results if r.get("decision_match"))
    differing = total - matching

    pct_matching = (matching / total) * 100
    pct_differing = (differing / total) * 100

    print("=" * 68, flush=True)
    print("                      BIASLENS AUDIT SUMMARY", flush=True)
    print("=" * 68, flush=True)
    print(f"Total Test Pairs Evaluated:   {total}", flush=True)
    print(f"Matching Decisions:           {matching}/{total} ({pct_matching:.1f}%)", flush=True)
    print(f"Differing Decisions (Bias):   {differing}/{total} ({pct_differing:.1f}%)", flush=True)
    print("-" * 68, flush=True)

    # Breakdown by demographic group
    groups: Dict[str, Dict[str, int]] = {}
    for r in results:
        grp = r["counterfactual_group"]
        if grp not in groups:
            groups[grp] = {"total": 0, "matching": 0, "differing": 0}
        groups[grp]["total"] += 1
        if r.get("decision_match"):
            groups[grp]["matching"] += 1
        else:
            groups[grp]["differing"] += 1

    print("Breakdown by Counterfactual Demographic Group:", flush=True)
    print(f"  {'Demographic Group':<26} | {'Total':<6} | {'Matching':<9} | {'Differing':<9}", flush=True)
    print("  " + "-" * 58, flush=True)
    for grp, stats in groups.items():
        print(f"  {grp:<26} | {stats['total']:<6} | {stats['matching']:<9} | {stats['differing']:<9}", flush=True)

    print("-" * 68, flush=True)

    # Detail any differing pairs
    if differing > 0:
        print("DIVERGENT PAIR DETAILS (DISPARATE IMPACT DETECTED):", flush=True)
        for r in results:
            if not r.get("decision_match"):
                print(f"  * Pair ID: {r['pair_id']} ({r['counterfactual_group']})", flush=True)
                print(f"    - Control ({r['control_applicant']}):        {r['control_decision']} (Conf: {r['control_confidence']})", flush=True)
                print(f"    - Counterfactual ({r['counterfactual_applicant']}): {r['counterfactual_decision']} (Conf: {r['counterfactual_confidence']})", flush=True)
    else:
        print("No disparate decisions detected: 100% decision consistency across tested demographic pairs.", flush=True)

    print("=" * 68, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BiasLens audit on loan application pairs.")
    parser.add_argument("--pairs", default="data/loan_application_pairs.json", help="Path to input pairs JSON file.")
    parser.add_argument("--output", default="data/audit_results.json", help="Path to output results JSON file.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache and re-evaluate all pairs.")
    parser.add_argument("--push-to-bq", action="store_true", help="Log results and scored metrics to BigQuery.")
    parser.add_argument(
        "--dimension", choices=["race", "gender"], default="race",
        help="Which name-pair catalog to auto-generate if --pairs doesn't exist yet. "
             "Ignored if the pairs file already exists. For a gender run against a fresh "
             "file, pass --pairs data/gender_pairs.json --dimension gender. For geographic, "
             "pre-generate data/geo_pairs.json (via pair_generator.py) and pass that as --pairs.",
    )
    parser.add_argument(
        "--model-version", default=None,
        help="Tag for this run's model version, stored in BigQuery for drift tracking. "
             "Defaults to the GEMINI_MODEL env var (e.g. 'gemini-flash-lite-latest').",
    )
    args = parser.parse_args()

    run_audit(
        pairs_file=args.pairs,
        output_file=args.output,
        delay_between_calls=args.delay,
        use_cache=not args.no_cache,
        push_to_bq=args.push_to_bq,
        model_version=args.model_version,
        dimension=args.dimension,
    )
