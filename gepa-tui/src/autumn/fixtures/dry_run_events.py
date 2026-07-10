"""Scripted, deterministic GEPA event replay with zero GEPA/network dependency, for manually verifying dashboard rendering."""

import time


def replay(dashboard, speed: float = 1.0) -> None:
    dashboard.on_optimization_start(
        {"trainset_size": 20, "valset_size": 10, "config": {}}
    )

    for i in range(1, 6):
        dashboard.on_iteration_start({"iteration": i})
        dashboard.on_candidate_accepted(
            {
                "iteration": i,
                "new_candidate_idx": i,
                "new_score": 0.5 + i * 0.08,
                "parent_ids": [i - 1] if i > 1 else [None],
            }
        )
        dashboard.on_pareto_front_updated(
            {
                "iteration": i,
                "new_front": [i],
                "displaced_candidates": [i - 1] if i > 1 else [],
            }
        )
        dashboard.on_valset_evaluated(
            {
                "iteration": i,
                "candidate_idx": i,
                "average_score": 0.5 + i * 0.08,
                "is_best_program": True,
                "candidate": {
                    "system_prompt": f"You are a helpful assistant (revision {i}).",
                },
            }
        )
        dashboard.on_budget_updated(
            {
                "iteration": i,
                "metric_calls_used": i * 10,
                "metric_calls_remaining": 100 - i * 10,
            }
        )
        dashboard.on_iteration_end({"iteration": i, "proposal_accepted": True})
        time.sleep(0.5 / speed)

    dashboard.on_merge_attempted(
        {"iteration": 5, "parent_ids": [3, 4], "merged_candidate": {}}
    )
    dashboard.on_merge_accepted(
        {"iteration": 5, "new_candidate_idx": 6, "parent_ids": [3, 4]}
    )
    dashboard.on_error(
        {
            "iteration": 5,
            "exception": RuntimeError("synthetic test error"),
            "will_continue": True,
        }
    )
    dashboard.on_optimization_end(
        {"best_candidate_idx": 5, "total_iterations": 5, "total_metric_calls": 50}
    )
