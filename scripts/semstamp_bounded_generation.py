"""Termination-safe SemStamp sentence rejection sampling.

The upstream implementation checks ``candidate not in accept_mask`` before its
``current_trials >= MAX_TRIALS`` fallback. A persistently rejected signature can
therefore bypass the fallback forever. This module preserves the upstream
algorithm except that both rejection filters apply only before the frozen trial
cap. At the cap, the current regular continuation is accepted deterministically.
"""


PATCH_ID = "semstamp-max-trials-control-flow-v1"


def validate_trial_cap(value):
    if type(value) is not int or value < 1:
        raise ValueError("MAX_TRIALS must be a positive integer")
    return value


def bounded_lsh_reject_completion(
    prompt,
    model,
    tokenizer,
    gen_config,
    lsh_model,
    lsh_dim,
    lmbd=1.0,
    device="cuda",
    margin=0.002,
    *,
    official_module,
):
    """Return ``(text, metrics)`` with at most MAX_TRIALS per accepted segment."""
    max_trials = validate_trial_cap(official_module.MAX_TRIALS)
    print(f"prompt: {prompt}")
    sent_end_criteria = official_module.SentenceEndCriteria(tokenizer)
    lsh_seed = lsh_model.get_hash([prompt])[0]
    accept_mask = official_module.get_mask_from_seed(
        lsh_dim, lmbd, lsh_seed
    )

    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(
        prompt, return_tensors="pt"
    ).to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    accepted_segments = 0
    max_trials_reached_segments = 0
    fallback_acceptances = 0
    max_current_trials_observed = 0
    current_trials = 0
    while True:
        stopping_criteria = official_module.StoppingCriteriaList(
            [sent_end_criteria]
        )
        if "opt" not in model.config._name_or_path:
            raise NotImplementedError("model type not supported")
        new_text, new_text_ids = official_module.gen_sent(
            model=model,
            tokenizer=tokenizer,
            text_ids=text_ids,
            gen_config=gen_config,
            stopping_criteria=stopping_criteria,
        )
        if new_text == "":
            print(
                "WARNING: stopped generation because generated nothing "
                "(after discarding last generated token)",
                flush=True,
            )
            break

        total_trials += 1
        current_trials += 1
        max_current_trials_observed = max(
            max_current_trials_observed, current_trials
        )
        accepted_text, _ = official_module.reject_close_generation(
            lsh_model, [new_text], margin=margin, cutoff=None
        )
        close_ok = len(accepted_text) != 0
        if not close_ok and current_trials < max_trials:
            continue

        lsh_candidate = lsh_model.get_hash([new_text])[0]
        signature_ok = lsh_candidate in accept_mask
        if not signature_ok and current_trials < max_trials:
            continue

        reached_cap = current_trials >= max_trials
        if reached_cap:
            max_trials_reached_segments += 1
            if not (close_ok and signature_ok):
                fallback_acceptances += 1
            print(
                "WARNING: bounded sampler reached max_trials "
                f"{max_trials}; use current regular continuation",
                flush=True,
            )

        accepted_segments += 1
        current_trials = 0
        lsh_seed = lsh_candidate
        accept_mask = official_module.get_mask_from_seed(
            lsh_dim, lmbd, lsh_seed
        )
        text += new_text
        text_ids = new_text_ids
        sent_end_criteria.update(text)
        if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens - 1:
            break

    if current_trials >= max_trials:
        raise AssertionError("unfinished segment exceeded frozen trial cap")
    if total_trials > (accepted_segments + 1) * max_trials:
        raise AssertionError("total trial bound violated")
    return text, {
        "patch_id": PATCH_ID,
        "max_trials": max_trials,
        "total_trials": total_trials,
        "accepted_segments": accepted_segments,
        "max_trials_reached_segments": max_trials_reached_segments,
        "fallback_acceptances": fallback_acceptances,
        "max_current_trials_observed": max_current_trials_observed,
        "terminated_with_empty_generation": new_text == "",
    }
