import os
import random
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import nevergrad as ng
from modifiable_features_fingerprinting import MODIFIABLE_FEATURES
from preprocessing.preprocessor import preprocessing_pipeline_partial

def blackbox_attack(
    detection,
    df: pd.DataFrame,
    modifiable_features: Optional[Iterable[str]] = None,
    *,
    budget: int = 50,
    noise_level: float = 0.1,
    tmp_dir: Optional[str] = None,
    return_perturbed: bool = False,
    protocol_column: str = "ip.proto",
    protocol_constraints: Optional[Dict[str, List[str]]] = None,
    optimizer_type: str = "ngopt",
    popsize: int = 20,
    use_fingerprinting_init: bool = False,
    fingerprinting_top_k: Optional[int] = None,
) -> Tuple[Dict[str, int], float, Optional[pd.DataFrame]]:
    """Perform a black‑box attack by swapping categorical feature values.

    Parameters
    ----------
    detection : object
        Detection object exposing a ``run_predict(df: pd.DataFrame) -> (labels, preds)``
        method.  The predictions should be binary or categorical and will be
        compared against the baseline predictions.
    df : pandas.DataFrame
        The dataset on which to perform the attack.  All columns are assumed to
        be categorical; numeric features should have been preprocessed or
        discretised beforehand.  The DataFrame must contain all features
        listed in ``modifiable_features``.
    modifiable_features : Iterable[str], optional
        A list of feature names that can be modified.  If ``None``, the list
        of features defined in :data:`MODIFIABLE_FEATURES` is used.  Features
        absent from ``df.columns`` will be ignored.
    budget : int, default 50
        The maximum number of evaluations that the optimiser is allowed to
        perform.  Increasing this value typically yields better attacks but
        increases run time.
    noise_level : float, default 0.1
        The fraction of rows to perturb for each candidate solution.  A value
        of 0.1 means that 10%% of the entries for each feature will be swapped.
    tmp_dir : str, optional
        Directory for intermediate preprocessing files.  If ``None``, a
        temporary directory under ``/tmp`` will be created.
    return_perturbed : bool, default False
        If ``True``, the perturbed DataFrame achieving the highest
        misclassification is returned as the third element of the tuple.

    Additional Parameters
    ---------------------
    protocol_column : str, default 'ip.proto'
        The name of the column indicating the layer‑4 protocol (e.g., 1 for
        ICMP, 6 for TCP, 17 for UDP).  When ``protocol_constraints`` is
        provided, this column is used to restrict modifications to rows
        whose protocol appears in the constraint mapping for a given
        feature.
    protocol_constraints : dict, optional
        A mapping from protocol values (converted to strings) to lists
        of feature names.  When supplied, only rows whose protocol
        matches the feature’s allowed protocol set will be perturbed.  If
        ``None``, all features are considered modifiable across all
        protocols.

    Returns
    -------
    best_params : dict
        A mapping from feature name to the index of the chosen category in its
        list of unique training values.  This dictionary describes the
        recommended adversarial configuration.
    improvement : float
        The increase in misclassification rate (in percentage points) between
        the baseline and the adversarial sample found by the optimisation.
    perturbed_df : pandas.DataFrame, optional
        The perturbed dataset corresponding to ``best_params`` if
        ``return_perturbed=True``; otherwise ``None``.

    Notes
    -----
    The attack modifies the dataset only in memory; original files are not
    touched.  Intermediate preprocessed files are written into the directory
    specified by ``tmp_dir`` (or a temporary directory under ``/tmp``) and
    deleted at the end of the run.
    """

    # Validate the list of modifiable features
    if modifiable_features is None:
        modifiable_features = list(MODIFIABLE_FEATURES)
    else:
        modifiable_features = list(modifiable_features)

    # Keep only features that are actually present in the DataFrame
    modifiable_features = [f for f in modifiable_features if f in df.columns]
    if not modifiable_features:
        raise ValueError(
            "No modifiable features are available in the provided DataFrame."
        )

    # ---------------------------------------------
    # Optional: pre-filter features via fingerprinting
    # (DroidBreaker-style model fingerprinting)
    # ---------------------------------------------
    if use_fingerprinting_init:
        from attack import perform_fingerprinting_modifiable_categorical_clean

        # Per risparmiare tempo, puoi campionare un sottoinsieme di righe
        # (es. 50k) se il dataset è enorme.
        df_fp = df

        tested_feats, sensitivities = perform_fingerprinting_modifiable_categorical_clean(
            detection=detection,
            model=getattr(detection, "model", None),
            df=df_fp,
            threshold=1.0,   # questo threshold interno viene solo usato per i log
            in_memory=True
        )

        if tested_feats:
            sens_array = np.array(sensitivities)
            # Costruiamo una mappa feature -> sensitivity (in %)
            fp_scores = {f: s for f, s in zip(tested_feats, sens_array)}

            # Se fingerprinting_top_k è specificato, teniamo solo le top-k
            if fingerprinting_top_k is not None and fingerprinting_top_k > 0:
                # ordina per sensitività decrescente
                sorted_feats = sorted(
                    fp_scores.keys(), key=lambda x: fp_scores[x], reverse=True
                )
                selected = set(sorted_feats[:fingerprinting_top_k])
            else:
                # altrimenti teniamo tutte quelle > 0 (o > threshold se vuoi)
                selected = {f for f, s in fp_scores.items() if s > 0.0}

            # Intersezione con le modifiable_features passate all’attacco
            before = set(modifiable_features)
            modifiable_features = [f for f in modifiable_features if f in selected]

            if not modifiable_features:
                # fallback: se per qualche motivo abbiamo filtrato via tutto,
                # torniamo alla lista originale per non rompere l’attacco
                modifiable_features = list(before)
            else:
                print(
                    f"[NetBreaker] Fingerprinting init attivo: "
                    f"{len(modifiable_features)}/{len(before)} feature mantenute "
                    f"({', '.join(modifiable_features)})"
                )
        else:
            print("[NetBreaker] Fingerprinting init attivo ma nessuna feature testata; uso lista originale.")
    # ---------------------------------------------

    # Create a temporary directory for intermediate preprocessing files
    tmp_base = tmp_dir or "/tmp/blackbox_attack_ng"
    os.makedirs(tmp_base, exist_ok=True)

    # Compute baseline predictions on the original data
    # Use a separate subdirectory to avoid clashes with modifications
    baseline_out_dir = os.path.join(tmp_base, "baseline")
    os.makedirs(baseline_out_dir, exist_ok=True)
    baseline_df_pp = preprocessing_pipeline_partial(
        df=df.copy(), output_dir=baseline_out_dir, dataset_name="baseline"
    )
    _, y_pred_baseline = detection.run_predict(baseline_df_pp)
    y_pred_baseline = np.asarray(y_pred_baseline)

    # Build dictionary of unique values for each modifiable feature
    # Drop NaN values to avoid swapping with missing values
    unique_values: Dict[str, List] = {
        feature: df[feature].dropna().unique().tolist() for feature in modifiable_features
    }

    # If protocol constraints are provided, build a mapping from each
    # feature to the set of protocol values (as strings) where that
    # feature appears.  This will be used to restrict perturbations to
    # rows whose ``protocol_column`` matches the allowed protocols for
    # each feature.  If no constraints are provided, all features are
    # allowed for all protocols.
    if protocol_constraints is not None:
        # invert the mapping: feature -> list of protocols
        feature_protocols: Dict[str, List[str]] = {}
        for proto_val, feature_list in protocol_constraints.items():
            for feat in feature_list:
                if feat in modifiable_features:
                    feature_protocols.setdefault(feat, []).append(proto_val)
    else:
        feature_protocols = {feat: [] for feat in modifiable_features}

    # Define Nevergrad parametrisation: each feature is mapped to a discrete
    # choice over the indices of its unique values. The ``Choice`` parameter can be used to
    # represent discrete variables.
    param_kwargs = {
        f"idx_{feature}": ng.p.Choice(range(len(unique_values[feature]) or 1))
        for feature in modifiable_features
    }
    parametrization = ng.p.Instrumentation(**param_kwargs)

    # Cache evaluation of the baseline to avoid repeated preprocessing when
    # the optimiser explores the same configuration multiple times
    baseline_cache: Dict[Tuple[int, ...], float] = {}

    def evaluate(**kwargs: int) -> float:
        """Objective function to be minimised by Nevergrad.

        The function constructs a perturbed version of ``df`` according to the
        indices provided in ``kwargs``, preprocesses it, and computes the
        misclassification rate relative to the baseline predictions.  Since
        Nevergrad minimises the objective, we return the *negative* of the
        misclassification rate: maximising the error equates to minimising
        ``-error``.
        """
        # Convert the kwargs into a tuple key for caching
        param_tuple = tuple(kwargs[f"idx_{feature}"] for feature in modifiable_features)
        if param_tuple in baseline_cache:
            return baseline_cache[param_tuple]

        # Create a copy of the dataframe to apply perturbations
        df_mod = df.copy()
        # Retrieve protocol column values as strings for matching
        proto_series = df_mod[protocol_column].astype(str) if protocol_column in df_mod.columns else pd.Series(["" for _ in range(len(df_mod))])
        for feature in modifiable_features:
            choices = unique_values[feature]
            if not choices:
                continue
            selected_idx = kwargs[f"idx_{feature}"] % len(choices)
            selected_val = choices[selected_idx]
            # Determine allowed rows: if protocol constraints exist, only
            # perturb rows whose protocol appears in feature_protocols[feature];
            # otherwise allow all rows.  Convert protocol values to strings
            # for comparison.
            if protocol_constraints is not None:
                allowed_protocols = feature_protocols.get(feature, [])
                if allowed_protocols:
                    allowed_mask = proto_series.isin(allowed_protocols)
                else:
                    # If the feature does not appear in any protocol list,
                    # it should not be perturbed
                    allowed_mask = pd.Series([False] * len(df_mod))
            else:
                allowed_mask = pd.Series([True] * len(df_mod))
            # Apply the swap on a fraction of rows within the allowed mask
            # determined by noise_level
            random_mask = np.random.rand(len(df_mod)) < noise_level
            mask = allowed_mask & random_mask
            if mask.any():
                df_mod.loc[mask, feature] = selected_val

        # Preprocess modified data
        mod_out_dir = os.path.join(tmp_base, "mod")
        os.makedirs(mod_out_dir, exist_ok=True)
        df_mod_pp = preprocessing_pipeline_partial(
            df=df_mod, output_dir=mod_out_dir, dataset_name="mod"
        )
        _, y_pred_mod = detection.run_predict(df_mod_pp)
        y_pred_mod = np.asarray(y_pred_mod)

        # Compute misclassification rate relative to baseline
        if y_pred_mod.shape != y_pred_baseline.shape:
            error = 0.0
        else:
            error = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
        # Cache the negative error (because we want to maximise error)
        baseline_cache[param_tuple] = -error
        return -error

    # Create optimiser with specified budget
    if optimizer_type.lower() == "es":
        # Evolution Strategy like DroidBreaker
        optimizer = ng.optimizers.EvolutionStrategy(
            recombination_ratio=1.0,
            popsize=popsize,
            only_offsprings=False,
            offsprings=popsize,
            ranker="simple",
        )(parametrization=parametrization, budget=budget)
    else:
        # Default: NGOpt
        optimizer = ng.optimizers.NGOpt(
            parametrization=parametrization, budget=budget, num_workers=1
        )

    # Perform optimisation
    recommendation = optimizer.minimize(evaluate)

    # Retrieve the best parameters and compute improvement
    best_params = {
        feature: recommendation.kwargs[f"idx_{feature}"] for feature in modifiable_features
    }
    best_improvement = -recommendation.loss  # negative of returned loss

    perturbed_df: Optional[pd.DataFrame] = None
    if return_perturbed:
        # Reconstruct the perturbed dataframe using the best parameters
        df_mod = df.copy()
        proto_series = df_mod[protocol_column].astype(str) if protocol_column in df_mod.columns else pd.Series(["" for _ in range(len(df_mod))])
        for feature in modifiable_features:
            choices = unique_values[feature]
            if not choices:
                continue
            selected_idx = best_params[feature] % len(choices)
            selected_val = choices[selected_idx]
            if protocol_constraints is not None:
                allowed_protocols = feature_protocols.get(feature, [])
                if allowed_protocols:
                    allowed_mask = proto_series.isin(allowed_protocols)
                else:
                    allowed_mask = pd.Series([False] * len(df_mod))
            else:
                allowed_mask = pd.Series([True] * len(df_mod))
            random_mask = np.random.rand(len(df_mod)) < noise_level
            mask = allowed_mask & random_mask
            if mask.any():
                df_mod.loc[mask, feature] = selected_val
        perturbed_df = df_mod

    # Clean up temporary directories if they were created by the function
    if tmp_dir is None:
        # Remove only the temporary directory created by this run
        try:
            # Recursively remove contents of tmp_base
            for root, dirs, files in os.walk(tmp_base, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except FileNotFoundError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            os.rmdir(tmp_base)
        except OSError:
            # If directory cannot be removed, leave it in place
            pass

    return best_params, best_improvement, perturbed_df
