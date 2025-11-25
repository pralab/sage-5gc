import logging
from pathlib import Path
from typing import Dict

import nevergrad as ng
import numpy as np
import pandas as pd

from preprocessing.preprocessor import preprocessing_pipeline_single_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BlackBoxAttack:
    """Black-box attack for network traffic classifiers."""

    def __init__(self) -> None:
        """"""
        self._optimizer_cls = None
        self._optimizer = None
        self._query_budget = None

        # Create a temporary directory for intermediate preprocessing files
        self._tmp_dir = Path(__file__).parent / "data/tmp/"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        idx_sample: int,
        dataset: pd.DataFrame,
        detector: object,
        query_budget: int = 100,
    ) -> None:
        """"""
        self._query_budget = query_budget
        sample = dataset.iloc[idx_sample]

        if sample["ip.proto"] == 1:  # ICMP
            return

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        baseline_out_dir = self._tmp_dir / "baseline"
        baseline_out_dir.mkdir(parents=True, exist_ok=True)

        baseline_df_pp = preprocessing_pipeline_single_dataset(
            df=dataset.copy(), output_dir=str(baseline_out_dir), dataset_name="baseline"
        )
        _, y_pred_baseline = detector.run_predict(baseline_df_pp)
        y_pred_baseline = np.asarray(y_pred_baseline)

        # --------------------------
        # [Step 2] Set up optimizer
        # --------------------------
        self._optimizer_cls = ng.optimizers.EvolutionStrategy(
            recombination_ratio=0.9,
            popsize=20,
            only_offsprings=False,
            offsprings=20,
            ranker="simple",
        )
        self._optimizer = self._init_optimizer(is_tcp=(sample["ip.proto"] == 6))

        # --------------------
        # [Step 3] Run attack
        # --------------------
        for idx in range(self._query_budget):
            x = self._optimizer.ask()
            x_adv = self._apply_modifications(sample, x.value)
            loss = self._compute_loss(x_adv, detector)
            logger.info(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")
            self._optimizer.tell(x, loss)

        # ----------------------
        # [Step 4] Save results
        # ----------------------
        recommendation = self._optimizer.provide_recommendation()
        best_params = recommendation.value
        best_loss = recommendation.loss

        logger.info(f"Best params: {best_params}")
        logger.info(f"Best loss: {best_loss}")

    def _apply_modifications(
        self, sample: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        for feature, value in params.items():
            key = feature.replace("_", ".", 1)
            sample[key] = value

        # TODO: Recompute ip.dsfield from ip.dsfield.dscp

        return sample

    def _compute_loss(
        self, idx_sample: int, x_adv: pd.Series, dataset: pd.DataFrame, detector: object
    ) -> float:
        adv_ds = dataset.copy()
        adv_ds.loc[idx_sample] = x_adv

        adv_out_dir = self._tmp_dir / "adv"
        adv_out_dir.mkdir(parents=True, exist_ok=True)

        adv_df_pp = preprocessing_pipeline_single_dataset(
            df=adv_ds, output_dir=str(adv_out_dir), dataset_name="adv"
        )
        detector.run_predict(adv_df_pp)

    def _init_optimizer(self, is_tcp: bool) -> ng.optimizers.base.Optimizer:
        base_param = ng.p.Dict(
            ip_ttl=ng.p.Scalar(lower=1, upper=255).set_integer_casting(),
            ip_id=ng.p.Scalar(lower=0, upper=2**16 - 1).set_integer_casting(),
            ip_flags_df=ng.p.Choice([0, 1]),
            ip_dsfield_dscp=ng.p.Scalar(lower=0, upper=63).set_integer_casting(),
            pfcp_duration_measurement=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_time_of_first_packet=None,
            pfcp_time_of_last_packet=None,
            pfcp_end_time=None,
            pfcp_recovery_time_stamp=None,
            pfcp_volume_measurement_dlnop=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_volume_measurement_dlvol=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_volume_measurement_tonop=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_volume_measurement_tovol=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_user_id_imei=ng.p.Scalar(
                lower=0, upper=999999999999999
            ).set_integer_casting(),
            pfcp_ue_ip_address_flag_sd=ng.p.Choice([0, 1]),
            pfcp_f_teid_flags_ch=ng.p.Choice([0, 1]),
            pfcp_f_teid_flags_ch_id=ng.p.Choice([0, 1]),
            pfcp_f_teid_flags_v6=ng.p.Choice([0, 1]),
        )

        if is_tcp:
            tcp_param = ng.p.Dict(
                **base_param,
                tcp_srcport=ng.p.Scalar(lower=0, upper=65000).set_integer_casting(),
                tcp_seq_raw=ng.p.Scalar(lower=0, upper=2**32 - 1).set_integer_casting(),
                tcp_ack_raw=ng.p.Scalar(lower=0, upper=2**32 - 1).set_integer_casting(),
                tcp_flags_urg=ng.p.Choice([0, 1]),
                tcp_flags_ack=ng.p.Choice([0, 1]),
                tcp_flags_psh=ng.p.Choice([0, 1]),
                tcp_flags_rst=ng.p.Choice([0, 1]),
                tcp_flags_syn=ng.p.Choice([0, 1]),
                tcp_flags_fin=ng.p.Choice([0, 1]),
                tcp_options_timestamp_tsval=ng.p.Scalar(
                    lower=0, upper=2**32 - 1
                ).set_integer_casting(),
                tcp_options_timestamp_tsecr=ng.p.Scalar(
                    lower=0, upper=2**32 - 1
                ).set_integer_casting(),
                tcp_window_size_value=ng.p.Scalar(
                    lower=0, upper=2**16 - 1
                ).set_integer_casting(),
            )
        else:
            udp_param = ng.p.Dict(
                **base_param,
                udp_srcport=ng.p.Scalar(lower=0, upper=65000).set_integer_casting(),
            )

        return self._optimizer_cls(
            tcp_param if is_tcp else udp_param, budget=self._query_budget
        )


if __name__ == "__main__":
    from ml_models import DetectionKnn

    path = Path(__file__).parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
    ds = pd.read_csv(path, sep=";", low_memory=False, encoding="utf-8")

    knn_det = DetectionKnn()
    knn_det.load_model(str(Path(__file__).parent / "trained_models/knn.pkl"))

    bb = BlackBoxAttack()
    bb.run(1, ds, knn_det, query_budget=100)


# def blackbox_attack():
#     """"""

#     # Cache evaluation of the baseline to avoid repeated preprocessing when
#     # the optimiser explores the same configuration multiple times
#     baseline_cache: Dict[Tuple[int, ...], float] = {}

#     def evaluate(**kwargs: int) -> float:
#         """Objective function to be minimised by Nevergrad.

#         The function constructs a perturbed version of ``df`` according to the
#         indices provided in ``kwargs``, preprocesses it, and computes the
#         misclassification rate relative to the baseline predictions.  Since
#         Nevergrad minimises the objective, we return the *negative* of the
#         misclassification rate: maximising the error equates to minimising
#         ``-error``.
#         """
#         # Convert the kwargs into a tuple key for caching
#         param_tuple = tuple(kwargs[f"idx_{feature}"] for feature in modifiable_features)
#         if param_tuple in baseline_cache:
#             return baseline_cache[param_tuple]

#         # Create a copy of the dataframe to apply perturbations
#         df_mod = df.copy()
#         # Retrieve protocol column values as strings for matching
#         proto_series = (
#             df_mod[protocol_column].astype(str)
#             if protocol_column in df_mod.columns
#             else pd.Series(["" for _ in range(len(df_mod))])
#         )
#         for feature in modifiable_features:
#             choices = unique_values[feature]
#             if not choices:
#                 continue
#             selected_idx = kwargs[f"idx_{feature}"] % len(choices)
#             selected_val = choices[selected_idx]
#             # Determine allowed rows: if protocol constraints exist, only
#             # perturb rows whose protocol appears in feature_protocols[feature];
#             # otherwise allow all rows.  Convert protocol values to strings
#             # for comparison.
#             if protocol_constraints is not None:
#                 allowed_protocols = feature_protocols.get(feature, [])
#                 if allowed_protocols:
#                     allowed_mask = proto_series.isin(allowed_protocols)
#                 else:
#                     # If the feature does not appear in any protocol list,
#                     # it should not be perturbed
#                     allowed_mask = pd.Series([False] * len(df_mod))
#             else:
#                 allowed_mask = pd.Series([True] * len(df_mod))
#             # Apply the swap on a fraction of rows within the allowed mask
#             # determined by noise_level
#             random_mask = np.random.rand(len(df_mod)) < noise_level
#             mask = allowed_mask & random_mask
#             if mask.any():
#                 df_mod.loc[mask, feature] = selected_val

#         # Preprocess modified data
#         mod_out_dir = os.path.join(tmp_base, "mod")
#         os.makedirs(mod_out_dir, exist_ok=True)
#         df_mod_pp = preprocessing_pipeline_partial(
#             df=df_mod, output_dir=mod_out_dir, dataset_name="mod"
#         )
#         _, y_pred_mod = detection.run_predict(df_mod_pp)
#         y_pred_mod = np.asarray(y_pred_mod)

#         # Compute misclassification rate relative to baseline
#         if y_pred_mod.shape != y_pred_baseline.shape:
#             error = 0.0
#         else:
#             error = float(np.mean(y_pred_mod != y_pred_baseline)) * 100.0
#         # Cache the negative error (because we want to maximise error)
#         baseline_cache[param_tuple] = -error
#         return -error

#     # Create optimiser with specified budget

#     # Perform optimisation
#     recommendation = optimizer.minimize(evaluate)

#     # Retrieve the best parameters and compute improvement
#     best_params = {
#         feature: recommendation.kwargs[f"idx_{feature}"]
#         for feature in modifiable_features
#     }
#     best_improvement = -recommendation.loss  # negative of returned loss

#     perturbed_df: Optional[pd.DataFrame] = None
#     if return_perturbed:
#         # Reconstruct the perturbed dataframe using the best parameters
#         df_mod = df.copy()
#         proto_series = (
#             df_mod[protocol_column].astype(str)
#             if protocol_column in df_mod.columns
#             else pd.Series(["" for _ in range(len(df_mod))])
#         )
#         for feature in modifiable_features:
#             choices = unique_values[feature]
#             if not choices:
#                 continue
#             selected_idx = best_params[feature] % len(choices)
#             selected_val = choices[selected_idx]
#             if protocol_constraints is not None:
#                 allowed_protocols = feature_protocols.get(feature, [])
#                 if allowed_protocols:
#                     allowed_mask = proto_series.isin(allowed_protocols)
#                 else:
#                     allowed_mask = pd.Series([False] * len(df_mod))
#             else:
#                 allowed_mask = pd.Series([True] * len(df_mod))
#             random_mask = np.random.rand(len(df_mod)) < noise_level
#             mask = allowed_mask & random_mask
#             if mask.any():
#                 df_mod.loc[mask, feature] = selected_val
#         perturbed_df = df_mod

#     # Clean up temporary directories if they were created by the function
#     if tmp_dir is None:
#         # Remove only the temporary directory created by this run
#         try:
#             # Recursively remove contents of tmp_base
#             for root, dirs, files in os.walk(tmp_base, topdown=False):
#                 for name in files:
#                     try:
#                         os.remove(os.path.join(root, name))
#                     except FileNotFoundError:
#                         pass
#                 for name in dirs:
#                     try:
#                         os.rmdir(os.path.join(root, name))
#                     except OSError:
#                         pass
#             os.rmdir(tmp_base)
#         except OSError:
#             # If directory cannot be removed, leave it in place
#             pass

#     return best_params, best_improvement, perturbed_df
