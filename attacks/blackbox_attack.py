import json
import logging
from pathlib import Path
import sys
from typing import Dict

import nevergrad as ng
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BlackBoxAttack:
    """Black-box attack for network traffic classifiers."""

    def __init__(self, optimizer_cls: ng.optimization.Optimizer) -> None:
        """
        Create a BlackBoxAttack instance.

        Parameters
        ----------
        optimizer_cls : ng.optimization.Optimizer
            The Nevergrad optimizer class to use for the attack.
        """
        self._optimizer_cls = optimizer_cls
        self._optimizer = None
        self._query_budget = None

        # Create a temporary directory for intermediate preprocessing files
        self._tmp_dir = Path(__file__).parent.parent / "tmp/"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        sample_idx: int,
        df: pd.DataFrame,
        detector: Detector,
        query_budget: int = 100,
    ) -> None:
        """
        Run the black-box attack on a given sample.

        Parameters
        ----------
        sample_idx : int
            The index of the sample to attack in the DataFrame.
        df : pd.DataFrame
            The DataFrame containing the dataset.
        detector : Detector
            The detector object with a get_score method.
        query_budget : int
            The maximum number of queries allowed for the attack.
        """
        self._query_budget = query_budget
        sample = df.iloc[sample_idx]

        if sample["ip.proto"] == 1:  # ICMP
            return

        logger.info(f"Starting attack on sample {sample_idx}...")

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        orig_score = detector.decision_function(df, sample_idx)
        logger.info(f"Original score for sample {sample_idx}: {orig_score}")

        logger.info(f"true label {detector.predict(df)[sample_idx]}")

        if orig_score < 0.0:
            logger.info(
                f"Sample {sample_idx} is already classified as benign. Skipping attack."
            )
            return

        # --------------------------
        # [Step 2] Set up optimizer
        # --------------------------
        self._optimizer = self._init_optimizer(is_tcp=(sample["ip.proto"] == 6))

        # --------------------
        # [Step 3] Run attack
        # --------------------
        for idx in range(self._optimizer.budget):
            x = self._optimizer.ask()
            x_adv = self._apply_modifications(sample, x.value)
            loss = self._compute_loss(sample_idx, x_adv, df, detector)
            logger.info(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")
            print(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")
            if loss < 0.0:
                logger.info(
                    f"Sample {sample_idx} evaded the detector after {idx + 1} queries."
                )
                break

            self._optimizer.tell(x, loss)

        # ----------------------
        # [Step 4] Save results
        # ----------------------
        recommendation = self._optimizer.provide_recommendation()
        best_params = recommendation.value
        best_loss = recommendation.loss
        self._save_results(sample_idx, pd.Series(best_params), best_loss, detector)

    def _save_results(
        self,
        sample_idx: int,
        best_params: pd.Series,
        best_loss: float,
        detector: object,
    ) -> None:
        results_path = Path(__file__).parent / "results/isolation.json"
        if results_path.exists():
            with results_path.open("r") as f:
                results = json.load(f)
        else:
            results = {}

        results[str(sample_idx)] = {
            "best_params": best_params.to_dict(),
            "best_loss": best_loss,
            "evaded": best_loss < detector.best_threshold,
        }

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _apply_modifications(
        self, sample: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        adv_sample = sample.copy()
        for feature, value in params.items():
            key = feature.replace("_", ".", 1)

            if key == "ip.id":
                adv_sample[key] = hex(value)
            if key == "ip.flags.df":
                adv_sample[key] = value
                # Modify the corresponding bit in ip.flags
                adv_sample["ip.flags"] = hex(
                    (adv_sample["ip.flags"] & ~0x02) | ((value & 0x01) << 1)
                )
            if key == "ip.dsfield.dscp":
                adv_sample[key] = value
                # Modify the corresponding bits in ip.dsfield
                adv_sample["ip.dsfield"] = hex(
                    (adv_sample["ip.dsfield"] & 0x03) | ((value & 0x3F) << 2)
                )
            else:
                adv_sample[key] = value

        return adv_sample

    def _compute_loss(
        self, sample_idx: int, x_adv: pd.Series, df: pd.DataFrame, detector: Detector
    ) -> float:
        adv_df = df.copy()
        adv_df.loc[sample_idx] = x_adv
        return detector.decision_function(adv_df, sample_idx)

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
            param = ng.p.Dict(
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
            param = ng.p.Dict(
                **base_param,
                udp_srcport=ng.p.Scalar(lower=0, upper=65000).set_integer_casting(),
            )

        param.random_state = np.random.RandomState(42)

        return self._optimizer_cls(parametrization=param, budget=self._query_budget)


if __name__ == "__main__":
    import joblib
    from pyod.models.hbos import HBOS

    path = Path(__file__).parent.parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False)

    optimizer_cls = ng.optimizers.EvolutionStrategy(
        recombination_ratio=0.9,
        popsize=20,
        only_offsprings=False,
        offsprings=20,
        ranker="simple",
    )

    bb = BlackBoxAttack(optimizer_cls)

    detector = joblib.load(
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl"
    )
    #logger.info(f"Loaded trained HBOS detector with threshold {detector.detector.threshold_}")

    results_path = Path(__file__).parent.parent / "results/hbos.json"
    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}

    for idx, row in dataset.iterrows():
        if str(idx) in results:
            logger.info(f"Sample {idx} already attacked. Skipping.")
            continue

        if pd.isna(row["ip.opt.time_stamp"]):
            continue

        bb.run(idx, dataset.copy(), detector, query_budget=60)
