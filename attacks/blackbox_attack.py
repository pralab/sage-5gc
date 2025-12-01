from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
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

        random.seed(42)

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
        logger.info(f"Detector threshold: {detector.detector.threshold_}")

        if orig_score < detector.detector.threshold_:
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

            if loss < detector.detector.threshold_:
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

    def _compute_loss(
        self, sample_idx: int, x_adv: pd.Series, df: pd.DataFrame, detector: Detector
    ) -> float:
        adv_df = df.copy()
        adv_df.loc[sample_idx] = x_adv
        return detector.decision_function(adv_df, sample_idx)

    def _apply_modifications(
        self, sample: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        adv_sample = sample.copy()
        for feature, value in params.items():
            key = feature.replace("_", ".", 1)

            if key == "ip.id":
                adv_sample[key] = hex(value)

            elif key == "ip.flags.df":
                adv_sample[key] = value
                # Modify the corresponding bit in ip.flags
                adv_sample["ip.flags"] = hex(
                    (adv_sample["ip.flags"] & ~0x02) | ((value & 0x01) << 1)
                )

            elif key == "ip.dsfield.dscp":
                adv_sample[key] = value
                # Modify the corresponding bits in ip.dsfield
                adv_sample["ip.dsfield"] = hex(
                    (adv_sample["ip.dsfield"] & 0x03) | ((value & 0x3F) << 2)
                )

            elif key == "pfcp.recovery_time_stamp":
                recovery_time = datetime.fromtimestamp(value, tz=timezone.utc)
                times = gen_pfcp_times_exp(recovery_time)
                adv_sample["pfcp.recovery_time_stamp"] = times["recovery_time_stamp"]
                adv_sample["pfcp.time_of_first_packet"] = times["time_of_first_packet"]
                adv_sample["pfcp.time_of_last_packet"] = times["time_of_last_packet"]
                adv_sample["pfcp.end_time"] = times["end_time"]

            elif key == "pfcp.response_time":
                adv_sample[key] = value / 1_000_000.0

            elif key == "udp.srcport":
                adv_sample[key] = value
                adv_sample["udp.port"] = value

            elif key == "tcp.srcport":
                adv_sample[key] = value
                adv_sample["tcp.port"] = value

            else:
                adv_sample[key] = value

        return adv_sample

    def _init_optimizer(self, is_tcp: bool) -> ng.optimizers.base.Optimizer:
        base_param = ng.p.Dict(
            # IP layer
            ip_ttl=ng.p.Scalar(lower=1, upper=255).set_integer_casting(),
            ip_id=ng.p.Scalar(lower=0, upper=2**16 - 1).set_integer_casting(),
            ip_flags_df=ng.p.Choice([0, 1]),
            ip_dsfield_dscp=ng.p.Scalar(lower=0, upper=63).set_integer_casting(),

            # PFCP layer
            pfcp_duration_measurement=ng.p.Scalar(
                lower=0, upper=2**32 - 1
            ).set_integer_casting(),
            pfcp_response_time=ng.p.Scalar(lower=0, upper=5000),
            pfcp_recovery_time_stamp=ng.p.Scalar(
                lower=int(
                    datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
                ),
                upper=int(
                    datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
                )
                - 1,
            ).set_integer_casting(),
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
            # pfcp_user_id_imei=ng.p.Scalar(
            #     lower=0, upper=999999999999999
            # ).set_integer_casting(),
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


def gen_pfcp_times_exp(
    recovery_time_stamp: datetime,
    mean_start_delay_sec: int = 30 * 60,  # Tipicaly within 30 min from recovery
    mean_session_duration_sec: int = 20 * 60,  # Average duration 20 min
    mean_end_delay_sec: int = 30,  # Shortly after last pkt (~30s)
) -> Dict[str, str]:
    def exp_sec(mean: float) -> int:
        """Generates an exponential random variable in seconds with given mean."""
        return max(0, int(random.expovariate(1.0 / mean)))

    # Generates PFCP time fields based on exponential distributions
    start_offset = exp_sec(mean_start_delay_sec)
    time_of_first_packet = recovery_time_stamp + timedelta(seconds=start_offset)

    traffic_duration = exp_sec(mean_session_duration_sec)
    time_of_last_packet = time_of_first_packet + timedelta(seconds=traffic_duration)

    end_delay = exp_sec(mean_end_delay_sec)
    end_time = time_of_last_packet + timedelta(seconds=end_delay)

    # Format timestamps with random nanoseconds
    def fmt(dt: datetime) -> str:
        base = dt.strftime("%b %d, %Y %H:%M:%S")
        nanos = random.randint(0, 999_999_999)
        return f"{base}.{nanos:09d} UTC"

    return {
        "recovery_time_stamp": fmt(recovery_time_stamp),
        "time_of_first_packet": fmt(time_of_first_packet),
        "time_of_last_packet": fmt(time_of_last_packet),
        "end_time": fmt(end_time),
    }


if __name__ == "__main__":
    import joblib
    from pyod.models.hbos import HBOS

    path = Path(__file__).parent.parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False)

    # optimizer_cls = ng.optimizers.EvolutionStrategy(
    #     recombination_ratio=0.9,
    #     popsize=20,
    #     only_offsprings=False,
    #     offsprings=20,
    #     ranker="simple",
    # )
    optimizer_cls = ng.optimizers.DifferentialEvolution(
        popsize=20,
        crossover="twopoints",
        propagate_heritage=True,
    )

    bb = BlackBoxAttack(optimizer_cls)

    detector = joblib.load(
        Path(__file__).parent.parent / "data/trained_models/IForest.pkl"
    )

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
