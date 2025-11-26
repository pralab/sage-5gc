import logging
from pathlib import Path
from typing import Dict

import nevergrad as ng
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
        sample_idx: int,
        df: pd.DataFrame,
        detector: object,
        query_budget: int = 100,
    ) -> None:
        """"""
        self._query_budget = query_budget
        sample = df.iloc[sample_idx]

        if sample["ip.proto"] == 1:  # ICMP
            return

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        baseline_out_dir = self._tmp_dir / "baseline"
        baseline_out_dir.mkdir(parents=True, exist_ok=True)

        baseline_df_pp = preprocessing_pipeline_single_dataset(
            df=df.copy(), output_dir=str(baseline_out_dir), dataset_name="baseline"
        )
        orig_score = detector.get_score(baseline_df_pp, sample_idx)
        logger.info(f"Original score for sample {sample_idx}: {orig_score}")

        # Early stopping if sample is already classified as benign
        if orig_score == 0.0:
            logger.info(
                f"Sample {sample_idx} is already classified as benign. Skipping."
            )
            return

        if hasattr(detector, "best_threshold"):
            if orig_score < detector.best_threshold:
                logger.info(
                    f"Sample {sample_idx} is already classified as benign. Skipping."
                )
                return

        if hasattr(detector, "threshold"):
            if orig_score < detector.threshold:
                logger.info(
                    f"Sample {sample_idx} is already classified as benign. Skipping."
                )
                return

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
            loss = self._compute_loss(sample_idx, x_adv, df, detector)
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
        self, sample_idx: int, x_adv: pd.Series, df: pd.DataFrame, detector: object
    ) -> float:
        # Create the adversarial dataset including the modified sample
        adv_df = df.copy()
        adv_df.loc[sample_idx] = x_adv

        # Preprocess the adversarial dataset
        adv_out_dir = self._tmp_dir / "adv"
        adv_out_dir.mkdir(parents=True, exist_ok=True)
        adv_df_pp = preprocessing_pipeline_single_dataset(
            df=adv_df, output_dir=str(adv_out_dir), dataset_name="adv"
        )

        return detector.get_score(adv_df_pp, sample_idx)

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
    from ml_models import DetectionKnn, DetectionRandomForest, DetectionIsolationForest

    path = Path(__file__).parent / "data/cleaned_datasets/dataset_3_cleaned.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False, encoding="utf-8")

    knn_det = DetectionIsolationForest()
    knn_det.load_model(
        str(Path(__file__).parent / "trained_models/isolation_forest.pkl")
    )

    bb = BlackBoxAttack()

    for idx, row in dataset.iterrows():
        if pd.isna(row["ip.opt.time_stamp"]):
            continue

        bb.run(idx, dataset.copy(), knn_det, query_budget=100)
