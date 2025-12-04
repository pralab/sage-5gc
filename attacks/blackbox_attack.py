from datetime import datetime
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

        random.seed(42)

    def run(
        self,
        sample_idx: int,
        sample: pd.Series,
        detector_name: str,
        detector: Detector,
        query_budget: int = 100,
    ) -> None:
        """
        Run the black-box attack on a given sample.

        Parameters
        ----------
        sample_idx : int
            The index of the sample in the dataset.
        sample: pd.Series
            The sample to attack.
        detector_name : str
            The name of the detector being attacked.
        detector : Detector
            The detector object with a get_score method.
        query_budget : int
            The maximum number of queries allowed for the attack.
        """
        self._query_budget = query_budget

        logger.info("Starting attack on sample...")

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        orig_score = detector.decision_function(pd.DataFrame([sample]))[0]
        logger.info(f"Original score: {orig_score}")
        logger.info(f"Detector threshold: {detector._detector.threshold_}")

        if orig_score < detector._detector.threshold_:
            logger.info("Sample is already classified as benign. Skipping attack.")
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
            loss = self._compute_loss(x_adv, detector)
            logger.info(f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}")

            if loss < detector._detector.threshold_:
                logger.info(f"Sample evaded the detector after {idx + 1} queries.")
                break

            self._optimizer.tell(x, loss)

        # ----------------------
        # [Step 4] Save results
        # ----------------------
        recommendation = self._optimizer.provide_recommendation()
        best_params = recommendation.value
        best_loss = recommendation.loss
        self._save_results(
            sample_idx,
            best_params,
            best_loss,
            detector_name,
            loss < detector._detector.threshold_,
        )

    def _init_optimizer(self, is_tcp: bool) -> ng.optimizers.base.Optimizer:
        param = ng.p.Dict(
            # IP layer
            ip_ttl=ng.p.Scalar(lower=2, upper=200).set_integer_casting(),
            ip_id=ng.p.Scalar(lower=0, upper=65534).set_integer_casting(),
            ip_len=ng.p.Scalar(lower=44, upper=653).set_integer_casting(),
            ip_checksum=ng.p.Scalar(lower=54, upper=65525).set_integer_casting(),
            # UDP layer
            udp_length=ng.p.Scalar(lower=24, upper=633).set_integer_casting(),
            udp_checksum=ng.p.Scalar(lower=2389, upper=41240).set_integer_casting(),
            # PFCP layer
            pfcp_apply_action_buff=ng.p.Choice([True, False]),
            pfcp_apply_action_forw=ng.p.Choice([True, False]),
            pfcp_apply_action_nocp=ng.p.Choice([True, False]),
            pfcp_f_teid_flags_ch=ng.p.Choice([True, False]),
            pfcp_f_teid_flags_ch_id=ng.p.Choice([True, False]),
            pfcp_f_teid_flags_v6=ng.p.Choice([True, False]),
            pfcp_s=ng.p.Choice([True, False]),
            pfcp_dst_interface=ng.p.Scalar(lower=0, upper=1).set_integer_casting(),
            pfcp_duration_measurement=ng.p.Scalar(
                lower=1747212643, upper=1753894838
            ).set_integer_casting(),
            pfcp_ie_len=ng.p.Scalar(lower=1, upper=50).set_integer_casting(),
            pfcp_ie_type=ng.p.Scalar(lower=10, upper=96).set_integer_casting(),
            pfcp_length=ng.p.Scalar(lower=12, upper=621).set_integer_casting(),
            pfcp_msg_type=ng.p.Scalar(lower=1, upper=57).set_integer_casting(),
            pfcp_pdr_id=ng.p.Scalar(lower=1, upper=2).set_integer_casting(),
            pfcp_recovery_time_stamp=ng.p.Scalar(
                lower=1747207882, upper=1753892199
            ).set_integer_casting(),
            pfcp_response_time=ng.p.Scalar(lower=2.0095e-05, upper=0.041239073),
            pfcp_response_to=ng.p.Scalar(lower=1, upper=2565).set_integer_casting(),
            pfcp_seid=ng.p.Scalar(lower=0, upper=4095).set_integer_casting(),
            pfcp_seqno=ng.p.Scalar(lower=0, upper=202364).set_integer_casting(),
            pfcp_f_teid_teid=ng.p.Scalar(lower=29, upper=65507).set_integer_casting(),
            pfcp_outer_hdr_creation_teid=ng.p.Scalar(
                lower=1, upper=6326
            ).set_integer_casting(),
            pfcp_flags=ng.p.Scalar(lower=32, upper=33).set_integer_casting(),
            pfcp_volume_measurement_dlnop=ng.p.Scalar(
                lower=0, upper=13195
            ).set_integer_casting(),
            pfcp_volume_measurement_dlvol=ng.p.Scalar(
                lower=0, upper=17834134
            ).set_integer_casting(),
            pfcp_volume_measurement_tonop=ng.p.Scalar(
                lower=0, upper=13195
            ).set_integer_casting(),
            pfcp_volume_measurement_tovol=ng.p.Scalar(
                lower=0, upper=17834134
            ).set_integer_casting(),
        )
        param.random_state = np.random.RandomState(42)
        return self._optimizer_cls(parametrization=param, budget=self._query_budget)

    def _save_results(
        self,
        sample_idx: int,
        best_params: dict,
        best_loss: float,
        detector_name: str,
        evaded: bool,
    ) -> None:
        results_path = (
            Path(__file__).parent.parent
            / f"results/blackbox_attack/{detector_name}.json"
        )
        if results_path.exists():
            with results_path.open("r") as f:
                results = json.load(f)
        else:
            results = {}

        results[str(sample_idx)] = {
            "best_params": best_params,
            "best_loss": best_loss,
            "evaded": bool(evaded),
        }

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _compute_loss(self, x_adv: pd.Series, detector: Detector) -> float:
        return detector.decision_function(pd.DataFrame([x_adv]))[0]

    def _apply_modifications(
        self, sample: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        adv_sample = sample.copy()
        for feature, value in params.items():
            key = MAPPING_FEAT[feature]

            if key in [
                "ip.checksum",
                "ip.id",
                "udp.checksum",
                "pfcp.seid",
                "pfcp.f_teid.teid",
                "pfcp.outer_hdr_creation.teidpfcp.flags",
            ]:
                adv_sample[key] = hex(value)

            elif key == "pfcp.recovery_time_stamp":
                adv_sample[key] = datetime.fromtimestamp(value).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            else:
                adv_sample[key] = value

        return adv_sample


MAPPING_FEAT = {
    # IP layer
    "ip_ttl": "ip.ttl",
    "ip_id": "ip.id",
    "ip_len": "ip.len",
    "ip_checksum": "ip.checksum",
    # UDP layer
    "udp_length": "udp.length",
    "udp_checksum": "udp.checksum",
    # PFCP layer
    "pfcp_apply_action_buff": "pfcp.apply_action.buff",
    "pfcp_apply_action_forw": "pfcp.apply_action.forw",
    "pfcp_apply_action_nocp": "pfcp.apply_action.nocp",
    "pfcp_f_teid_flags_ch": "pfcp.f_teid_flags.ch",
    "pfcp_f_teid_flags_ch_id": "pfcp.f_teid_flags.ch_id",
    "pfcp_f_teid_flags_v6": "pfcp.f_teid_flags.v6",
    "pfcp_s": "pfcp.s",
    "pfcp_dst_interface": "pfcp.dst_interface",
    "pfcp_duration_measurement": "pfcp.duration_measurement",
    "pfcp_ie_len": "pfcp.ie_len",
    "pfcp_ie_type": "pfcp.ie_type",
    "pfcp_length": "pfcp.length",
    "pfcp_msg_type": "pfcp.msg_type",
    "pfcp_pdr_id": "pfcp.pdr_id",
    "pfcp_recovery_time_stamp": "pfcp.recovery_time_stamp",
    "pfcp_response_time": "pfcp.response_time",
    "pfcp_response_to": "pfcp.response_to",
    "pfcp_seid": "pfcp.seid",
    "pfcp_seqno": "pfcp.seqno",
    "pfcp_f_teid_teid": "pfcp.f_teid.teid",
    "pfcp_outer_hdr_creation_teid": "pfcp.outer_hdr_creation.teid",
    "pfcp_flags": "pfcp.flags",
    "pfcp_volume_measurement_dlnop": "pfcp.volume_measurement.dlnop",
    "pfcp_volume_measurement_dlvol": "pfcp.volume_measurement.dlvol",
    "pfcp_volume_measurement_tonop": "pfcp.volume_measurement.tonop",
    "pfcp_volume_measurement_tovol": "pfcp.volume_measurement.tovol",
}


if __name__ == "__main__":
    import joblib

    path = Path(__file__).parent.parent / "data/datasets/malicious_test_dataset.csv"
    dataset = pd.read_csv(path, sep=";", low_memory=False).drop(
        columns=["ip.opt.time_stamp"]
    )

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

    detector_name = "IForest"
    detector: Detector = joblib.load(
        Path(__file__).parent.parent / f"data/trained_models/{detector_name}.pkl"
    )

    results_path = (
        Path(__file__).parent.parent / f"results/blackbox_attack/{detector_name}.json"
    )
    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}

    for idx, row in dataset.iterrows():
        if str(idx) in results:
            logger.info(f"Sample {idx} already attacked. Skipping.")
            continue

        bb.run(idx, row, detector_name, detector, query_budget=100)
