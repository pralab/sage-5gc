"""Black-box feature-space attack for network traffic classifiers."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict

import joblib
import nevergrad as ng
from nevergrad.optimization.base import ConfiguredOptimizer, Optimizer
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from ml_models import Detector

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ATTACK_TYPE_MAP = {
    0: "flooding",  # PFCP Flooding
    1: "session_deletion",  # PFCP Deletion
    2: "session_modification",  # PFCP Modification
    5: "upf_pdn0_fault",  # UPF PDN-0 Fault
    6: "restoration_teid",  # PFCP Restoration-TEID
}

# These features cannot be modified during the attack, as they are
# essential for preserving the original intent of the malicious sample.
ATTACK_FEATURES = {
    "flooding": ["pfcp.msg_type"],
    "session_deletion": [
        "pfcp.msg_type",
    ],
    "session_modification": [
        "pfcp.msg_type",
        "pfcp.seid",
    ],
    "upf_pdn0_fault": [
        "pfcp.node_id_ipv4",
        "pfcp.pdr_id",
        "pfcp.f_teid_flags.ch",
        "pfcp.f_teid_flags.ch_id",
        "pfcp.f_teid_flags.v6",
    ],
    "restoration_teid": [
        "pfcp.f_teid.teid",
        "pfcp.pdr_id",
    ],
}

FEAT_MAPPING = {
    # IP layer
    "ip.ttl": ng.p.Scalar(lower=2, upper=200).set_integer_casting(),
    "ip.id": ng.p.Scalar(lower=0, upper=65534).set_integer_casting(),
    "ip.len": ng.p.Scalar(lower=44, upper=653).set_integer_casting(),
    "ip.checksum": ng.p.Scalar(lower=54, upper=65525).set_integer_casting(),
    # UDP layer
    "udp.checksum": ng.p.Scalar(lower=2389, upper=41240).set_integer_casting(),
    # PFCP layer
    "pfcp.apply_action.buff": ng.p.Choice([True, False]),
    "pfcp.apply_action.forw": ng.p.Choice([True, False]),
    "pfcp.apply_action.nocp": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.ch": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.ch_id": ng.p.Choice([True, False]),
    "pfcp.f_teid_flags.v6": ng.p.Choice([True, False]),
    "pfcp.s": ng.p.Choice([True, False]),
    "pfcp.dst_interface": ng.p.Scalar(lower=0, upper=1).set_integer_casting(),
    "pfcp.duration_measurement": ng.p.Scalar(
        lower=1747212643, upper=1753894838
    ),
    "pfcp.ie_type": ng.p.Scalar(lower=10, upper=96).set_integer_casting(),
    "pfcp.msg_type": ng.p.Scalar(lower=1, upper=57).set_integer_casting(),
    "pfcp.pdr_id": ng.p.Scalar(lower=1, upper=2).set_integer_casting(),
    "pfcp.response_time": ng.p.Scalar(lower=2.0095e-05, upper=0.041239073),
    "pfcp.response_to": ng.p.Scalar(lower=1, upper=2565).set_integer_casting(),
    "pfcp.seid": ng.p.Scalar(lower=0, upper=4095).set_integer_casting(),
    "pfcp.seqno": ng.p.Scalar(lower=0, upper=202364).set_integer_casting(),
    "pfcp.f_teid.teid": ng.p.Scalar(
        lower=29, upper=65507
    ).set_integer_casting(),
    "pfcp.outer_hdr_creation.teid": ng.p.Scalar(
        lower=1, upper=6326
    ).set_integer_casting(),
    "pfcp.flags": ng.p.Scalar(lower=32, upper=33).set_integer_casting(),
    "pfcp.volume_measurement.dlnop": ng.p.Scalar(
        lower=0, upper=13195
    ).set_integer_casting(),
    "pfcp.volume_measurement.dlvol": ng.p.Scalar(
        lower=0, upper=17834134
    ).set_integer_casting(),
    "pfcp.volume_measurement.tonop": ng.p.Scalar(
        lower=0, upper=13195
    ).set_integer_casting(),
    "pfcp.volume_measurement.tovol": ng.p.Scalar(
        lower=0, upper=17834134
    ).set_integer_casting(),
    "pfcp.node_id_ipv4": ng.p.Choice(
        [
            "192.168.130.144",
            "192.168.14.164",
            "192.168.14.153",
            "192.168.14.129",
            "192.168.14.150",
            "192.168.14.176",
            "192.168.130.176",
        ]
    ),
    "pfcp.f_seid.ipv4": ng.p.Choice(
        [
            "192.168.14.155",
            "192.168.130.144",
            "192.168.14.164",
            "192.168.14.153",
            "192.168.14.129",
            "192.168.14.150",
            "192.168.14.176",
            "192.168.130.176",
        ]
    ),
    "pfcp.f_teid.ipv4_addr": ng.p.Choice(
        [
            "192.168.130.144",
            "192.168.14.153",
            "192.168.14.150",
            "192.168.130.176",
            "192.168.14.162",
            "192.168.130.179",
        ]
    ),
    "pfcp.ue_ip_addr_ipv4": ng.p.Choice(
        [
            "10.45.5.12",
            "10.45.6.10",
            "10.45.1.90",
            "10.45.3.174",
            "10.45.4.226",
            "10.45.4.107",
            "10.45.4.97",
            "10.45.5.82",
            "10.45.4.70",
            "10.45.5.56",
            "10.45.4.100",
            "10.45.5.194",
            "10.45.3.38",
            "10.45.1.26",
            "10.45.3.182",
            "10.45.3.132",
            "10.45.4.135",
            "10.45.2.54",
            "10.45.3.36",
            "10.45.4.48",
            "10.45.4.93",
            "10.45.4.155",
            "10.45.3.214",
            "10.45.4.35",
            "10.45.3.0",
            "10.45.4.60",
            "10.45.4.223",
            "10.45.3.242",
            "10.45.5.67",
            "10.45.6.157",
        ]
    ),
    "pfcp.outer_hdr_creation.ipv4": ng.p.Choice(
        [
            "192.168.14.155",
            "192.168.130.178",
            "192.168.14.164",
            "192.168.130.138",
            "192.168.14.129",
            "192.168.130.139",
            "192.168.14.176",
            "192.168.130.186",
            "192.168.130.182",
            "192.168.130.179",
            "192.168.130.181",
        ]
    ),
}


class BlackBoxAttack:
    """Black-box attack for network traffic classifiers."""

    def __init__(self, optimizer_cls: ConfiguredOptimizer) -> None:
        """
        Create a BlackBoxAttack instance.

        Parameters
        ----------
        optimizer_cls : ng.optimization.Optimizer
            The Nevergrad optimizer class to use for the attack.
        """
        self._optimizer_cls: ConfiguredOptimizer = optimizer_cls
        self._optimizer: Optimizer = None
        self._query_budget: int = None

        random.seed(42)

    def run(
        self,
        sample_idx: int,
        sample: pd.Series,
        attack_type: int,
        detector: Detector,
        results_path: Path | str,
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
        attack_type : int
            The type of attack of the sample.
        detector : Detector
            The detector object with a get_score method.
            The maximum number of queries allowed for the attack.
        results_path : Path | str
            The path to save the attack results.
        query_budget : int
            The maximum number of queries allowed for the attack.
        """
        self._query_budget = query_budget

        logger.info(f"Starting attack on sample {sample_idx}...")

        # --------------------------------------
        # [Step 1] Compute baseline predictions
        # --------------------------------------
        orig_score = detector.decision_function(pd.DataFrame([sample]))[0]
        logger.info(f"Original score: {orig_score}")

        if hasattr(detector, "_detector"):
            logger.info(f"Detector threshold: {detector._detector.threshold_}")
            if orig_score < detector._detector.threshold_:
                logger.info(
                    "Sample is already classified as benign. Skipping attack."
                )
                return
        else:
            y_pred = detector.predict(pd.DataFrame([sample]))
            if y_pred == 0:
                logger.info(
                    "Sample is already classified as benign. Skipping attack."
                )
                return

        # --------------------------
        # [Step 2] Set up optimizer
        # --------------------------
        self._optimizer: Optimizer = self._init_optimizer(attack_type)

        # --------------------
        # [Step 3] Run attack
        # --------------------
        for idx in range(self._optimizer.budget):  # type: ignore
            x = self._optimizer.ask()
            x_adv = self._apply_modifications(sample, x.value)
            loss = self._compute_loss(x_adv, detector)
            logger.info(
                f"Iteration {idx + 1}/{self._query_budget}: loss = {loss}"
            )

            if hasattr(detector, "_detector"):
                if loss < detector._detector.threshold_:
                    logger.info(
                        f"Sample evaded the detector after {idx + 1} queries."
                    )
                    break
            else:
                y_pred = detector.predict(pd.DataFrame([x_adv]))
                if y_pred == 0:
                    logger.info(
                        f"Sample evaded the detector after {idx + 1} queries."
                    )
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
            attack_type,
            best_params,
            best_loss,
            results_path,
            bool(loss < detector._detector.threshold_)
            if hasattr(detector, "_detector")
            else bool(y_pred == 0),
        )

    def _init_optimizer(self, attack_type: int) -> Optimizer:
        params = {}
        for feature, parametrization in FEAT_MAPPING.items():
            # Skip features that should not be modified to preserve the attack intent
            if feature not in ATTACK_FEATURES[ATTACK_TYPE_MAP[attack_type]]:
                params[feature] = parametrization

        params = ng.p.Dict(**params)
        params.random_state = np.random.RandomState(42)
        return self._optimizer_cls(
            parametrization=params, budget=self._query_budget
        )

    def _save_results(
        self,
        sample_idx: int,
        attack_type: int,
        best_params: dict,
        best_loss: float,
        results_path: Path | str,
        evaded: bool,
    ) -> None:
        if Path(results_path).exists():
            with Path(results_path).open("r") as f:
                results = json.load(f)
        else:
            results = {}

        results[str(sample_idx)] = {
            "attack_type": attack_type,
            "best_params": best_params,
            "best_loss": best_loss,
            "evaded": bool(evaded),
        }

        with Path(results_path).open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _compute_loss(self, x_adv: pd.Series, detector: Detector) -> float:
        return detector.decision_function(pd.DataFrame([x_adv]))[0]

    def _apply_modifications(
        self, sample: pd.Series, params: Dict[str, int]
    ) -> pd.Series:
        adv_sample = sample.copy()
        for feature, value in params.items():
            if feature in [
                "ip.checksum",
                "ip.id",
                "udp.checksum",
                "pfcp.seid",
                "pfcp.f_teid.teid",
                "pfcp.outer_hdr_creation.teidpfcp.flags",
            ]:
                adv_sample[feature] = hex(value)
            else:
                adv_sample[feature] = value

        return adv_sample


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Black-box attack on network traffic classifiers."
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        choices=[
            "ABOD",
            "CBLOF",
            "COPOD",
            "ECOD",
            "Ensemble_SVC_C10_G10_HBOS_KNN_ABOD_INNE_PCA",
            "Ensemble_SVC_C10_G10_HBOS_KNN_GMM_INNE_PCA",
            "Ensemble_SVC_C10_G10_HBOS_KNN_LOF_INNE_PCA",
            "Ensemble_SVC_C100_G100_HBOS_KNN_LOF_INNE_FeatureBagging",
            "FeatureBagging",
            "GMM",
            "HBOS",
            "IForest",
            "INNE",
            "KNN",
            "LODA",
            "LOF",
            "PCA",
        ],
        required=True,
        help="The name of the trained model to attack",
    )
    argparser.add_argument(
        "--ds-path",
        type=str,
        default=None,
        help="The path to the attacks dataset file",
        required=True,
    )
    argparser.add_argument(
        "--optmizer",
        type=str,
        choices=["ES", "DE"],
        default="ES",
        help="The Nevergrad optimizer to use for the attack",
    )
    args = argparser.parse_args()

    dataset = pd.read_csv(args.ds_path, sep=";", low_memory=False)
    labels = dataset["ip.opt.time_stamp"].copy()
    dataset = dataset.drop(columns=["ip.opt.time_stamp"])

    if args.optmizer == "ES":
        optimizer_cls = ng.optimizers.EvolutionStrategy(
            recombination_ratio=0.9,
            popsize=20,
            only_offsprings=False,
            offsprings=20,
            ranker="simple",
        )
    else:
        optimizer_cls = ng.optimizers.DifferentialEvolution(
            popsize=20,
            crossover="twopoints",
            propagate_heritage=True,
        )

    bb = BlackBoxAttack(optimizer_cls)

    detector: Detector = joblib.load(
        Path(__file__).parent.parent
        / f"data/trained_models/with_scaler/{args.model_name}.pkl"
    )

    results_path = (
        Path(__file__).parent.parent
        / f"results/with_scaler/blackbox_attack/{optimizer_cls.__class__.__name__.lower()}"
        / f"{args.model_name.lower()}.json"
    )
    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in dataset.iterrows():
        if str(idx) in results:
            logger.info(f"Sample {idx} already attacked. Skipping.")
            continue

        bb.run(
            idx,  # type: ignore
            row,
            int(labels.iloc[idx]),  # type: ignore
            detector,
            results_path,
            query_budget=100,
        )
