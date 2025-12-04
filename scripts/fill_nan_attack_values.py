'''import pandas as pd
import numpy as np
import random
from pathlib import Path
import logging
from faker import Faker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


fake = Faker()

"""
pfcp.node_id_ipv4                       OK
pfcp.f_seid.ipv4                        OK
pfcp.f_teid.ipv4_addr                   OK
pfcp.f_teid.teid                        OK
pfcp.f_teid_flags.ch                    OK
pfcp.f_teid_flags.ch_id                 OK
pfcp.f_teid_flags.v6                    OK
pfcp.outer_hdr_creation.ipv4            OK
pfcp.outer_hdr_creation.teid            OK
pfcp.dst_interface                      OK
pfcp.pdr_id                             OK
pfcp.apply_action.buff                  OK
pfcp.apply_action.forw                  OK
pfcp.apply_action.nocp                  OK
pfcp.ue_ip_addr_ipv4                    OK
pfcp.duration_measurement               OK
pfcp.volume_measurement.dlnop           OK
pfcp.volume_measurement.dlvol           OK
pfcp.volume_measurement.tonop           OK
pfcp.volume_measurement.tovol           OK
pfcp.ie_type                            OK
pfcp.ie_len                             OK
pfcp.response_time                      OK
pfcp.response_to                        OK
"""

import random

def generate_smart_attacker_row():
    row = {}

    # --- 1. TOPOLOGIA CREDIBILE (Subnet Private) ---
    # Generiamo l'indirizzo IPv4 per pfcp.node_id_ipv4 e pfcp.ue_ip_addr_ipv4
    row['pfcp.node_id_ipv4'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    row['pfcp.ue_ip_addr_ipv4'] = f"10.45.{random.randint(0, 16)}.{random.randint(2, 254)}"  # CGNAT Range

    # --- 2. TUNNELING COERENTE ---
    # Generiamo il TEID e F-SEID IP
    row['pfcp.f_teid.teid'] = hex(random.randint(29, 65507))  # TEID tra 29 e 65507
    row['pfcp.pdr_id'] = random.choice([1, 2])  # ID basso

    row['pfcp.f_seid.ipv4'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    row['pfcp.f_teid.ipv4_addr'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"

    # --- 3. FLAGS LOGICI ---
    # Se IPv4, 'f_teid_flags.v6' è False
    row['pfcp.f_teid_flags.v6'] = False  # IPv6 è False dato che usiamo IPv4
    row['pfcp.f_teid_flags.ch'] = random.choice([True, False])  # CH è raro, probabilmente False
    row['pfcp.f_teid_flags.ch_id'] = random.choice([True, False])  # Rare flag

    # --- 4. AZIONI ---
    # Simuliamo che un attacco DoS disabiliti forwarding
    attack_type = random.choice(['DoS_Flood', 'DoS_Deletion', 'DoS_Modification', 'DoS_Fault', 'DoS_Restoration'])

    if attack_type in ['DoS_Flood', 'DoS_Deletion', 'DoS_Modification', 'DoS_Fault', 'DoS_Restoration']:
        row['pfcp.apply_action.forw'] = False  # Disabilita forwarding per DoS
        row['pfcp.apply_action.buff'] = random.choice([True, False])  # Potrebbe esserci buffering
        row['pfcp.apply_action.nocp'] = random.choice(
            [True, False])  # Potrebbe esserci manipolazione senza control plane
    else:
        # Comportamento normale per altri tipi di traffico
        action_choice = random.choice([0, 1, 2])  # 0 = forw, 1 = buff, 2 = nocp
        if action_choice == 0:
            row['pfcp.apply_action.forw'] = True
            row['pfcp.apply_action.buff'] = False
            row['pfcp.apply_action.nocp'] = False
        elif action_choice == 1:
            row['pfcp.apply_action.forw'] = False
            row['pfcp.apply_action.buff'] = True
            row['pfcp.apply_action.nocp'] = False
        else:
            row['pfcp.apply_action.forw'] = False
            row['pfcp.apply_action.buff'] = False
            row['pfcp.apply_action.nocp'] = True

    # --- 5. METRICHE VEROSIMILI ---
    simulated_packets = random.randint(0, 13195)
    avg_pkt_size = random.randint(500, 1200)  # Byte
    row['pfcp.volume_measurement.tonop'] = simulated_packets
    row['pfcp.volume_measurement.tovol'] = simulated_packets * avg_pkt_size

    dl_ratio = random.uniform(0.7, 0.9)  # Downlink maggiore
    row['pfcp.volume_measurement.dlnop'] = int(simulated_packets * dl_ratio)
    row['pfcp.volume_measurement.dlvol'] = int(row['pfcp.volume_measurement.tovol'] * dl_ratio)

    # --- 6. DURATA E TEMPI ---
    row['pfcp.duration_measurement'] = random.uniform(1747212643.0, 1753894838.0)  # Secondi
    row['pfcp.time_of_first_packet'] = random.randint(1747212464, 1753894823)  # Timestamp in secondi
    row['pfcp.time_of_last_packet'] = random.randint(1747212640, 1753894834)  # Timestamp in secondi

    # --- 7. PARAMETRI TECNICI ---
    row['pfcp.ie_type'] = random.randint(10, 96)  # Tipo di IE tra 10 e 96
    row['pfcp.ie_len'] = random.randint(1, 50)  # Lunghezza dell'IE tra 1 e 50
    row['pfcp.response_time'] = random.uniform(2.0095e-05, 0.041239073)  # Tempo di risposta tra 2.0095e-05 e 0.041239073
    row['pfcp.response_to'] = random.randint(1, 2565)  # Tempo di risposta tra 1 e 2565

    return row
'''

import pandas as pd
import numpy as np
import random
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# MAPPING LABEL → ATTACK TYPE
# ============================================================================

ATTACK_TYPE_MAP = {
    0.0: 'DoS_Flood',  # PFCP Flooding
    1.0: 'DoS_Deletion',  # PFCP Deletion
    2.0: 'DoS_Modification',  # PFCP Modification
    3.0: 'Reconnaissance',  # NMAP Scan
    4.0: 'Lateral_Movement',  # Reverse Shell
    5.0: 'DoS_Fault',  # UPF PDN-0 Fault
    6.0: 'DoS_Restoration',  # PFCP Restoration-TEID
}

# ============================================================================
# GENERATORE DI VALORI PFCP REALISTICI (5G Core)
# ============================================================================

def generate_smart_attacker_row(attack_type: str = None):
    """
    Genera sample PFCP realistico per 5G Core Network.

    Parameters
    ----------
    attack_type : str, optional
        Tipo di attacco (es: 'DoS_Flood', 'Reconnaissance').
        Se None, viene scelto random.

    Returns
    -------
    dict
        Dizionario con valori PFCP realistici.
    """
    row = {}

    # Se attack_type non specificato, scegline uno random
    if attack_type is None:
        attack_type = random.choice(list(ATTACK_TYPE_MAP.values()))

    # Topologia 5G
    row['pfcp.node_id_ipv4'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    row['pfcp.ue_ip_addr_ipv4'] = f"10.45.{random.randint(0, 16)}.{random.randint(2, 254)}"

    # Tunneling GTP-U
    row['pfcp.f_teid.teid'] = hex(random.randint(29, 65507))
    row['pfcp.pdr_id'] = float(random.choice([1, 2]))

    row['pfcp.f_seid.ipv4'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    row['pfcp.f_teid.ipv4_addr'] = f"192.168.{random.randint(14, 130)}.{random.randint(2, 254)}"
    row['pfcp.outer_hdr_creation.ipv4'] = f"192.168.{random.randint(14, 130)}.{random.randint(129, 186)}"
    row['pfcp.outer_hdr_creation.teid'] = hex(random.randint(0x1, 0x18B6))

    # F-TEID Flags
    row['pfcp.f_teid_flags.v6'] = False
    row['pfcp.f_teid_flags.ch'] = random.choice([True, False])
    row['pfcp.f_teid_flags.ch_id'] = random.choice([True, False])

    # Apply Actions (varia secondo tipo attacco)
    if attack_type in ['DoS_Flood', 'DoS_Deletion', 'DoS_Modification', 'DoS_Fault', 'DoS_Restoration']:
        row['pfcp.apply_action.forw'] = False
        row['pfcp.apply_action.buff'] = random.choice([True, False])
        row['pfcp.apply_action.nocp'] = random.choice([True, False])
    else:
        action_choice = random.choice([0, 1, 2])
        if action_choice == 0:
            row['pfcp.apply_action.forw'] = True
            row['pfcp.apply_action.buff'] = False
            row['pfcp.apply_action.nocp'] = False
        elif action_choice == 1:
            row['pfcp.apply_action.forw'] = False
            row['pfcp.apply_action.buff'] = True
            row['pfcp.apply_action.nocp'] = False
        else:
            row['pfcp.apply_action.forw'] = False
            row['pfcp.apply_action.buff'] = False
            row['pfcp.apply_action.nocp'] = True

    # Destination Interface
    row['pfcp.dst_interface'] = float(random.choice([0, 1]))

    # Volume Measurements
    simulated_packets = random.randint(0, 13195)
    avg_pkt_size = random.randint(500, 1200)
    dl_ratio = random.uniform(0.7, 0.9)

    row['pfcp.volume_measurement.tonop'] = float(simulated_packets)
    row['pfcp.volume_measurement.tovol'] = float(simulated_packets * avg_pkt_size)
    row['pfcp.volume_measurement.dlnop'] = float(int(simulated_packets * dl_ratio))
    row['pfcp.volume_measurement.dlvol'] = float(int(row['pfcp.volume_measurement.tovol'] * dl_ratio))

    # Timestamps PFCP
    row['pfcp.duration_measurement'] = float(random.uniform(1747212643.0, 1753894838.0))
    row['pfcp.time_of_first_packet'] = float(random.randint(1747212464, 1753894823))
    row['pfcp.time_of_last_packet'] = float(random.randint(1747212640, 1753894834))
    row['pfcp.end_time'] = float(random.randint(1747212642, 1753894838))
    row['pfcp.recovery_time_stamp'] = float(random.randint(1747207882, 1753892199))

    # IE Type/Length
    row['pfcp.ie_type'] = float(random.randint(10, 96))
    row['pfcp.ie_len'] = float(random.randint(1, 50))

    # Response Time
    row['pfcp.response_time'] = random.uniform(2.0095e-05, 0.041239073)
    row['pfcp.response_to'] = float(random.randint(1, 2565))

    return row


# ============================================================================
# PFCP FEATURES DA RIEMPIRE
# ============================================================================

PFCP_FEATURES = [
    'pfcp.node_id_ipv4',
    'pfcp.f_seid.ipv4',
    'pfcp.f_teid.ipv4_addr',
    'pfcp.f_teid.teid',
    'pfcp.f_teid_flags.ch',
    'pfcp.f_teid_flags.ch_id',
    'pfcp.f_teid_flags.v6',
    'pfcp.outer_hdr_creation.ipv4',
    'pfcp.outer_hdr_creation.teid',
    'pfcp.dst_interface',
    'pfcp.pdr_id',
    'pfcp.apply_action.buff',
    'pfcp.apply_action.forw',
    'pfcp.apply_action.nocp',
    'pfcp.ue_ip_addr_ipv4',
    'pfcp.duration_measurement',
    'pfcp.volume_measurement.dlnop',
    'pfcp.volume_measurement.dlvol',
    'pfcp.volume_measurement.tonop',
    'pfcp.volume_measurement.tovol',
    'pfcp.time_of_first_packet',
    'pfcp.time_of_last_packet',
    'pfcp.end_time',
    'pfcp.recovery_time_stamp',
    'pfcp.ie_type',
    'pfcp.ie_len',
    'pfcp.response_time',
    'pfcp.response_to',
]


# ============================================================================
# FUNZIONE DI FILLING PER UN SINGOLO SAMPLE
# ============================================================================

def fill_nan_pfcp_fields(sample: pd.Series, seed: int = None) -> tuple[pd.Series, int, str]:
    """
    Riempie SOLO i campi PFCP NaN di un sample con valori realistici.
    I valori già presenti vengono PRESERVATI.

    Parameters
    ----------
    sample : pd.Series
        Sample malevolo originale.
    seed : int, optional
        Random seed per riproducibilità.

    Returns
    -------
    filled_sample : pd.Series
        Sample con NaN riempiti.
    nan_count : int
        Numero di NaN effettivamente riempiti.
    attack_type : str
        Tipo di attacco estratto dalla label.
    """
    if seed is not None:
        random.seed(seed)

    # Estrai attack type dalla label
    label = sample.get('ip.opt.time_stamp', None)
    if pd.notna(label):
        attack_type = ATTACK_TYPE_MAP.get(float(label), 'Unknown')
    else:
        attack_type = 'Unknown'

    # Genera valori realistici basati su attack type
    generated_values = generate_smart_attacker_row(attack_type)

    # Riempi SOLO i campi NaN (preserva valori esistenti)
    filled_sample = sample.copy()
    nan_count = 0

    for field in PFCP_FEATURES:
        # Controlla se il campo esiste E se è NaN
        if field in filled_sample.index and pd.isna(filled_sample[field]):
            # Riempi SOLO se il valore generato esiste
            if field in generated_values:
                filled_sample[field] = generated_values[field]
                nan_count += 1

    return filled_sample, nan_count, attack_type


# ============================================================================
# ATTACCO COMPLETO SU DATASET
# ============================================================================

def fill_malicious_dataset_nans(
        input_path: str | Path,
        output_path: str | Path,
        label_col: str = "ip.opt.time_stamp",
        seed: int = 42
) -> pd.DataFrame:
    """
    Carica dataset malevolo, riempie NaN PFCP con valori realistici e salva.

    Parameters
    ----------
    input_path : str | Path
        Path al dataset CSV di test malevolo.
    output_path : str | Path
        Path dove salvare il dataset con NaN filled.
    label_col : str
        Nome della colonna label (per filtrare solo malware).
    seed : int
        Random seed base per riproducibilità.

    Returns
    -------
    pd.DataFrame
        Dataset con NaN filled.
    """
    logger.info(f"Loading malicious dataset from: {input_path}")
    df = pd.read_csv(input_path, sep=";", low_memory=False)

    # Filtra solo sample malevoli (label non NaN)
    if label_col in df.columns:
        df_malicious = df[df[label_col].notna()].copy()
        logger.info(f"Found {len(df_malicious)} malicious samples (out of {len(df)} total)")
    else:
        logger.warning(f"Label column '{label_col}' not found!  Processing all samples...")
        df_malicious = df.copy()

    # Statistiche NaN iniziali per PFCP features
    pfcp_cols_present = [col for col in PFCP_FEATURES if col in df_malicious.columns]
    initial_nan_count = df_malicious[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Initial NaN count in PFCP fields: {initial_nan_count}")

    # Statistiche per attack type
    attack_type_stats = {}

    # Riempi NaN per ogni sample
    filled_samples = []
    total_filled = 0

    for idx, (row_idx, sample) in enumerate(df_malicious.iterrows()):
        # Usa seed diverso per ogni sample (riproducibile)
        filled_sample, nan_count, attack_type = fill_nan_pfcp_fields(sample, seed=seed + idx)
        filled_samples.append(filled_sample)
        total_filled += nan_count

        # Aggiorna statistiche per attack type
        if attack_type not in attack_type_stats:
            attack_type_stats[attack_type] = {'samples': 0, 'nans_filled': 0}
        attack_type_stats[attack_type]['samples'] += 1
        attack_type_stats[attack_type]['nans_filled'] += nan_count

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_malicious)} samples...")

    df_filled = pd.DataFrame(filled_samples)

    # Statistiche NaN finali
    final_nan_count = df_filled[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Final NaN count in PFCP fields: {final_nan_count}")
    logger.info(f"Filled {total_filled} NaN values total")

    # Salva dataset con NaN filled
    df_filled.to_csv(output_path, sep=";", index=False)
    logger.info(f"✅ Dataset with filled NaNs saved to: {output_path}")

    # Log statistiche per attack type
    logger.info("\n" + "=" * 60)
    logger.info("FILLING STATISTICS BY ATTACK TYPE")
    logger.info("=" * 60)
    for attack_type, stats in sorted(attack_type_stats.items()):
        logger.info(f"{attack_type:20s} → {stats['samples']:4d} samples, {stats['nans_filled']:5d} NaNs filled")

    return df_filled


def fill_malicious_dataset_nans_and_reassemble(
        input_path: str | Path,
        output_path: str | Path,
        label_col: str = "ip.opt.time_stamp",
        seed: int = 42
) -> pd.DataFrame:
    """
    Riempie NaN nei sample malevoli e riassembla con i benign originali.
    """
    logger.info(f"Loading mixed dataset from: {input_path}")
    df = pd.read_csv(input_path, sep=";", low_memory=False)

    # Separa benign e malicious
    if label_col in df.columns:
        df_benign = df[df[label_col].isna()].copy()
        df_malicious = df[df[label_col].notna()].copy()
        logger.info(f"Found {len(df_benign)} benign + {len(df_malicious)} malicious samples")
    else:
        logger.warning(f"Label column '{label_col}' not found!  Processing all samples...")
        df_benign = pd.DataFrame()
        df_malicious = df.copy()

    # Statistiche NaN iniziali
    pfcp_cols_present = [col for col in PFCP_FEATURES if col in df_malicious.columns]
    initial_nan_count = df_malicious[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Initial NaN count in malicious PFCP fields: {initial_nan_count}")

    # Riempi NaN per ogni sample malevolo
    filled_samples = []
    total_filled = 0
    attack_type_stats = {}

    for idx, (row_idx, sample) in enumerate(df_malicious.iterrows()):
        filled_sample, nan_count, attack_type = fill_nan_pfcp_fields(sample, seed=seed + idx)
        filled_samples.append(filled_sample)
        total_filled += nan_count

        if attack_type not in attack_type_stats:
            attack_type_stats[attack_type] = {'samples': 0, 'nans_filled': 0}
        attack_type_stats[attack_type]['samples'] += 1
        attack_type_stats[attack_type]['nans_filled'] += nan_count

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_malicious)} malicious samples...")

    df_malicious_filled = pd.DataFrame(filled_samples)

    # ✅ RIASSEMBLA: benign originali + malicious filled
    df_final = pd.concat([df_benign, df_malicious_filled], axis=0, ignore_index=True)

    # Statistiche finali
    final_nan_count = df_malicious_filled[pfcp_cols_present].isna().sum().sum()
    logger.info(f"Final NaN count in malicious PFCP fields: {final_nan_count}")
    logger.info(f"Filled {total_filled} NaN values total")

    # Salva dataset completo riassemblato
    df_final.to_csv(output_path, sep=";", index=False)
    logger.info(f"✅ Full dataset (benign + malicious filled) saved to: {output_path}")

    # Log statistiche per attack type
    logger.info("\n" + "=" * 60)
    logger.info("FILLING STATISTICS BY ATTACK TYPE")
    logger.info("=" * 60)
    for attack_type, stats in sorted(attack_type_stats.items()):
        logger.info(f"{attack_type:20s} → {stats['samples']:4d} samples, {stats['nans_filled']:5d} NaNs filled")

    return df_final


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Configurazione paths
    INPUT_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset.csv"
    OUTPUT_PATH = Path(__file__).parent.parent / "data/datasets/test_dataset_filled.csv"

    # Esegui filling
    df_filled = fill_malicious_dataset_nans_and_reassemble(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        label_col="ip.opt.time_stamp",
        seed=42
    )

    # Statistiche finali
    logger.info("\n" + "=" * 60)
    logger.info("FILLING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples with filled NaNs: {len(df_filled)}")

    # Mostra distribuzione labels
    if "ip.opt.time_stamp" in df_filled.columns:
        label_dist = df_filled["ip.opt.time_stamp"].value_counts().sort_index()
        logger.info("\nLabel distribution:")
        for label, count in label_dist.items():
            attack_name = ATTACK_TYPE_MAP.get(float(label), "Unknown")
            logger.info(f"  Label {label} ({attack_name:20s}): {count} samples")

    # Mostra shape finale
    logger.info(f"\nFinal dataset shape: {df_filled.shape}")

    # Verifica NaN residui
    pfcp_cols = [col for col in PFCP_FEATURES if col in df_filled.columns]
    remaining_nans = df_filled[pfcp_cols].isna().sum().sum()
    if remaining_nans == 0:
        logger.info("✅ All PFCP NaN values successfully filled!")
    else:
        logger.warning(f"⚠️ {remaining_nans} PFCP NaN values still remain")