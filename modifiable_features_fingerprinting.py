# ============================================================================
# MODIFIABLE FEATURE (from tables IP/TCP/UDP/PFCP)
# ============================================================================
MODIFIABLE_FEATURES = {
    # === Core IP Headers ===
    'ip.ttl',                # Time To Live ok
    'ip.id',                 # Fragment identification
    'ip.flags.df',           # Don't Fragment flag
    'ip.dsfield.dscp',       # DSCP (QoS marking)

    # === UDP Core ===
    'udp.srcport',  # UDP source port

    # === TCP Core ===
    'tcp.srcport',           # Source port
    'tcp.seq_raw',           # Sequence number (raw)
    'tcp.ack_raw',           # Acknowledgment number (raw)
    'tcp.flags.syn',         # SYN flag
    'tcp.flags.ack',         # ACK flag
    'tcp.flags.reset',         # RST flag
    'tcp.flags.fin',         # FIN flag
    'tcp.flags.push',         # PSH flag
    'tcp.options.timestamp.tsval',  # Timestamp Value
    'tcp.options.timestamp.tsecr',  # Timestamp Echo Reply
    'tcp.window_size_value',  # Window size

    # === PFCP Accounting ===
    'pfcp.duration_measurement',      # Duration measurement
    'pfcp.time_of_first_packet',      # Time of first packet
    'pfcp.time_of_last_packet',       # Time of last packet
    'pfcp.end_time',                  # End time
    'pfcp.recovery_time_stamp',       # Recovery timestamp
    'pfcp.volume_measurement.dlnop',  # Downlink volume (no payload)
    'pfcp.volume_measurement.dlvol',  # Downlink volume
    'pfcp.volume_measurement.tonop',  # Total volume (no payload)
    'pfcp.volume_measurement.tovol',  # Total volume
    'pfcp.user_id.imei',              # IMEI
    'pfcp.ue_ip_address_flag.sd',     # UE IP address flag
    'pfcp.f_teid_flags.ch',           # F-TEID flags
    'pfcp.f_teid_flags.ch_id',        # F-TEID channel ID
    'pfcp.f_teid_flags.v6',           # F-TEID IPv6 flag
}

# ============================================================================
# FEATURE FAMILIES
# ============================================================================

MODIFIABLE_FEATURE_FAMILIES = {
    'ip_header': [
        'ip.ttl',  # Time To Live ok
        'ip.id',  # Fragment identification
        'ip.flags.df',  # Don't Fragment flag
        'ip.dsfield.dscp',  # DSCP (QoS marking)
    ],
    'tcp_ports': [
        'tcp.srcport',
    ],
    'tcp_sequence': [
        'tcp.seq_raw',
        'tcp.ack_raw',
    ],
    'tcp_flags': [
        'tcp.flags.syn',
        'tcp.flags.ack',
        'tcp.flags.rst',
        'tcp.flags.fin',
        'tcp.flags.push',
    ],
    'tcp_options': [
        'tcp.option_kind',
        'tcp.option_len',
        'tcp.options.timestamp.tsval',
        'tcp.options.timestamp.tsecr',
    ],
    'tcp_window': [
        'tcp.window_size_value',
    ],
    'udp_ports': [
        'udp.srcport',
    ],
    'pfcp_timing': [
        'pfcp.duration_measurement',
        'pfcp.time_of_first_packet',
        'pfcp.time_of_last_packet',
        'pfcp.end_time',
        'pfcp.recovery_time_stamp',
    ],
    'pfcp_volume': [
        'pfcp.volume_measurement.dlnop',
        'pfcp.volume_measurement.dlvol',
        'pfcp.volume_measurement.tonop',
        'pfcp.volume_measurement.tovol',
    ],
    'pfcp_user': [
        'pfcp.user_id.imei',
        'pfcp.ue_ip_address_flag.sd',
        'pfcp.f_teid_flags.ch',
        'pfcp.f_teid_flags.ch_id',
        'pfcp.f_teid_flags.v6',
    ],
}


def is_modifiable(feature: str) -> bool:
    """Checks if a feature is editable."""
    if feature not in MODIFIABLE_FEATURES:
        print(f"Feature not found: {feature}")
    return feature in MODIFIABLE_FEATURES


def get_modifiable_families() -> dict:
    """Returns the editable feature families."""
    return MODIFIABLE_FEATURE_FAMILIES.copy()


def filter_modifiable_features(feature_list: list[str]) -> list[str]:
    """Filter keeping only editable features."""
    return [f for f in feature_list if is_modifiable(f)]


def get_all_modifiable_features() -> set:
    """Returns complete set of editable features."""

    return MODIFIABLE_FEATURES.copy()
