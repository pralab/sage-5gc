from datetime import datetime, timedelta
import random


def exp_sec(mean: float) -> int:
    """Generates an exponential random variable in seconds with given mean."""
    return max(0, int(random.expovariate(1.0 / mean)))


def gen_pfcp_times_exp(
    recovery_time_stamp: datetime,
    mean_start_delay_sec: int = 30 * 60,  # tipicamente entro 30 min dal recovery
    mean_session_duration_sec: int = 20 * 60,  # durata media 20 min
    mean_end_delay_sec: int = 30,  # end_time poco dopo l'ultimo pkt (~30s)
):
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


recovery = datetime(2025, 1, 1, 10, 0, 0)
times = gen_pfcp_times_exp(recovery)
for k, v in times.items():
    print(k, "=>", v)
