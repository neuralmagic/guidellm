from typing import List, Tuple
import numpy as np

def linear_interpolate(target: float, lower: Tuple[float, float], upper: Tuple[float, float]) -> float:
    """
    Linearly interpolates a value at 'target' given two points.
    If the target equals one of the bounds, the corresponding value is returned.
    """
    lower_ref, lower_measurement = lower
    upper_ref, upper_measurement = upper

    if upper_ref == lower_ref:
        return lower_measurement
    if target <= lower_ref:
        return lower_measurement
    if target >= upper_ref:
        return upper_measurement

    t = (target - lower_ref) / (upper_ref - lower_ref)
    return lower_measurement + t * (upper_measurement - lower_measurement)

def stretch_list(arr: List[float], target_length: int):
    if len(arr) == target_length:
        return np.array(arr)
    
    original_x = np.linspace(0, 1, len(arr))
    target_x = np.linspace(0, 1, target_length)
    stretched_arr = list(np.interp(target_x, original_x, arr))
    return stretched_arr

def interpolate_measurements(target: float, lower_ref_measurements_pair: Tuple[float, List[float]], upper_ref_measurements_pair: Tuple[float, List[float]]) -> List[float]:
    """
    Interpolates each corresponding measurement value between lower and upper benchmarks.
    Assumes that lower_measurements and upper_measurements have the same length.
    """
    lower_ref, lower_measurements = lower_ref_measurements_pair
    upper_ref, upper_measurements = upper_ref_measurements_pair

    if len(lower_measurements) < len(upper_measurements):
        lower_measurements = stretch_list(lower_measurements, len(upper_measurements))
    if len(lower_measurements) > len(upper_measurements):
        upper_measurements = stretch_list(upper_measurements, len(lower_measurements))

    return [
        linear_interpolate(target, (lower_ref, lower_measurements[i]), (upper_ref, upper_measurements[i]))
        for i in range(len(lower_measurements))
    ]

def interpolate_data_points(data_points: List[Tuple[float, List[float]]],
                            target_ref: List[float]) -> List[Tuple[float, List[float]]]:
    """
    Given sorted data_points as tuples of (scalar, measurements) and a list of target scalar values,
    interpolate the measurements for each target.
    
    The data_points must be sorted by the scalar value in ascending order.
    Only target scalar values that fall within the min and max of the data_points are considered.
    """
    if not data_points:
        return []

    lower_bound = data_points[0][0]
    upper_bound = data_points[-1][0]
    # Filter target_ref to only include values within the provided range.
    valid_targets = [t for t in target_ref if lower_bound <= t <= upper_bound]

    interpolated_results = []
    # Pointer to the current lower data point index.
    lower_idx = 0

    for target in sorted(valid_targets):
        # Advance the lower_idx until we find the correct interval.
        while (lower_idx < len(data_points) - 1 and target > data_points[lower_idx + 1][0]):
            lower_idx += 1

        # If the target exactly matches a known scalar value, use its measurements.
        if target == data_points[lower_idx][0]:
            interpolated_results.append((target, data_points[lower_idx][1][:]))
        # Otherwise, if target lies between two data points, interpolate.
        elif lower_idx < len(data_points) - 1:
            lower_ref, lower_measurements = data_points[lower_idx]
            upper_ref, upper_measurements = data_points[lower_idx + 1]
            interpolated = interpolate_measurements(target, (lower_ref, lower_measurements),
                                                    (upper_ref, upper_measurements))
            interpolated_results.append((target, interpolated))
        else:
            # If for some reason target is above the highest known data point, ignore it.
            continue

    return interpolated_results