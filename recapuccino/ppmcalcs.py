from numpy.typing import NDArray


def calculate_ppm_deltas(mz: NDArray, mz_target: NDArray) -> NDArray:
    """
    generate ppm differences between, e.g., experimental and database calculated mz values
    :param mz: experimental mz values
    :param mz_target: mz values from db search
    :return:
    """
    return ((mz_target - mz) / mz) * 1e6


def calculate_mz_calibrated(ppm_delta_predicted, mz_exp):
    """
    apply calibration ppm deltas to experimental mz to get calibrated mz values
    :param ppm_delta_predicted: ppm correction factor that should be applied to mz values
    :param mz_exp: experimental mz values that need to be re-calibrated
    :return:
    """
    return 1e-6 * mz_exp * ppm_delta_predicted + mz_exp
