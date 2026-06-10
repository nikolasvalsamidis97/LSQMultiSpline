import matplotlib.pyplot as plt
import numpy as np

from lsq_multivariate_spline import LSQMultivariateSpline


def test_lsq_multivariate_spline_1d():
    """Plot noisy 1D data before fitting a spline."""
    rng = np.random.default_rng(0)
    x0 = np.sort(rng.uniform(0, 8 * np.pi, 100))
    signal = np.cos(x0) + np.sin(x0 / 2) + np.cos(x0 / 3)
    noise = 0.1 * rng.normal(size=signal.shape)
    y_noisy = signal + noise

    interior_knots = np.linspace(x0.min(), x0.max(), 14)[1:-1]
    spline = LSQMultivariateSpline(
        x=x0,
        y=y_noisy,
        t=interior_knots,
        w=None,
        bbox=None,
        k=3,
        eps=1e-6,
    )

    x_fit = np.linspace(x0.min(), x0.max(), 600)
    y_fit = spline(x_fit)
    y_fit_train = spline(x0)

    residual_noisy = y_noisy - y_fit_train
    mae_noisy = np.mean(np.abs(residual_noisy))
    rmse_noisy = np.sqrt(np.mean(residual_noisy**2))

    fig, ax = plt.subplots()
    ax.scatter(x0, y_noisy, s=18, label="Noisy samples")
    ax.plot(x_fit, y_fit, color="tab:red", linewidth=2.0, label="Spline fit")
    ax.set_title("Noisy 1D samples")
    ax.set_xlabel("x0")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    plt.show()

    print()
    print("1D LSQMultivariateSpline Fit Report")
    print("=" * 37)
    print(f"{'Spline degree':<28} {3:>12d}")
    print(f"{'Interior knots':<28} {interior_knots.size:>12d}")
    print(f"{'Number of coefficients':<28} {spline.get_coeffs().shape[0]:>12d}")
    print()
    print("Training Fit")
    print("-" * 37)
    print(f"{'Weighted residual':<28} {spline.get_residual():>12.6g}")
    print(f"{'MAE vs noisy samples':<28} {mae_noisy:>12.6g}")
    print(f"{'RMSE vs noisy samples':<28} {rmse_noisy:>12.6g}")


if __name__ == "__main__":
    test_lsq_multivariate_spline_1d()
