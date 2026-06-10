from lsq_multivariate_spline import LSQMultivariateSpline


def test_lsq_multivariate_spline_2d():
    """Plot noisy 2D data and a fitted spline surface."""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-2.0, 2.0, 220)
    x1 = rng.uniform(-2.0, 2.0, 220)
    signal = 0.55 * (x0**2 - x1**2) + 0.15 * x0 * x1
    noise = 0.05 * rng.normal(size=signal.shape)
    y_noisy = signal + noise
    x = np.column_stack((x0, x1))

    interior_knots = [
        np.linspace(x0.min(), x0.max(), 6)[1:-1],
        np.linspace(x1.min(), x1.max(), 6)[1:-1],
    ]
    spline = LSQMultivariateSpline(
        x=x,
        y=y_noisy,
        t=interior_knots,
        w=None,
        bbox=None,
        k=3,
        eps=1e-6,
    )

    x0_grid = np.linspace(x0.min(), x0.max(), 45)
    x1_grid = np.linspace(x1.min(), x1.max(), 45)
    x0_mesh, x1_mesh = np.meshgrid(x0_grid, x1_grid, indexing="ij")
    x_eval = np.column_stack((x0_mesh.ravel(), x1_mesh.ravel()))
    y_fit_mesh = spline(x_eval).reshape(x0_mesh.shape)

    y_fit_train = spline(x)
    residual_noisy = y_noisy - y_fit_train
    mae_noisy = np.mean(np.abs(residual_noisy))
    rmse_noisy = np.sqrt(np.mean(residual_noisy**2))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(
        x0_mesh,
        x1_mesh,
        y_fit_mesh,
        cmap="viridis",
        edgecolor="none",
        alpha=0.65,
    )
    scatter = ax.scatter(
        x0,
        x1,
        y_noisy,
        c=y_noisy,
        cmap="viridis",
        s=12,
    )
    ax.set_title("Noisy saddle samples with fitted spline surface")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("y")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="y_noisy")
    fig.tight_layout()
    plt.show()

    print()
    print("2D LSQMultivariateSpline Fit Report")
    print("=" * 37)
    print("Function: saddle, z = 0.55*(x0**2 - x1**2) + 0.15*x0*x1")
    print()
    print(f"{'Spline degrees':<28} {str([3, 3]):>12}")
    print(f"{'Interior knots x0':<28} {interior_knots[0].size:>12d}")
    print(f"{'Interior knots x1':<28} {interior_knots[1].size:>12d}")
    print(f"{'Number of coefficients':<28} {spline.get_coeffs().shape[0]:>12d}")
    print()
    print("Training Fit")
    print("-" * 37)
    print(f"{'Weighted residual':<28} {spline.get_residual():>12.6g}")
    print(f"{'MAE vs noisy samples':<28} {mae_noisy:>12.6g}")
    print(f"{'RMSE vs noisy samples':<28} {rmse_noisy:>12.6g}")


if __name__ == "__main__":
    test_lsq_multivariate_spline_2d()
