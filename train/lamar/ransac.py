"""RANSAC of the density map -> floorplan similarity whose scale is restricted to the band of Z-FLoc.

The density map and the floorplan are both metric, so the scale of the similarity is known up to the errors of the
reconstruction: Z-FLoc (zfloc/matching/geometry.py, scale_bounds) rejects the hypotheses whose scale is outside
resolution_ref / resolution_src * (1 -/+ scale_tol), scale_tol = 0.35 (zfloc/config.py).
"""
import numpy as np

SCALE_TOL = 0.35  # scale_tol of Z-FLoc


def scale_band(density_px_per_m: float, plan_px_per_m: float, tol: float = SCALE_TOL) -> tuple[float, float]:
    """
    Scale band of the density map -> floorplan similarity (Z-FLoc scale_bounds).
    Args:
        density_px_per_m (float): density map pixels per metre (proj_params["scale"])
        plan_px_per_m (float): floorplan pixels per metre (resized plan: plan_params["scale"] / 0.055)
        tol (float): relative half width of the band

    Returns:
        tuple[float, float]: smallest and largest scale
    """
    centre = plan_px_per_m / density_px_per_m  # = resolution_ref / resolution_src of Z-FLoc
    return centre * (1.0 - tol), centre * (1.0 + tol)


def similarity_matrix(q: complex, t: complex) -> np.ndarray:
    """
    The similarity z -> q z + t of the complex plane as a matrix.
    Args:
        q (complex): scale * exp(i * angle)
        t (complex): translation

    Returns:
        np.ndarray: tensor with shape [2, 3], as of cv2.estimateAffinePartial2D
    """
    return np.array([[q.real, -q.imag, t.real], [q.imag, q.real, t.imag]])


def estimate_similarity_ransac(src: np.ndarray, dst: np.ndarray, thr: float, scale_range: tuple[float, float],
                               rng: np.random.Generator, max_iters: int = 2000, chunk: int = 256) -> np.ndarray | None:
    """
    RANSAC of a similarity (rotation, uniform scale, translation, no reflection) from correspondences, whose minimal
    hypotheses (2 correspondences) with a scale outside scale_range are rejected; the best one is refitted on its
    inliers (least squares) with the scale clipped to scale_range.
    Args:
        src (np.ndarray): tensor with shape [N, 2], density map px
        dst (np.ndarray): tensor with shape [N, 2], floorplan px
        thr (float): inlier threshold on the reprojection error, floorplan px
        scale_range (tuple[float, float]): smallest and largest scale
        rng (np.random.Generator): draws the minimal samples
        max_iters (int): minimal samples (the default of cv2.estimateAffinePartial2D)
        chunk (int): hypotheses scored at once (memory)

    Returns:
        np.ndarray | None: tensor with shape [2, 3] (src -> dst), or None without a hypothesis in the band
    """
    if len(src) < 2:
        return None
    zs = src[:, 0].astype(np.float64) + 1j * src[:, 1]
    zd = dst[:, 0].astype(np.float64) + 1j * dst[:, 1]
    i, j = rng.integers(0, len(zs), size=(2, max_iters))
    ds = zs[j] - zs[i]
    valid = np.abs(ds) > 1e-6
    q = (zd[j] - zd[i])[valid] / ds[valid]  # scale * exp(i * angle)
    t = zd[i][valid] - q * zs[i][valid]
    in_band = (np.abs(q) >= scale_range[0]) & (np.abs(q) <= scale_range[1])
    q, t = q[in_band], t[in_band]
    if len(q) == 0:
        return None

    counts = np.concatenate([(np.abs(q[k:k + chunk, None] * zs[None] + t[k:k + chunk, None] - zd[None]) < thr).sum(1)
                             for k in range(0, len(q), chunk)])
    best = int(np.argmax(counts))
    inliers = np.abs(q[best] * zs + t[best] - zd) < thr
    if inliers.sum() < 2:
        return similarity_matrix(q[best], t[best])

    # Least squares on the inliers (2D Umeyama), the scale clipped to the band
    ms, md = zs[inliers].mean(), zd[inliers].mean()
    cs, cd = zs[inliers] - ms, zd[inliers] - md
    q_fit = np.sum(np.conj(cs) * cd) / np.sum(np.abs(cs) ** 2)
    q_fit = q_fit / np.abs(q_fit) * np.clip(np.abs(q_fit), *scale_range)
    return similarity_matrix(q_fit, md - q_fit * ms)
