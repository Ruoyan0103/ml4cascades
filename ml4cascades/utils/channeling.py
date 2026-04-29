import numpy as np
from itertools import permutations, product


def get_equivalent_directions(hkl):
    """All symmetry-equivalent directions in a cubic crystal for a given Miller index."""
    h, k, l = hkl
    directions = set()
    for perm in permutations([h, k, l]):
        for signs in product([1, -1], repeat=3):
            d = tuple(s * p for s, p in zip(signs, perm))
            directions.add(d)
    return [np.array(d, dtype=float) for d in directions]


def min_angle_to_axes(velocity, hkl):
    """Minimum angle (degrees) between velocity and any equivalent axial direction."""
    d = np.array(velocity, dtype=float)
    d /= np.linalg.norm(d)
    min_angle, closest = 180.0, None
    for ax in get_equivalent_directions(hkl):
        ax_unit = ax / np.linalg.norm(ax)
        angle = np.degrees(np.arccos(np.clip(np.dot(d, ax_unit), -1.0, 1.0)))
        if angle < min_angle:
            min_angle, closest = angle, ax
    return min_angle, closest


def min_glancing_to_planes(velocity, hkl):
    """Minimum glancing angle (degrees) between velocity and any equivalent crystal plane."""
    d = np.array(velocity, dtype=float)
    d /= np.linalg.norm(d)
    min_glancing, closest_n = 90.0, None
    for n in get_equivalent_directions(hkl):
        n_unit = n / np.linalg.norm(n)
        glancing = 90.0 - np.degrees(np.arccos(np.clip(abs(np.dot(d, n_unit)), 0.0, 1.0)))
        if glancing < min_glancing:
            min_glancing, closest_n = glancing, n
    return min_glancing, closest_n


def check_channeling(velocity, E_keV=20, axial_crit=7.0, planar_crit=5.0):
    """
    Report angles from a PKA direction to all major channeling directions in diamond cubic Ge.

    Critical angles are approximate for 20 keV Ge→Ge (Lindhard theory):
      - Axial:  ~5–8°  (varies by axis; <110> is most open in diamond cubic)
      - Planar: ~3–6°  (varies by plane)

    Parameters
    ----------
    velocity : array-like (3,)
        PKA velocity or direction vector (only direction matters, any units).
    E_keV : float
        PKA kinetic energy in keV (for display only).
    axial_crit : float
        Threshold angle (degrees) for flagging axial channeling.
    planar_crit : float
        Threshold angle (degrees) for flagging planar channeling.
    """
    v = np.array(velocity, dtype=float)
    v_unit = v / np.linalg.norm(v)

    print(f"PKA direction : {list(map(int, velocity))}")
    print(f"Unit vector   : ({v_unit[0]:+.4f}, {v_unit[1]:+.4f}, {v_unit[2]:+.4f})")
    print()

    axial_families = [
        ('<100>', [1, 0, 0]),
        ('<110>', [1, 1, 0]),
        ('<111>', [1, 1, 1]),
        ('<112>', [1, 1, 2]),
    ]
    planar_families = [
        ('{100}', [1, 0, 0]),
        ('{110}', [1, 1, 0]),
        ('{111}', [1, 1, 1]),
    ]

    W = 56
    print(f"{'─' * W}")
    print(f"  Axial channeling   (critical ≈ {axial_crit}° for {E_keV} keV Ge)")
    print(f"{'─' * W}")
    for name, hkl in axial_families:
        angle, closest = min_angle_to_axes(velocity, hkl)
        c = closest.astype(int)
        flag = "  ← CHANNELING" if angle < axial_crit else ""
        print(f"  {name:6s}  {angle:5.1f}°   nearest [{c[0]:+d},{c[1]:+d},{c[2]:+d}]{flag}")

    print()
    print(f"{'─' * W}")
    print(f"  Planar channeling  (critical ≈ {planar_crit}° for {E_keV} keV Ge)")
    print(f"{'─' * W}")
    for name, hkl in planar_families:
        angle, closest_n = min_glancing_to_planes(velocity, hkl)
        c = closest_n.astype(int)
        flag = "  ← CHANNELING" if angle < planar_crit else ""
        print(f"  {name:6s}  {angle:5.1f}°   plane  ({c[0]:+d},{c[1]:+d},{c[2]:+d}){flag}")
    print()


if __name__ == '__main__':
    directions = [
        [-1480, -1540,  864],
        [1437, -580, 1706],
    ]
    for d in directions:
        print('=' * 56)
        check_channeling(d, E_keV=20)
