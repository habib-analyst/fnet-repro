import numpy as np

from fnetrepro.data import (
    FAKE_SUBTYPES,
    FRAME_H,
    FRAME_W,
    MEL_BINS,
    SUBTYPES,
    T_FRAMES,
    build_dataset,
    generate_split,
    make_identities,
)


def test_split_shapes_and_balance():
    ids = make_identities(4, seed=3)
    split = generate_split(ids, n_per_identity=8, seed=11)
    n = 4 * 8
    assert split["audio"].shape == (n, 1, MEL_BINS, T_FRAMES)
    assert split["visual"].shape == (n, T_FRAMES, 1, FRAME_H, FRAME_W)
    assert split["y"].shape == (n,)
    # Generation alternates real / fake, so the split is balanced.
    assert int((split["y"] == 0).sum()) == n // 2
    assert int((split["y"] == 1).sum()) == n // 2
    for name in FAKE_SUBTYPES:
        assert int((split["subtype"] == SUBTYPES.index(name)).sum()) > 0


def test_generation_is_deterministic():
    ids = make_identities(3, seed=5)
    a = generate_split(ids, n_per_identity=4, seed=9)
    b = generate_split(ids, n_per_identity=4, seed=9)
    assert np.array_equal(a["audio"], b["audio"])
    assert np.array_equal(a["visual"], b["visual"])
    assert np.array_equal(a["y"], b["y"])


def test_build_dataset_identity_disjoint_and_counts():
    ds = build_dataset()
    assert len(ds["train"]["y"]) == 728
    assert len(ds["val"]["y"]) == 240
    assert len(ds["test"]["y"]) == 300
    # Identity pools come from disjoint slices of one identity list, and
    # each split records only its own pool's local indices.
    assert int(ds["train"]["identity"].max()) < 14
    assert int(ds["val"]["identity"].max()) < 4
    assert int(ds["test"]["identity"].max()) < 6
    # Labels agree with subtypes everywhere.
    for split in ds.values():
        real = split["subtype"] == SUBTYPES.index("real")
        assert np.array_equal(real, split["y"] == 0)


def test_visual_frames_in_unit_range():
    ids = make_identities(2, seed=1)
    split = generate_split(ids, n_per_identity=2, seed=2)
    assert split["visual"].min() >= 0.0
    assert split["visual"].max() <= 1.0
    assert split["audio"].min() >= 0.0
