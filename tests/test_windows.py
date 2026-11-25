from esdvideo.windows import MIN_SECONDS, WINDOW, tile, train_windows, target


def test_tile_scores_every_second_once():
    for n in [MIN_SECONDS, 5, 31, 32, 33, 63, 64, 65, 100, 1411]:
        scored = [s for _, _, a, b in tile(n) for s in range(a, b)]
        assert scored == list(range(n)), n
        for start, length, a, b in tile(n):
            assert 0 <= start and start + length <= n and length <= WINDOW and start <= a < b <= start + length


def test_tile_skips_very_short_clips():
    assert tile(MIN_SECONDS - 1) == []


def test_train_windows():
    assert train_windows(20) == [(0, 20)]
    assert train_windows(48) == [(0, 32), (8, 32), (16, 32)]
    assert train_windows(MIN_SECONDS - 1) == []


def test_target_is_single_segment():
    assert target(["Hemostasis"] * 7) == '{"segments":[{"phase":"Hemostasis","start_seconds":0,"end_seconds":7}]}'
