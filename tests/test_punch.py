"""Contract tests for :mod:`hole.punch`.

In the initial repository every stubbed function raises
``NotImplementedError``, so all fifteen tests fail.  The tests are the
specification: do not weaken or edit them while implementing the stubs.

Grid conventions used below match ``hole.punch``: ``mask[y][x]`` and
``height[y][x]`` are row-major, ``x`` grows right, ``y`` grows down, and cells
are the unit squares ``[x, x + 1] x [y, y + 1]``.  Every float comparison uses
an absolute tolerance of ``1e-9``.
"""

import pytest

from hole.punch import (
    boundary_cells,
    clip_triangles,
    depth_bias,
    hole_batch,
    hole_mask,
    lod_consistent,
    stitch_ring,
    validate_hole,
)

TOL = 1e-9


def _mask(height, width, holes):
    grid = [[False] * width for _ in range(height)]
    for x, y in holes:
        grid[y][x] = True
    return grid


def _height_field(height, width):
    return [[float(10 * y + x) for x in range(width)] for y in range(height)]


def _centres(width, height):
    return [(x + 0.5, y + 0.5) for y in range(height) for x in range(width)]


def _chebyshev_depth(mask):
    """Same depth as the ``depth_bias`` contract, used to build valid heights."""
    rows = len(mask)
    cols = len(mask[0])
    depths = []
    for y in range(rows):
        row = []
        for x in range(cols):
            if not mask[y][x]:
                row.append(0)
                continue
            best = min(x + 1, cols - x, y + 1, rows - y)
            for other_y in range(rows):
                for other_x in range(cols):
                    if mask[other_y][other_x]:
                        continue
                    best = min(best, max(abs(other_x - x), abs(other_y - y)))
            row.append(best)
        depths.append(row)
    return depths


def _signed_area(polygon):
    total = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def test_defect01_hole_mask_counts_boundary_as_inside():
    """Defect 1: a point on an edge or vertex must not be treated as outside."""
    polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    points = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (0.0, 4.0),
        (2.0, 0.0),
        (4.0, 2.0),
        (2.0, 4.0),
        (0.0, 2.0),
        (2.0, 2.0),
        (4.0 + 5e-10, 2.0),
        (5.0, 2.0),
    ]

    flags = hole_mask(points, [polygon])

    assert flags == [True] * 10 + [False]


def test_defect02_hole_mask_uses_even_odd_for_self_intersection_and_nesting():
    """Defect 2: a concave/self-intersecting or nested boundary is even-odd."""
    bowtie = [(-2.0, -2.0), (2.0, 2.0), (2.0, -2.0), (-2.0, 2.0)]
    lobe_flags = hole_mask(
        [(-1.5, 0.5), (1.5, 0.5), (0.0, 0.5), (0.0, 1.5)],
        [bowtie],
    )

    assert lobe_flags == [True, True, False, False]

    outer = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
    inner = [(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]
    nested_flags = hole_mask([(1.0, 1.0), (3.0, 3.0), (5.0, 1.0)], [outer, inner])

    assert nested_flags == [True, False, True]


def test_defect03_boundary_cells_include_diagonal_neighbours():
    """Defect 3: diagonal rim cells must be boundary cells too."""
    mask = _mask(5, 5, [(2, 2)])

    assert boundary_cells(mask) == [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ]


def test_defect04_stitch_ring_order_is_stable_and_ring_shaped():
    """Defect 4: set-iteration order is not a ring; the order is prescribed."""
    mask = _mask(3, 3, [(1, 1)])
    height = _height_field(3, 3)
    expected = [
        (0, 0, 0.0),
        (1, 0, 1.0),
        (2, 0, 2.0),
        (2, 1, 12.0),
        (2, 2, 22.0),
        (1, 2, 21.0),
        (0, 2, 20.0),
        (0, 1, 10.0),
    ]

    assert stitch_ring(mask, height) == expected
    assert stitch_ring(mask, height) == expected


def test_defect05_stitch_ring_wraps_the_last_vertex_to_the_first():
    """Defect 5: the rim must close, including the last-to-first segment."""
    holes = [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)]
    mask = _mask(5, 5, holes)
    height = [[0.0] * 5 for _ in range(5)]

    ring = stitch_ring(mask, height)
    cells = [(x, y) for x, y, _ in ring]

    assert len(cells) == 16
    assert len(set(cells)) == 16
    for index, cell in enumerate(cells):
        following = cells[(index + 1) % len(cells)]
        assert max(abs(cell[0] - following[0]), abs(cell[1] - following[1])) == 1


def test_defect06_depth_bias_grows_with_distance_from_the_rim():
    """Defect 6: a constant offset must be replaced by a relative one."""
    holes = [(x, y) for y in range(1, 6) for x in range(1, 6)]
    mask = _mask(7, 7, holes)

    values = depth_bias(mask, 2.0)

    assert len(values) == 25
    assert values[0] == 2.0
    assert values[7] == 4.0
    assert values[12] == 6.0
    assert min(values) == 2.0
    assert max(values) == 6.0


def test_defect07_lod_consistent_keeps_a_corner_hole():
    """Defect 7: the fine mask cannot be reused as the coarse LOD mask."""
    mask = _mask(6, 6, [(5, 5)])

    coarse = lod_consistent(mask, 2)

    assert len(coarse) == 9
    assert coarse == [False] * 8 + [True]


def test_defect08_clip_triangles_keeps_the_area_outside_the_hole():
    """Defect 8: a triangle touching a hole must be clipped, not deleted."""
    triangle = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
    mask = _mask(4, 4, [(0, 0)])

    pieces = clip_triangles([triangle], mask)

    assert pieces
    total = sum(_signed_area(piece) for piece in pieces)
    assert total == pytest.approx(3.5, abs=TOL)
    for piece in pieces:
        assert len(piece) >= 3
        assert _signed_area(piece) > TOL
        for x, y in piece:
            assert not (TOL < x < 1.0 - TOL and TOL < y < 1.0 - TOL)


def test_defect09_validate_hole_reports_open_boundary_and_bad_fields():
    """Defect 9: an open rim, a NaN height, or a bump must be reported."""
    open_mask = _mask(3, 3, [(1, 0)])
    open_depth = _chebyshev_depth(open_mask)
    open_height = [[10.0 - value for value in row] for row in open_depth]
    open_issues = validate_hole(open_mask, open_height)

    assert "boundary_open" in open_issues
    assert open_issues == sorted(open_issues)

    mask = _mask(5, 5, [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)])
    depth = _chebyshev_depth(mask)
    height = [[10.0 - value for value in row] for row in depth]

    assert validate_hole(mask, height) == []

    spike = [row[:] for row in height]
    spike[2][2] = 99.0
    assert "depth_non_monotonic" in validate_hole(mask, spike)

    poisoned = [row[:] for row in height]
    poisoned[2][1] = float("nan")
    assert "height_nan" in validate_hole(mask, poisoned)


def test_defect10_hole_batch_does_not_reuse_the_previous_mask():
    """Defect 10: each frame must be evaluated with its own mask."""
    big = _mask(5, 5, [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)])
    small = _mask(5, 5, [(2, 2)])
    height = [[0.0] * 5 for _ in range(5)]
    frames = [
        {"id": "first", "mask": big, "height": height},
        {"id": "second", "mask": small, "height": height},
    ]

    result = hole_batch(frames, {"bias": 1.5, "lod": 2})

    assert result["count"] == 2
    assert [frame["id"] for frame in result["frames"]] == ["first", "second"]
    assert len(result["frames"][0]["bias"]) == 9
    assert len(result["frames"][1]["bias"]) == 1
    assert result["frames"][1]["bias"] == depth_bias(small, 1.5)


def test_empty_polygons_produce_an_all_false_mask():
    """Complement: a hole with no boundary has no cells."""
    flags = hole_mask([(0.0, 0.0), (2.0, 3.0)], [])

    assert flags == [False, False]


def test_single_cell_hole_marks_exactly_one_cell():
    """Complement: a one-cell hole covers exactly one unit cell."""
    tiny = [(1.4, 1.4), (1.6, 1.4), (1.6, 1.6), (1.4, 1.6)]

    flags = hole_mask(_centres(4, 4), [tiny])

    assert flags == [index == 5 for index in range(16)]


def test_stitch_ring_count_matches_boundary_cells_and_closes():
    """Complement: one vertex per boundary cell, no duplicates, closed ring."""
    mask = _mask(3, 3, [(1, 1)])
    height = _height_field(3, 3)

    ring = stitch_ring(mask, height)
    cells = boundary_cells(mask)
    ring_cells = [(x, y) for x, y, _ in ring]

    assert len(ring_cells) == len(cells) == 8
    assert len(set(ring_cells)) == 8
    assert set(ring_cells) == set(cells)
    assert all(z == height[y][x] for x, y, z in ring)
    assert max(
        abs(ring_cells[-1][0] - ring_cells[0][0]),
        abs(ring_cells[-1][1] - ring_cells[0][1]),
    ) == 1


def test_coarse_lod_hole_area_never_shrinks():
    """Complement: conservative LOD coverage must not lose hole area."""
    holes = [(0, 0), (2, 2), (4, 4)]
    mask = _mask(5, 5, holes)

    coarse = lod_consistent(mask, 2)
    fine_area = sum(1 for row in mask for cell in row if cell)
    coarse_area = sum(1 for cell in coarse if cell) * 2 * 2

    assert coarse_area >= fine_area
    for x, y in holes:
        assert coarse[(y // 2) * 3 + (x // 2)]


def test_batch_result_matches_processing_each_frame_alone():
    """Complement: batch output equals processing each frame alone."""
    first = _mask(5, 5, [(x, y) for y in (1, 2, 3) for x in (1, 2, 3)])
    second = _mask(5, 5, [(2, 2)])
    height = [[0.0] * 5 for _ in range(5)]
    frames = [
        {"id": "left", "mask": first, "height": height},
        {"id": "right", "mask": second, "height": height},
    ]
    config = {"bias": 3.0, "lod": 2}

    result = hole_batch(frames, config)

    expected = {
        "count": 2,
        "frames": [
            {
                "id": "left",
                "bias": depth_bias(first, 3.0),
                "lod": lod_consistent(first, 2),
                "issues": validate_hole(first, height),
            },
            {
                "id": "right",
                "bias": depth_bias(second, 3.0),
                "lod": lod_consistent(second, 2),
                "issues": validate_hole(second, height),
            },
        ],
    }

    assert result == expected
