"""Contract stubs for terrain-hole masking, stitching, and LOD handling.

The terrain is a regular grid of unit cells.  ``mask[y][x]`` is ``True``
exactly for the cells that were dug away, and ``height[y][x]`` is the
world-space elevation of that cell in metres.  Cell ``(x, y)`` covers the
world-space square ``[x, x + 1] x [y, y + 1]``; ``x`` grows to the right and
``y`` grows downward, matching screen-space orientation.  A mask and a height
field must have the same rectangular shape and the heights must be finite.

Floating-point comparisons use an absolute tolerance of ``1e-9`` (``EPS``
below) unless a docstring says otherwise.  The module is deterministic and
offline and imports only :mod:`math`.  Every public function is intentionally a
stub that raises ``NotImplementedError`` until its contract is implemented.
"""

import math

EPS = 1e-9

__all__ = [
    "EPS",
    "boundary_cells",
    "clip_triangles",
    "depth_bias",
    "hole_batch",
    "hole_mask",
    "lod_consistent",
    "stitch_ring",
    "validate_hole",
]

# Neighbour offsets for the 8-connected (Chebyshev) stencil used throughout.
_NEIGHBOURS_8 = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dx != 0 or dy != 0
)

# The four axis-aligned 4-connected offsets.
_NEIGHBOURS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid_shape(mask):
    """Return ``(rows, cols)`` for a rectangular grid, tolerating an empty one."""
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    return rows, cols


def _point_segment_distance(px, py, ax, ay, bx, by):
    """Euclidean distance from point ``p`` to segment ``ab``."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def _chebyshev_depths(mask):
    """Chebyshev distance from every cell to the nearest non-hole cell.

    Cells outside the grid count as non-hole, so the border of the padded
    grid is seeded at distance zero.  Returns ``(rows, cols, grid)`` where
    ``grid[y][x]`` is the integer depth.  Only :mod:`math` is used, so the
    breadth-first queue is a plain list with a moving head index.
    """
    rows, cols = _grid_shape(mask)
    if rows == 0 or cols == 0:
        return rows, cols, []

    padded_rows = rows + 2
    padded_cols = cols + 2
    distances = [[-1] * padded_cols for _ in range(padded_rows)]
    queue = []
    for y in range(padded_rows):
        for x in range(padded_cols):
            inner = 1 <= y <= rows and 1 <= x <= cols
            if not inner or not mask[y - 1][x - 1]:
                distances[y][x] = 0
                queue.append((x, y))

    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1
        nxt = distances[y][x] + 1
        for dx, dy in _NEIGHBOURS_8:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < padded_cols and 0 <= ny < padded_rows:
                if distances[ny][nx] < 0 or distances[ny][nx] > nxt:
                    if distances[ny][nx] < 0:
                        distances[ny][nx] = nxt
                        queue.append((nx, ny))

    grid = [[distances[y + 1][x + 1] for x in range(cols)] for y in range(rows)]
    return rows, cols, grid


def _hole_components(mask, rows, cols):
    """Return 8-connected components of hole cells as lists of ``(x, y)``."""
    seen = [[False] * cols for _ in range(rows)]
    components = []
    for y in range(rows):
        for x in range(cols):
            if not mask[y][x] or seen[y][x]:
                continue
            seen[y][x] = True
            stack = [(x, y)]
            component = []
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for dx, dy in _NEIGHBOURS_8:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
            components.append(component)
    return components


def _rim_is_closed(rim):
    """Check the rim closure conditions for one hole component."""
    if not rim:
        return False
    # Every rim cell must have at least two distinct rim neighbours.
    for x, y in rim:
        neighbours = 0
        for dx, dy in _NEIGHBOURS_8:
            if (x + dx, y + dy) in rim:
                neighbours += 1
        if neighbours < 2:
            return False
    # All rim cells must form a single 8-connected component.
    remaining = set(rim)
    start = next(iter(remaining))
    remaining.discard(start)
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _NEIGHBOURS_8:
            cell = (x + dx, y + dy)
            if cell in remaining:
                remaining.discard(cell)
                stack.append(cell)
    return not remaining


def _clip_halfplane(poly, a, b, c):
    """Sutherland-Hodgman clip keeping points with ``a*x + b*y <= c``."""
    if not poly:
        return []
    clipped = []
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        d1 = a * x1 + b * y1 - c
        d2 = a * x2 + b * y2 - c
        in1 = d1 <= EPS
        in2 = d2 <= EPS
        if in1:
            clipped.append((x1, y1))
        if in1 != in2:
            denominator = d1 - d2
            if denominator != 0.0:
                t = d1 / denominator
                clipped.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return clipped


def _polygon_area(poly):
    """Signed shoelace area of a polygon given as ``(x, y)`` pairs."""
    total = 0.0
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def _clean_polygon(poly):
    """Drop duplicate/tiny vertices, discard zero-area fragments, orient CCW."""
    cleaned = []
    for point in poly:
        if cleaned:
            last = cleaned[-1]
            if abs(point[0] - last[0]) <= 1e-12 and abs(point[1] - last[1]) <= 1e-12:
                continue
        cleaned.append(point)
    if len(cleaned) >= 2:
        first = cleaned[0]
        last = cleaned[-1]
        if abs(first[0] - last[0]) <= 1e-12 and abs(first[1] - last[1]) <= 1e-12:
            cleaned.pop()
    if len(cleaned) < 3:
        return None
    area = _polygon_area(cleaned)
    if area < 0.0:
        cleaned.reverse()
        area = -area
    if area <= EPS:
        return None
    return cleaned


def _subtract_square(poly, x0, y0, x1, y1):
    """Return convex pieces of ``poly`` outside the square ``[x0,x1]x[y0,y1]``.

    The complement of the open square is partitioned into four convex half-plane
    intersections (left, right, below, above), so each piece stays convex and
    the pieces only share edges.
    """
    regions = (
        ((1.0, 0.0, x0),),
        ((-1.0, 0.0, -x1),),
        ((-1.0, 0.0, -x0), (1.0, 0.0, x1), (0.0, 1.0, y0)),
        ((-1.0, 0.0, -x0), (1.0, 0.0, x1), (0.0, -1.0, -y1)),
    )
    pieces = []
    for halfplanes in regions:
        current = list(poly)
        for a, b, c in halfplanes:
            current = _clip_halfplane(current, a, b, c)
            if not current:
                break
        cleaned = _clean_polygon(current)
        if cleaned is not None:
            pieces.append(cleaned)
    return pieces


def hole_mask(points, polygons) -> list[bool]:
    """Return even-odd inside flags for world-space points.

    Contract:
      * ``points`` is a sequence of ``(x, y)`` float pairs.
      * ``polygons`` is a sequence of polygons.  A polygon is a sequence of
        ``(x, y)`` vertices and is implicitly closed (the last vertex connects
        back to the first).  The edges of every polygon together form one
        even-odd region: a point is inside when a ray towards ``+x`` crosses
        an odd number of edges.  Therefore a polygon nested inside another
        polygon cuts a hole, and a self-intersecting polygon (a "bowtie")
        keeps the crossing lobes inside while the gap between them stays
        outside.  OR-ing the polygons or using a winding rule is wrong.
      * Boundary points count as inside.  A point whose distance to any edge
        or vertex is at most ``1e-9`` is ``True`` even when the ray cast lands
        on the outside side.  Skipping that check drops a row of rim cells and
        opens a seam along the hole edge.
      * Return ``list[bool]`` with exactly ``len(points)`` entries in input
        order.  Empty ``polygons`` returns all ``False``.  Neither argument is
        mutated.
    """
    edges = []
    for polygon in polygons:
        count = len(polygon)
        for index in range(count):
            ax, ay = polygon[index]
            bx, by = polygon[(index + 1) % count]
            edges.append((ax, ay, bx, by))

    flags = []
    for px, py in points:
        if not edges:
            flags.append(False)
            continue

        # Boundary points (within EPS of any edge) are inside by definition.
        on_boundary = False
        for ax, ay, bx, by in edges:
            if _point_segment_distance(px, py, ax, ay, bx, by) <= EPS:
                on_boundary = True
                break
        if on_boundary:
            flags.append(True)
            continue

        # Even-odd parity from a ray towards +x.  Half-open crossing rule
        # (ay <= py < by or by <= py < ay) counts a vertex-touching ray once.
        crossings = 0
        for ax, ay, bx, by in edges:
            if (ay <= py < by) or (by <= py < ay):
                t = (py - ay) / (by - ay)
                ix = ax + t * (bx - ax)
                if ix > px + EPS:
                    crossings += 1
        flags.append(crossings % 2 == 1)
    return flags


def boundary_cells(mask) -> list[tuple[int, int]]:
    """Return the cells that touch the hole without being part of it.

    Contract:
      * ``mask`` is a rectangular ``list[list[bool]]``; ``True`` marks a hole
        cell.
      * A cell is a boundary cell iff its own flag is ``False`` and at least
        one of its eight neighbours (Chebyshev distance 1: horizontal,
        vertical, or diagonal) is ``True``.  Looking only at the four
        axis-aligned neighbours is a defect: the diagonal rim cells would be
        omitted and a diagonal seam would remain at the hole edge.
      * Cells outside the grid do not exist and are ignored.
      * Return ``(x, y)`` pairs sorted by ``(y, x)`` ascending, so callers get
        a stable order.  A mask without hole cells returns ``[]``.
    """
    rows, cols = _grid_shape(mask)
    cells = []
    for y in range(rows):
        for x in range(cols):
            if mask[y][x]:
                continue
            for dx, dy in _NEIGHBOURS_8:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < cols and 0 <= ny < rows and mask[ny][nx]:
                    cells.append((x, y))
                    break
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    return cells


def stitch_ring(mask, height) -> list[tuple[int, int, float]]:
    """Build a closed stitch ring around the hole.

    Contract:
      * ``mask`` is the hole mask and ``height`` is the same-shaped terrain
        elevation field (metres, finite floats).
      * The ring contains exactly the :func:`boundary_cells` of ``mask``,
        once each, as ``(x, y, height[y][x])``.  Vertices must not repeat and
        the first vertex is not appended at the end: the caller closes the
        polygon from the last entry back to the first.
      * The order is the ring order, not set-iteration order.  Let ``(cx, cy)``
        be the arithmetic mean of all hole-cell coordinates and sort by
        increasing ``theta = atan2(y - cy, x - cx)``.  The ``y`` axis grows
        downward, so increasing ``theta`` walks the rim clockwise on screen.
        Ties are broken by squared distance to ``(cx, cy)`` and then by
        ``(y, x)``.  A set with an implementation-defined iteration order is a
        defect: adjacent stitch quads would flip winding between frames.
      * Boundary vertices are 8-adjacent along the ring, including the
        implicit last-to-first edge; otherwise the ring is open and a segment
        of the rim has no stitch band.
      * Return ``[]`` when the mask has no hole cells.
    """
    rows, cols = _grid_shape(mask)
    hole_cells = [
        (x, y) for y in range(rows) for x in range(cols) if mask[y][x]
    ]
    if not hole_cells:
        return []

    cx = sum(x for x, _ in hole_cells) / len(hole_cells)
    cy = sum(y for _, y in hole_cells) / len(hole_cells)

    def sort_key(cell):
        x, y = cell
        dx = x - cx
        dy = y - cy
        return (math.atan2(dy, dx), dx * dx + dy * dy, y, x)

    ring = sorted(boundary_cells(mask), key=sort_key)
    return [(x, y, float(height[y][x])) for x, y in ring]


def depth_bias(mask, bias) -> list[float]:
    """Return the per-cell depth offset that hides the hole rim.

    Contract:
      * ``mask`` is the hole mask and ``bias`` is a finite non-negative scalar
        in world units.
      * For a hole cell ``p`` let ``d(p)`` be the Chebyshev distance from
        ``p`` to the nearest non-hole cell, with every cell outside the grid
        counted as non-hole.  A rim cell therefore has ``d == 1`` and an
        interior cell is deeper.  The offset of ``p`` is ``bias * d(p)``.
        Using the constant ``bias`` for every cell is a defect: a hole close
        to the camera would still bleed its shallow layer through the rim.
      * Return one float per hole cell in row-major order (ascending ``y``,
        then ``x``), so the length equals the number of ``True`` cells.  An
        empty hole returns ``[]``.  Values are non-negative and accurate
        within ``1e-9``.  Neither argument is mutated.
    """
    rows, cols, depths = _chebyshev_depths(mask)
    values = []
    for y in range(rows):
        for x in range(cols):
            if mask[y][x]:
                values.append(float(bias) * depths[y][x])
    return values


def lod_consistent(mask, lod) -> list[bool]:
    """Downsample the hole mask to a coarse LOD without losing a hole.

    Contract:
      * ``mask`` is a rectangular fine mask and ``lod`` is a positive integer
        tile size measured in fine cells.
      * The coarse grid has ``ceil(height / lod)`` rows and
        ``ceil(width / lod)`` columns; partial tiles on the right and bottom
        edges cover fewer fine cells.
      * Coarse cell ``(cx, cy)`` is ``True`` iff at least one fine cell
        ``(x, y)`` with ``cx * lod <= x < (cx + 1) * lod`` and
        ``cy * lod <= y < (cy + 1) * lod`` is a hole.  This union rule is
        conservative: a coarse tile may cover extra terrain but must never
        drop a hole, because a dropped hole shows up as a black blob that
        flashes at distance.  Reusing the fine mask directly is a defect
        because the coarse grid then has the wrong resolution.
      * Return the coarse mask flattened row-major as ``list[bool]`` (length
        ``ch * cw``).  ``lod == 1`` reproduces the fine mask flattened.
        ``mask`` and ``lod`` are not mutated.
    """
    rows, cols = _grid_shape(mask)
    if rows == 0 or cols == 0:
        return []
    coarse_rows = (rows + lod - 1) // lod
    coarse_cols = (cols + lod - 1) // lod
    coarse = []
    for cy in range(coarse_rows):
        y_start = cy * lod
        y_end = min(y_start + lod, rows)
        for cx in range(coarse_cols):
            x_start = cx * lod
            x_end = min(x_start + lod, cols)
            is_hole = False
            for y in range(y_start, y_end):
                row = mask[y]
                for x in range(x_start, x_end):
                    if row[x]:
                        is_hole = True
                        break
                if is_hole:
                    break
            coarse.append(is_hole)
    return coarse


def clip_triangles(tris, mask) -> list:
    """Clip triangles against the hole cells.

    Contract:
      * ``tris`` is a sequence of triangles; each triangle is a sequence of
        three ``(x, y)`` world-space float vertices.  ``mask`` is the hole
        mask on the unit grid, so cell ``(x, y)`` is the square
        ``[x, x + 1] x [y, y + 1]``.  Cells outside the mask are terrain, not
        holes.
      * For every triangle return the part of its area that lies outside the
        hole cells.  Deleting a whole triangle because it touches a hole is a
        defect: the hole would grow by up to a triangle and the rim would
        bleed.  The union of the returned pieces must equal the input union
        minus the hole cells, and pieces must not overlap.
      * Pieces are convex polygons, each a list of ``(x, y)`` floats with
        positive signed area under the shoelace formula.  A piece has at least
        three vertices, no repeated closing vertex, and positive area.  A
        triangle may be split along grid lines, so the original triangle count
        need not be preserved.  Zero-area fragments are dropped.
      * Return a flat ``list`` of polygons.  A triangle that lies entirely in
        the hole contributes nothing.  The inputs are not mutated.
    """
    rows, cols = _grid_shape(mask)
    pieces = []
    for triangle in tris:
        poly = _clean_polygon([(float(x), float(y)) for x, y in triangle])
        if poly is None:
            continue
        if len(poly) != 3:
            continue
        min_x = max(0, int(math.floor(min(p[0] for p in poly))))
        max_x = min(cols - 1, int(math.ceil(max(p[0] for p in poly))))
        min_y = max(0, int(math.floor(min(p[1] for p in poly))))
        max_y = min(rows - 1, int(math.ceil(max(p[1] for p in poly))))

        current = [poly]
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if not mask[y][x]:
                    continue
                next_pieces = []
                for piece in current:
                    next_pieces.extend(
                        _subtract_square(piece, float(x), float(y), float(x + 1), float(y + 1))
                    )
                current = next_pieces
                if not current:
                    break
            if not current:
                break
        pieces.extend(current)
    return pieces


def validate_hole(mask, height) -> list[str]:
    """Return sorted problem codes for a candidate hole.

    Contract:
      * ``mask`` is the hole mask and ``height`` is the same-shaped terrain
        elevation field.  The function reports problems instead of raising.
      * Checks:
          - ``"height_nan"``: any height is not finite.
          - ``"boundary_open"``: a hole is not a closed depression.  For each
            8-connected component of hole cells, collect its rim: the
            non-hole cells that are 8-adjacent to at least one cell of that
            component.  The component is closed iff no cell of it lies on the
            outer border of the grid, every rim cell has at least two distinct
            rim neighbours (8-adjacency), and all rim cells belong to one
            8-connected rim component.
          - ``"depth_non_monotonic"``: inside a hole the floor must not get
            higher towards the centre.  Let ``d`` be the same Chebyshev depth
            as :func:`depth_bias` (cells outside the grid count as non-hole).
            For every pair of 4-adjacent hole cells ``p`` and ``q`` with
            ``d(p) < d(q)``, require ``height[q] <= height[p] + 1e-9``.  A bump
            inside the hole violates this rule.
      * Return the codes as a sorted ``list[str]`` without duplicates; an
        empty list means the hole is valid.  Neither argument is mutated.
    """
    rows, cols = _grid_shape(mask)
    issues = set()

    for y in range(rows):
        for x in range(cols):
            if not math.isfinite(height[y][x]):
                issues.add("height_nan")

    if rows and cols:
        components = _hole_components(mask, rows, cols)
        for component in components:
            rim = set()
            for x, y in component:
                if x == 0 or y == 0 or x == cols - 1 or y == rows - 1:
                    issues.add("boundary_open")
                for dx, dy in _NEIGHBOURS_8:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < cols and 0 <= ny < rows and not mask[ny][nx]:
                        rim.add((nx, ny))
            if not _rim_is_closed(rim):
                issues.add("boundary_open")

        _, _, depths = _chebyshev_depths(mask)
        for y in range(rows):
            for x in range(cols):
                if not mask[y][x]:
                    continue
                for dx, dy in _NEIGHBOURS_4:
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < cols and 0 <= ny < rows):
                        continue
                    if not mask[ny][nx]:
                        continue
                    if depths[y][x] < depths[ny][nx]:
                        if height[ny][nx] > height[y][x] + 1e-9:
                            issues.add("depth_non_monotonic")
                    elif depths[ny][nx] < depths[y][x]:
                        if height[y][x] > height[ny][nx] + 1e-9:
                            issues.add("depth_non_monotonic")

    return sorted(issues)


def hole_batch(frames, config) -> dict:
    """Process a batch of terrain-hole frames independently.

    Contract:
      * ``frames`` is a sequence of dictionaries.  Each frame has ``"id"`` (a
        string), ``"mask"`` (the hole mask), and ``"height"`` (the same-shaped
        elevation field).  Extra keys are ignored.
      * ``config`` is a dictionary with ``"bias"`` (finite float >= 0) and
        ``"lod"`` (positive int); both keys are required.
      * Return::

          {
              "count": len(frames),
              "frames": [
                  {
                      "id": frame["id"],
                      "bias": depth_bias(frame["mask"], config["bias"]),
                      "lod": lod_consistent(frame["mask"], config["lod"]),
                      "issues": validate_hole(frame["mask"], frame["height"]),
                  },
                  ...
              ],
          }

        The list preserves input order.  Every frame is evaluated with its own
        mask and height; caching or reusing the previous frame's mask is a
        defect (frame two would inherit frame one's hole).  An empty batch
        returns ``{"count": 0, "frames": []}``.
      * Frame dictionaries and ``config`` are not mutated.
    """
    bias = config["bias"]
    lod = config["lod"]
    results = []
    for frame in frames:
        mask = frame["mask"]
        height = frame["height"]
        results.append(
            {
                "id": frame["id"],
                "bias": depth_bias(mask, bias),
                "lod": lod_consistent(mask, lod),
                "issues": validate_hole(mask, height),
            }
        )
    return {"count": len(frames), "frames": results}
