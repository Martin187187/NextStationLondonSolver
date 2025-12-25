from __future__ import annotations

from typing import Literal


NodeType = Literal["triangle", "square", "pentagon", "circle", "any"]
Point = tuple[int, int]
RegionId = str


def region_for_point(point: Point) -> RegionId:
	"""Return one of 13 region ids.

	Regions:
	- 9 block regions using the bins:
	  x in 0..2, 3..6, 7..9 and y in 0..2, 3..6, 7..9
	- plus 4 corner regions that override the corners:
	  (0,0), (9,0), (0,9), (9,9)
	"""
	x, y = point

	# Corner overrides (separate regions)
	if (x, y) == (0, 0):
		return "corner_sw"
	if (x, y) == (9, 0):
		return "corner_se"
	if (x, y) == (0, 9):
		return "corner_nw"
	if (x, y) == (9, 9):
		return "corner_ne"

	def x_bin(xv: int) -> str:
		if 0 <= xv <= 2:
			return "x0_2"
		if 3 <= xv <= 6:
			return "x3_6"
		if 7 <= xv <= 9:
			return "x7_9"
		raise ValueError(f"x={xv} outside expected range 0..9")

	def y_bin(yv: int) -> str:
		if 0 <= yv <= 2:
			return "y0_2"
		if 3 <= yv <= 6:
			return "y3_6"
		if 7 <= yv <= 9:
			return "y7_9"
		raise ValueError(f"y={yv} outside expected range 0..9")

	return f"{x_bin(x)}__{y_bin(y)}"


def build_8_neighbor_edges(points: set[Point]) -> set[tuple[Point, Point]]:
	offsets = [
		(-1, -1),
		(-1, 0),
		(-1, 1),
		(0, -1),
		(0, 1),
		(1, -1),
		(1, 0),
		(1, 1),
	]

	# Connect along each of the 8 directions even with gaps, e.g. (1,0) -> (3,0)
	# when (2,0) is missing. For each direction we connect to the nearest visible node.
	if not points:
		return set()

	xs = [x for x, _ in points]
	ys = [y for _, y in points]
	max_step = max(max(xs) - min(xs), max(ys) - min(ys)) + 2

	edges: set[tuple[Point, Point]] = set()
	for x, y in points:
		for dx, dy in offsets:
			for step in range(1, max_step + 1):
				candidate = (x + dx * step, y + dy * step)
				if candidate in points:
					a = (x, y)
					b = candidate
					edges.add((a, b) if a <= b else (b, a))
					break
	return edges


def main() -> None:
	# Define your nodes here (integer grid coordinates) with their type inline.
	# Example: (x, y): "triangle". Valid types: triangle, square, pentagon, circle, any
	nodes: dict[Point, NodeType] = {
		(0, 0): "triangle",
		(1, 0): "square",
		(3, 0): "pentagon",
		(4, 0): "circle",
		(5, 0): "triangle",
		(7, 0): "circle",
		(9, 0): "square",
		(1, 1): "circle",
		(6, 1): "pentagon",
		(8, 1): "triangle",
		(0, 2): "circle",
		(2, 2): "square",
		(3, 2): "circle",
		(5, 2): "pentagon",
		(8, 2): "circle",
		(9, 2): "pentagon",
		(3, 3): "pentagon",
		(4, 3): "triangle",
		(6, 3): "square",
		(7, 3): "triangle",
		(9, 3): "triangle",
		(0, 4): "pentagon",
		(2, 4): "square",
		(4, 4): "circle",
		(7, 4): "circle",
		(1, 5): "triangle",
		(2, 5): "square",
		(4, 5): "pentagon",
		(5, 5): "square",
		(8, 5): "pentagon",
		(0, 6): "square",
		(2, 6): "pentagon",
		(4, 6): "triangle",
		(5, 6): "any",
		(6, 6): "circle",
		(7, 6): "circle",
		(9, 6): "square",
		(0, 7): "circle",
		(3, 7): "triangle",
		(6, 7): "square",
		(9, 7): "triangle",
		(1, 8): "pentagon",
		(3, 8): "square",
		(6, 8): "pentagon",
		(8, 8): "square",
		(9, 8): "pentagon",
		(0, 9): "pentagon",
		(1, 9): "triangle",
		(2, 9): "square",
		(4, 9): "triangle",
		(5, 9): "circle",
		(7, 9): "triangle",
		(9, 9): "circle",
	}

	points: set[Point] = set(nodes.keys())

	edges = build_8_neighbor_edges(points)

	try:
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as exc:
		raise SystemExit(
			"matplotlib is required to plot. Install with: pip install matplotlib"
		) from exc

	fig, ax = plt.subplots(figsize=(6, 6))

	# Draw sudoku-like region boundaries for your specified bins.
	# Outer box (covers 0..9 grid)
	ax.axvline(-0.5, color="0.85", linewidth=2.0, zorder=0)
	ax.axvline(9.5, color="0.85", linewidth=2.0, zorder=0)
	ax.axhline(-0.5, color="0.85", linewidth=2.0, zorder=0)
	ax.axhline(9.5, color="0.85", linewidth=2.0, zorder=0)

	# Internal boundaries between bins: after 2 and after 6
	for boundary in (2.5, 6.5):
		ax.axvline(boundary, color="0.85", linewidth=2.0, zorder=0)
		ax.axhline(boundary, color="0.85", linewidth=2.0, zorder=0)

	# Draw edges (grey lines)
	for (x1, y1), (x2, y2) in edges:
		ax.plot([x1, x2], [y1, y2], color="0.7", linewidth=1.0, zorder=1)

	# Draw nodes (shape depends on node type)
	marker_by_type: dict[NodeType, str] = {
		"triangle": "^",
		"square": "s",
		"pentagon": "p",
		"circle": "o",
		"any": "X",
	}

	points_by_type: dict[NodeType, list[Point]] = {
		"triangle": [],
		"square": [],
		"pentagon": [],
		"circle": [],
		"any": [],
	}
	for point, point_type in nodes.items():
		points_by_type[point_type].append(point)

	# Color nodes by region (13 regions total), keep shape by type.
	region_ids = sorted({region_for_point(p) for p in points})
	cmap = plt.get_cmap("tab20")
	region_color: dict[RegionId, tuple[float, float, float, float]] = {
		rid: cmap(i % 20) for i, rid in enumerate(region_ids)
	}

	for point_type, typed_points in points_by_type.items():
		if not typed_points:
			continue
		xs = [x for x, _ in typed_points]
		ys = [y for _, y in typed_points]
		colors = [region_color[region_for_point(p)] for p in typed_points]
		ax.scatter(
			xs,
			ys,
			s=90,
			marker=marker_by_type[point_type],
			c=colors,
			edgecolors="black",
			linewidths=0.8,
			zorder=2,
		)

	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle=":", linewidth=0.8, color="0.9")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title("8-direction grid connectivity")
	plt.show()


if __name__ == "__main__":
	main()
