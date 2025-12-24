from __future__ import annotations


def build_8_neighbor_edges(points: set[tuple[int, int]]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
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

	edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
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
	# Define your nodes here (integer grid coordinates).
	points: set[tuple[int, int]] = {
		(0, 0),
		(1, 0),
		(3, 0),
		(4, 0),
		(5, 0),
		(7, 0),
		(9, 0),
		(1, 1),
		(6, 1),
		(8, 1),
        (0, 2),
        (2, 2),
        (3, 2),
        (5, 2),
        (8, 2),
        (9, 2),
        (3, 3),
        (4, 3),
        (6, 3),
        (7, 3),
        (9, 3),
        (0, 4),
        (2, 4),
        (4, 4),
        (7, 4),
        (1, 5),
        (2, 5),
        (4, 5),
        (5, 5), 
        (8, 5),
        (0, 6),
        (2, 6),
        (4, 6),
        (5, 6),
        (6, 6),
        (7, 6),
        (9, 6),
        (0, 7),
        (3, 7),
        (6, 7),
        (9, 7),
        (1, 8),
        (3, 8),
        (6, 8),
        (8, 8),
        (9, 8),
        (0, 9),
        (1, 9),
        (2, 9),
        (4, 9),
        (5, 9),
        (7, 9),
        (9, 9),
	}

	edges = build_8_neighbor_edges(points)

	try:
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as exc:
		raise SystemExit(
			"matplotlib is required to plot. Install with: pip install matplotlib"
		) from exc

	fig, ax = plt.subplots(figsize=(6, 6))

	# Draw edges (grey lines)
	for (x1, y1), (x2, y2) in edges:
		ax.plot([x1, x2], [y1, y2], color="0.7", linewidth=1.0, zorder=1)

	# Draw nodes
	xs = [x for x, _ in points]
	ys = [y for _, y in points]
	ax.scatter(xs, ys, s=60, color="black", zorder=2)

	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle=":", linewidth=0.8, color="0.9")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title("8-direction grid connectivity")
	plt.show()


if __name__ == "__main__":
	main()
