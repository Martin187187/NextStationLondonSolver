from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable, Literal, Optional, Tuple, Union


NodeType = Literal["triangle", "square", "pentagon", "circle", "any"]
Action = Union[NodeType, Literal["junction"]]
Point = tuple[int, int]
RegionId = str
NodeValue = Union[NodeType, Tuple[NodeType, bool]]


def _split_node_value(value: NodeValue) -> tuple[NodeType, bool]:
	if isinstance(value, tuple):
		node_type, special = value
		return node_type, bool(special)
	return value, False


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


def build_adjacency(edges: set[tuple[Point, Point]]) -> dict[Point, set[Point]]:
	adj: dict[Point, set[Point]] = {}
	for a, b in edges:
		adj.setdefault(a, set()).add(b)
		adj.setdefault(b, set()).add(a)
	return adj


def _orientation(a: Point, b: Point, c: Point) -> int:
	"""Return orientation of (a,b,c): 0 collinear, 1 clockwise, 2 counterclockwise."""
	ax, ay = a
	bx, by = b
	cx, cy = c
	val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
	if val == 0:
		return 0
	return 1 if val > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
	"""True if b lies on segment a-c (assuming collinear)."""
	ax, ay = a
	bx, by = b
	cx, cy = c
	return min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy)


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
	"""Return True if segments a1-a2 and b1-b2 intersect (including collinear overlap)."""
	o1 = _orientation(a1, a2, b1)
	o2 = _orientation(a1, a2, b2)
	o3 = _orientation(b1, b2, a1)
	o4 = _orientation(b1, b2, a2)

	# General case
	if o1 != o2 and o3 != o4:
		return True

	# Special cases
	if o1 == 0 and _on_segment(a1, b1, a2):
		return True
	if o2 == 0 and _on_segment(a1, b2, a2):
		return True
	if o3 == 0 and _on_segment(b1, a1, b2):
		return True
	if o4 == 0 and _on_segment(b1, a2, b2):
		return True

	return False


def segment_would_intersect_any(existing_segments: list[tuple[Point, Point]], a: Point, b: Point) -> bool:
	"""True if segment a-b intersects any existing segment (except shared endpoints).

	Sharing an endpoint is allowed; overlapping the same segment is forbidden.
	"""
	for s1, s2 in existing_segments:
		if segments_intersect(a, b, s1, s2):
			# Disallow reusing the exact same segment.
			if {a, b} == {s1, s2}:
				return True
			# Allow touching at endpoints.
			if a in (s1, s2) or b in (s1, s2):
				continue
			return True
	return False


def path_would_self_intersect(path: list[Point], nxt: Point) -> bool:
	"""Check if adding edge (path[-1] -> nxt) would make the path self-intersect."""
	if len(path) < 2:
		return False

	new_a = path[-1]
	new_b = nxt

	# Compare against all prior segments except the last one (shares endpoint new_a).
	for i in range(len(path) - 2):
		seg_a = path[i]
		seg_b = path[i + 1]
		if segments_intersect(new_a, new_b, seg_a, seg_b):
			# Allow touching only at the shared endpoint new_a.
			if new_a in (seg_a, seg_b):
				continue
			return True

	return False


def score_path(path: list[Point]) -> int:
	"""Score a path.

	Score = (# distinct regions visited by the path)
	        * (maximum number of nodes the path has in any single region).
	"""
	if not path:
		return 0

	regions = [region_for_point(p) for p in path]
	visited_regions = set(regions)
	counts = Counter(regions)
	max_visits_in_one_region = max(counts.values())
	return len(visited_regions) * max_visits_in_one_region


def score_two_paths(path1: list[Point], path2: Optional[list[Point]]) -> int:
	"""Score the union of visited nodes across both paths."""
	visited: set[Point] = set(path1)
	if path2:
		visited.update(path2)
	return score_path(list(visited))


def normalize_action(action: str) -> NodeType:
	a = action.strip().lower()
	synonyms = {
		"sphere": "circle",
		"ball": "circle",
	}
	a = synonyms.get(a, a)
	allowed: set[str] = {"triangle", "square", "pentagon", "circle", "any"}
	if a not in allowed:
		raise ValueError(f"Unknown action/type: {action!r}. Allowed: {sorted(allowed)}")
	return a  # type: ignore[return-value]


def parse_action(action: str) -> Action:
	a = action.strip().lower()
	if a == "junction":
		return "junction"
	return normalize_action(a)


def action_matches_node(action_type: NodeType, node_type: NodeType) -> bool:
	"""Return True if consuming `action_type` may move onto `node_type`.

	Rules:
	- Action 'any' matches every node type.
	- Node type 'any' matches every action type.
	- Otherwise, exact match.
	"""
	return action_type == "any" or node_type == "any" or action_type == node_type


def all_two_path_solutions_ordered(
	*,
	start: Point,
	actions: list[str],
	adjacency: dict[Point, set[Point]],
	node_types: dict[Point, NodeType],
	allow_revisit: bool = False,
	can_skip_action: bool = False,
	forbid_self_intersections: bool = True,
) -> list[tuple[list[Point], list[Point]]]:
	"""Return all solutions for ordered actions, supporting a single 'junction'.

	- Before junction: extend path1.
	- On 'junction': create path2 starting from ANY node visited so far (in path1).
	- The next *matched* action after junction must extend path2.
	- Afterwards each action may extend either path.
	- No segment intersections across BOTH paths (except shared endpoints).
	"""
	tokens: list[Action] = [parse_action(a) for a in actions]
	if start not in node_types:
		raise ValueError(f"Start node {start} not present in nodes")

	solutions: list[tuple[list[Point], list[Point]]] = []

	def dfs(
		idx: int,
		path1: list[Point],
		path2: Optional[list[Point]],
		segments: list[tuple[Point, Point]],
		force_next_on_path2: bool,
	) -> None:
		if idx == len(tokens):
			solutions.append((path1.copy(), (path2 or []).copy()))
			return

		token = tokens[idx]

		# Optionally skip this action.
		if can_skip_action:
			# Skipping does not satisfy the "first action after junction must be on path2" rule.
			dfs(idx + 1, path1, path2, segments, force_next_on_path2)

		if token == "junction":
			# Only allow creating one second path.
			if path2 is not None:
				return
			for join_point in sorted(set(path1)):
				dfs(idx + 1, path1, [join_point], segments, True)
			return

		target_type = token

		def extend(which: int) -> None:
			nonlocal path1, path2, segments, force_next_on_path2
			if which == 1:
				current_path = path1
			else:
				if path2 is None:
					return
				current_path = path2

			tail = current_path[-1]
			for nxt in sorted(adjacency.get(tail, set())):
				nxt_type = node_types.get(nxt)
				if nxt_type is None or not action_matches_node(target_type, nxt_type):
					continue
				if not allow_revisit and nxt in current_path:
					continue
				if forbid_self_intersections and segment_would_intersect_any(segments, tail, nxt):
					continue

				current_path.append(nxt)
				segments.append((tail, nxt))
				next_force = force_next_on_path2
				if which == 2 and force_next_on_path2:
					next_force = False
				dfs(idx + 1, path1, path2, segments, next_force)
				segments.pop()
				current_path.pop()

		# Decide which path to extend.
		if path2 is None:
			extend(1)
			return
		if force_next_on_path2:
			extend(2)
			return
		extend(1)
		extend(2)

	dfs(0, [start], None, [], False)
	return solutions


def all_two_path_solutions_unordered(
	*,
	start: Point,
	actions: list[str],
	adjacency: dict[Point, set[Point]],
	node_types: dict[Point, NodeType],
	allow_revisit: bool = False,
	can_skip_action: bool = False,
	forbid_self_intersections: bool = True,
) -> list[tuple[list[Point], list[Point]]]:
	"""Return all solutions when action order does NOT matter, supporting a single 'junction'.

	Interpretation for unordered mode:
	- You may consume remaining actions in any order.
	- When 'junction' is consumed, path2 is created from any node visited so far in path1.
	- The next *matched/moved* action after creating path2 must extend path2.
	- Afterwards, each consumed action may extend either path.
	- No segment intersections across BOTH paths (except shared endpoints).

	Skipping (`can_skip_action=True`) consumes an action without moving, and does not satisfy the
	"first action after junction goes to path2" rule.
	"""
	tokens: list[Action] = [parse_action(a) for a in actions]
	junction_count = sum(1 for t in tokens if t == "junction")
	if junction_count > 1:
		raise ValueError("Only one 'junction' action is supported")

	remaining = Counter([t for t in tokens if t != "junction"])
	junction_remaining = junction_count == 1

	if start not in node_types:
		raise ValueError(f"Start node {start} not present in nodes")

	solutions: list[tuple[list[Point], list[Point]]] = []
	seen_states: set[
		tuple[
			tuple[Point, ...],
			tuple[Point, ...],
			tuple[tuple[NodeType, int], ...],
			bool,
			bool,
		]
	] = set()

	def dfs(
		path1: list[Point],
		path2: Optional[list[Point]],
		segments: list[tuple[Point, Point]],
		remaining_counts: Counter,
		junction_left: bool,
		force_next_on_path2: bool,
	) -> None:
		state = (
			tuple(path1),
			tuple(path2 or []),
			tuple(sorted(remaining_counts.items())),
			junction_left,
			force_next_on_path2,
		)
		if state in seen_states:
			return
		seen_states.add(state)

		if (not remaining_counts) and (not junction_left):
			solutions.append((path1.copy(), (path2 or []).copy()))
			return

		# Optionally skip an action (consume it without moving).
		if can_skip_action:
			for t in sorted(list(remaining_counts.keys())):
				remaining_counts[t] -= 1
				removed = False
				if remaining_counts[t] == 0:
					del remaining_counts[t]
					removed = True
				dfs(path1, path2, segments, remaining_counts, junction_left, force_next_on_path2)
				# restore
				if removed:
					remaining_counts[t] = 1
				else:
					remaining_counts[t] += 1
			if junction_left:
				# Skip junction: consume it with no effect.
				dfs(path1, path2, segments, remaining_counts, False, force_next_on_path2)

		# Consume junction (create path2).
		if junction_left and path2 is None:
			for join_point in sorted(set(path1)):
				dfs(path1, [join_point], segments, remaining_counts, False, True)

		# Consume a node-type action by moving on one of the paths.
		if not remaining_counts:
			return

		visited_global: set[Point] = set(path1)
		if path2:
			visited_global.update(path2)

		# Decide which path(s) we are allowed to extend.
		if path2 is None:
			allowed_paths = (1,)
		elif force_next_on_path2:
			allowed_paths = (2,)
		else:
			allowed_paths = (1, 2)

		for target_type in sorted(list(remaining_counts.keys())):
			for which in allowed_paths:
				current_path = path1 if which == 1 else path2
				if current_path is None:
					continue
				tail = current_path[-1]
				for nxt in sorted(adjacency.get(tail, set())):
					nxt_type = node_types.get(nxt)
					if nxt_type is None or not action_matches_node(target_type, nxt_type):
						continue
					if not allow_revisit:
						# Disallow visiting any node already used by either path,
						# except that path2 may *start* from a path1 node.
						if nxt in visited_global:
							continue
					if forbid_self_intersections and segment_would_intersect_any(segments, tail, nxt):
						continue

					remaining_counts[target_type] -= 1
					removed = False
					if remaining_counts[target_type] == 0:
						del remaining_counts[target_type]
						removed = True

					current_path.append(nxt)
					segments.append((tail, nxt))
					next_force = force_next_on_path2
					if which == 2 and force_next_on_path2:
						next_force = False
					dfs(path1, path2, segments, remaining_counts, junction_left, next_force)
					segments.pop()
					current_path.pop()

					# restore counts
					if removed:
						remaining_counts[target_type] = 1
					else:
						remaining_counts[target_type] += 1

	dfs([start], None, [], remaining, junction_remaining, False)
	return solutions


def all_paths_by_actions(
	*,
	start: Point,
	actions: Iterable[str],
	adjacency: dict[Point, set[Point]],
	node_types: dict[Point, NodeType],
	allow_revisit: bool = False,
	order_matters: bool = False,
	can_skip_action: bool = False,
	forbid_self_intersections: bool = True,
) -> list[list[Point]]:
	"""Return all paths that start at `start` and then follow `actions` by node type.

	If `order_matters` is True, actions are consumed in the given order.
	If `order_matters` is False, actions are matched in any order (multiset).

	For each consumed action (node type), we move to a neighbor whose node type matches.
	The returned paths include the start node as the first element.
	"""
	# Disallow junction in this single-path solver.
	if any(a.strip().lower() == "junction" for a in actions):
		raise ValueError("'junction' requires the two-path solver")

	actions_norm = [normalize_action(a) for a in actions]
	if start not in node_types:
		raise ValueError(f"Start node {start} not present in nodes")

	results: list[list[Point]] = []

	if order_matters:
		def dfs_ordered(current: Point, step: int, path: list[Point]) -> None:
			if step == len(actions_norm):
				results.append(path.copy())
				return

			# Optionally skip an action (consume it without moving).
			if can_skip_action:
				dfs_ordered(current, step + 1, path)

			target_type = actions_norm[step]
			for nxt in sorted(adjacency.get(current, set())):
				nxt_type = node_types.get(nxt)
				if nxt_type is None or not action_matches_node(target_type, nxt_type):
					continue
				if not allow_revisit and nxt in path:
					continue
				if forbid_self_intersections and path_would_self_intersect(path, nxt):
					continue
				path.append(nxt)
				dfs_ordered(nxt, step + 1, path)
				path.pop()

		dfs_ordered(start, 0, [start])
		return results

	remaining = Counter(actions_norm)
	seen_states: set[tuple[Point, tuple[tuple[NodeType, int], ...], tuple[Point, ...]]] = set()

	def dfs_unordered(current: Point, remaining_counts: Counter, path: list[Point]) -> None:
		state = (current, tuple(sorted(remaining_counts.items())), tuple(path))
		if state in seen_states:
			return
		seen_states.add(state)

		if not remaining_counts:
			results.append(path.copy())
			return

		# Optionally skip an action (drop any one remaining required type without moving).
		if can_skip_action:
			for t in sorted(list(remaining_counts.keys())):
				remaining_counts[t] -= 1
				removed = False
				if remaining_counts[t] == 0:
					del remaining_counts[t]
					removed = True
				dfs_unordered(current, remaining_counts, path)
				# restore
				if removed:
					remaining_counts[t] = 1
				else:
					remaining_counts[t] += 1

		for nxt in sorted(adjacency.get(current, set())):
			nxt_type = node_types.get(nxt)
			if nxt_type is None:
				continue
			if not allow_revisit and nxt in path:
				continue
			if forbid_self_intersections and path_would_self_intersect(path, nxt):
				continue

			# Moving onto a node can consume ANY remaining action that matches it.
			# This includes:
			# - consuming 'any' action for any node
			# - consuming any specific action when moving onto an 'any' node
			# - consuming the node's own type action (exact match)
			consumable_types = [t for t in remaining_counts.keys() if action_matches_node(t, nxt_type)]
			for consume_t in sorted(consumable_types):
				if remaining_counts.get(consume_t, 0) <= 0:
					continue
				remaining_counts[consume_t] -= 1
				removed = False
				if remaining_counts[consume_t] == 0:
					del remaining_counts[consume_t]
					removed = True
				path.append(nxt)
				dfs_unordered(nxt, remaining_counts, path)
				path.pop()
				# restore
				if removed:
					remaining_counts[consume_t] = 1
				else:
					remaining_counts[consume_t] += 1

	dfs_unordered(start, remaining, [start])
	return results


def main(*, plot: bool) -> None:
	# Optional: print all possible paths that match a sequence of node types.
	# Example:
	#   START_NODE = (0, 0)
	#   ACTIONS = ["sphere", "circle"]
	# Toggle:
	# - ORDER_MATTERS=True  -> actions must be matched in the given order
	# - ORDER_MATTERS=False -> actions can be matched in any order
	# - CAN_SKIP_ACTION=True -> actions are optional (the solver may skip them)
	# - FORBID_SELF_INTERSECTIONS=True -> paths whose edges cross are rejected
	ORDER_MATTERS = False
	CAN_SKIP_ACTION = False
	FORBID_SELF_INTERSECTIONS = True
	START_NODE: Optional[Point] = (0, 0)
	ACTIONS: list[str] = ["circle", "square", "pentagon", "pentagon", "any", "junction"]

	# Define your nodes here (integer grid coordinates) with their type inline.
	# Example: (x, y): "triangle". Valid types: triangle, square, pentagon, circle, any
	# You can optionally mark nodes as special:
	#   (x, y): ("triangle", True)
	nodes: dict[Point, NodeValue] = {
		(0, 0): "triangle",
		(1, 0): "square",
		(3, 0): "pentagon",
		(4, 0): ("circle", True),
		(5, 0): "triangle",
		(6, 0): ("any", True),
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
		(9, 3): ("triangle", True),
		(0, 4): "pentagon",
		(2, 4): "square",
		(4, 4): "circle",
		(7, 4): "circle",
		(1, 5): "triangle",
		(2, 5): "square",
		(4, 5): "pentagon",
		(5, 5): "square",
		(8, 5): "pentagon",
		(0, 6): ("square", True),
		(2, 6): "pentagon",
		(4, 6): "triangle",
		(5, 6): ("any", True),
		(6, 6): "circle",
		(7, 6): "circle",
		(9, 6): "square",
		(0, 7): "circle",
		(3, 7): "triangle",
		(6, 7): "square",
		(9, 7): "triangle",
		(1, 8): "pentagon",
		(3, 8): "square",
		(6, 8): ("pentagon", True),
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

	node_types: dict[Point, NodeType] = {}
	special_points: set[Point] = set()
	for point, value in nodes.items():
		node_type, special = _split_node_value(value)
		node_types[point] = node_type
		if special:
			special_points.add(point)

	points: set[Point] = set(node_types.keys())

	edges = build_8_neighbor_edges(points)
	adjacency = build_adjacency(edges)

	if START_NODE is not None and ACTIONS:
		print(f"Start: {START_NODE}")
		print(f"Actions: {ACTIONS}")

		TOP_K = 5
		if any(a.strip().lower() == "junction" for a in ACTIONS):
			if ORDER_MATTERS:
				solutions = all_two_path_solutions_ordered(
					start=START_NODE,
					actions=ACTIONS,
					adjacency=adjacency,
					node_types=node_types,
					allow_revisit=False,
					can_skip_action=CAN_SKIP_ACTION,
					forbid_self_intersections=FORBID_SELF_INTERSECTIONS,
				)
			else:
				solutions = all_two_path_solutions_unordered(
					start=START_NODE,
					actions=ACTIONS,
					adjacency=adjacency,
					node_types=node_types,
					allow_revisit=False,
					can_skip_action=CAN_SKIP_ACTION,
					forbid_self_intersections=FORBID_SELF_INTERSECTIONS,
				)
			print(f"Paths found: {len(solutions)}")
			scored = [(score_two_paths(p1, p2 if p2 else None), p1, p2) for (p1, p2) in solutions]
			scored.sort(key=lambda t: (-t[0], t[1], t[2]))
			for score, p1, p2 in scored[:TOP_K]:
				print(f"path1={p1} path2={p2} -> score={score}")
		else:
			paths = all_paths_by_actions(
				start=START_NODE,
				actions=ACTIONS,
				adjacency=adjacency,
				node_types=node_types,
				allow_revisit=False,
				order_matters=ORDER_MATTERS,
				can_skip_action=CAN_SKIP_ACTION,
				forbid_self_intersections=FORBID_SELF_INTERSECTIONS,
			)
			print(f"Paths found: {len(paths)}")
			scored_paths = [(score_path(p), p) for p in paths]
			scored_paths.sort(key=lambda sp: (-sp[0], sp[1]))
			for score, path in scored_paths[:TOP_K]:
				print(f"{path} -> score={score}")

	if not plot:
		return

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
	for point, point_type in node_types.items():
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

	# Special nodes: add a simple halo ring overlay
	if special_points:
		xs = [x for x, _ in special_points]
		ys = [y for _, y in special_points]
		ax.scatter(
			xs,
			ys,
			s=260,
			marker="o",
			facecolors="none",
			edgecolors="black",
			linewidths=2.0,
			zorder=3,
		)

	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle=":", linewidth=0.8, color="0.9")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title("8-direction grid connectivity")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--plot", action="store_true", help="Show the matplotlib plot")
	args = parser.parse_args()
	main(plot=args.plot)
