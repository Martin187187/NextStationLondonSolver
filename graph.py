from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import heapq
from typing import Iterable, Literal, Optional, Tuple, Union


NodeType = Literal["triangle", "square", "pentagon", "circle", "any"]
Action = Union[NodeType, Literal["junction"]]
Point = tuple[int, int]
RegionId = str
NodeValue = Union[NodeType, Tuple[NodeType, bool]]


def normalize_edge(a: Point, b: Point) -> tuple[Point, Point]:
	"""Normalize an undirected edge so (a,b) and (b,a) compare equal."""
	return (a, b) if a <= b else (b, a)


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



def score_path(
	path: list[Point],
	bonus_edges: Optional[dict[tuple[Point, Point], int]] = None,
) -> int:
	"""Score a path.

	Base score = (# distinct regions visited by the path)
	             * (maximum number of nodes the path has in any single region).

	If `bonus_edges` is provided, the edge bonus is added on top.
	"""
	if not path:
		return 0

	regions = [region_for_point(p) for p in path]
	visited_regions = set(regions)
	counts = Counter(regions)
	max_visits_in_one_region = max(counts.values())
	base = len(visited_regions) * max_visits_in_one_region
	return base + edge_bonus_for_path(path, bonus_edges or {})


def edge_bonus_for_path(path: list[Point], bonus_edges: dict[tuple[Point, Point], int]) -> int:
	"""Sum bonus points for edges traversed in `path`.

	Edges are treated as undirected and should be provided normalized via `normalize_edge`.
	"""
	if len(path) < 2 or not bonus_edges:
		return 0
	bonus = 0
	for i in range(len(path) - 1):
		bonus += bonus_edges.get(normalize_edge(path[i], path[i + 1]), 0)
	return bonus


def score_path_with_edge_bonus(path: list[Point], bonus_edges: dict[tuple[Point, Point], int]) -> int:
	# Backwards-compatible helper; prefer calling `score_path(path, bonus_edges=...)`.
	return score_path(path, bonus_edges=bonus_edges)


def score_two_paths_with_edge_bonus(
	path1: list[Point],
	path2: Optional[list[Point]],
	bonus_edges: dict[tuple[Point, Point], int],
) -> int:
	"""Base score (union of visited nodes) + bonus for unique traversed edges across both paths."""
	base = score_two_paths(path1, path2)
	if not bonus_edges:
		return base
	used_edges: set[tuple[Point, Point]] = set()
	for path in (path1, path2 or []):
		for i in range(len(path) - 1):
			used_edges.add(normalize_edge(path[i], path[i + 1]))
	bonus = sum(bonus_edges.get(e, 0) for e in used_edges)
	return base + bonus



def score_two_paths(
	path1: list[Point],
	path2: Optional[list[Point]],
	bonus_edges: Optional[dict[tuple[Point, Point], int]] = None,
) -> int:
	"""Score the union of visited nodes across both paths.

	If `bonus_edges` is provided, the bonus is computed over the unique edges used
	across both paths (same behavior as `score_two_paths_with_edge_bonus`).
	"""
	visited: set[Point] = set(path1)
	if path2:
		visited.update(path2)
	base = score_path(list(visited))
	if not bonus_edges:
		return base
	used_edges: set[tuple[Point, Point]] = set()
	for path in (path1, path2 or []):
		for i in range(len(path) - 1):
			used_edges.add(normalize_edge(path[i], path[i + 1]))
	bonus = sum(bonus_edges.get(e, 0) for e in used_edges)
	return base + bonus


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


_ACTION_ORDER: tuple[NodeType, ...] = ("triangle", "square", "pentagon", "circle", "any")
_ACTION_INDEX: dict[NodeType, int] = {t: i for i, t in enumerate(_ACTION_ORDER)}


@dataclass(frozen=True)
class _PrecomputedGraph:
	points: tuple[Point, ...]
	point_to_idx: dict[Point, int]
	node_type_by_idx: tuple[NodeType, ...]
	adj: tuple[tuple[tuple[int, int], ...], ...]  # per node: ((nbr_idx, edge_idx), ...)
	edge_points: tuple[tuple[Point, Point], ...]  # edge_idx -> (a_point, b_point) normalized
	intersect_mask_by_edge: tuple[int, ...]  # edge_idx -> bitmask of conflicting edges
	bonus_value_by_edge: tuple[int, ...]  # edge_idx -> bonus value
	bonus_edge_mask: int
	region_masks: tuple[int, ...]  # region_idx -> bitmask of points in region


def _build_precomputed_graph(
	*,
	points: set[Point],
	adjacency: dict[Point, set[Point]],
	node_types: dict[Point, NodeType],
	bonus_edges: dict[tuple[Point, Point], int],
) -> _PrecomputedGraph:
	idx_points = tuple(sorted(points))
	point_to_idx = {p: i for i, p in enumerate(idx_points)}

	node_type_by_idx = tuple(node_types[p] for p in idx_points)

	# Edge indexing (normalized, undirected)
	edges_norm = set()
	for a, nbrs in adjacency.items():
		for b in nbrs:
			edges_norm.add(normalize_edge(a, b))
	edge_points = tuple(sorted(edges_norm))
	edge_to_idx = {e: i for i, e in enumerate(edge_points)}

	# Adjacency with edge ids for fast transitions
	adj_idx: list[list[tuple[int, int]]] = [[] for _ in idx_points]
	for a in idx_points:
		ai = point_to_idx[a]
		for b in adjacency.get(a, set()):
			bi = point_to_idx[b]
			ei = edge_to_idx[normalize_edge(a, b)]
			adj_idx[ai].append((bi, ei))
	for i in range(len(adj_idx)):
		adj_idx[i].sort(key=lambda t: (idx_points[t[0]], t[1]))
	adj = tuple(tuple(v) for v in adj_idx)

	# Precompute disallowed edge intersections.
	# Two edges conflict if their segments intersect and they do NOT share an endpoint.
	# Reusing the same undirected segment is also treated as a conflict.
	conf_masks = [0] * len(edge_points)
	for i, (a1, a2) in enumerate(edge_points):
		for j in range(i + 1, len(edge_points)):
			b1, b2 = edge_points[j]
			if not segments_intersect(a1, a2, b1, b2):
				continue
			# Allow touching at endpoints.
			if len({a1, a2} & {b1, b2}) > 0:
				continue
			conf_masks[i] |= 1 << j
			conf_masks[j] |= 1 << i

	# Bonus edges mask + per-edge value
	bonus_value_by_edge = [0] * len(edge_points)
	bonus_edge_mask = 0
	for e, pts in bonus_edges.items():
		ei = edge_to_idx.get(normalize_edge(*e))
		if ei is None:
			continue
		bonus_value_by_edge[ei] = int(pts)
		bonus_edge_mask |= 1 << ei

	# Regions: build masks for fast base-score from visited bitmask
	region_ids = sorted({region_for_point(p) for p in idx_points})
	region_index = {rid: i for i, rid in enumerate(region_ids)}
	region_masks = [0] * len(region_ids)
	for p in idx_points:
		ri = region_index[region_for_point(p)]
		region_masks[ri] |= 1 << point_to_idx[p]

	return _PrecomputedGraph(
		points=idx_points,
		point_to_idx=point_to_idx,
		node_type_by_idx=node_type_by_idx,
		adj=adj,
		edge_points=edge_points,
		intersect_mask_by_edge=tuple(conf_masks),
		bonus_value_by_edge=tuple(bonus_value_by_edge),
		bonus_edge_mask=bonus_edge_mask,
		region_masks=tuple(region_masks),
	)


def _actions_to_counts(actions: Iterable[str]) -> tuple[int, int, int, int, int]:
	counts = [0, 0, 0, 0, 0]
	for a in actions:
		t = normalize_action(a)
		counts[_ACTION_INDEX[t]] += 1
	return (counts[0], counts[1], counts[2], counts[3], counts[4])


def _dec_counts(counts: tuple[int, int, int, int, int], idx: int) -> tuple[int, int, int, int, int]:
	# Small, hot helper; avoid list allocations.
	if idx == 0:
		return (counts[0] - 1, counts[1], counts[2], counts[3], counts[4])
	if idx == 1:
		return (counts[0], counts[1] - 1, counts[2], counts[3], counts[4])
	if idx == 2:
		return (counts[0], counts[1], counts[2] - 1, counts[3], counts[4])
	if idx == 3:
		return (counts[0], counts[1], counts[2], counts[3] - 1, counts[4])
	return (counts[0], counts[1], counts[2], counts[3], counts[4] - 1)


def _counts_total(counts: tuple[int, int, int, int, int]) -> int:
	return counts[0] + counts[1] + counts[2] + counts[3] + counts[4]


def _node_matches_action(action_type: NodeType, node_type: NodeType) -> bool:
	# Mirror action_matches_node(), but inlined for speed.
	return action_type == "any" or node_type == "any" or action_type == node_type


def _popcount(x: int) -> int:
	"""Return number of set bits in x.

	Compatible with older Python versions that don't have int.bit_count().
	"""
	try:
		return x.bit_count()  # type: ignore[attr-defined]
	except AttributeError:
		return bin(x).count("1")


def _bonus_sum_for_used_edges(mask: int, bonus_value_by_edge: tuple[int, ...]) -> int:
	bonus = 0
	m = mask
	while m:
		lsb = m & -m
		ei = (lsb.bit_length() - 1)
		bonus += bonus_value_by_edge[ei]
		m ^= lsb
	return bonus


def _base_score_for_visited(visited_mask: int, region_masks: tuple[int, ...]) -> int:
	if visited_mask == 0:
		return 0
	regions_visited = 0
	max_in_one = 0
	for rm in region_masks:
		c = _popcount(visited_mask & rm)
		if c:
			regions_visited += 1
			if c > max_in_one:
				max_in_one = c
	return regions_visited * max_in_one


def best_two_path_solutions_unordered_topk(
	*,
	start: Point,
	actions: list[str],
	adjacency: dict[Point, set[Point]],
	node_types: dict[Point, NodeType],
	bonus_edges: dict[tuple[Point, Point], int],
	allow_revisit: bool = False,
	can_skip_action: bool = False,
	forbid_self_intersections: bool = True,
	top_k: int = 5,
) -> list[tuple[int, list[Point], list[Point]]]:
	"""Fast top-K solver for the unordered two-path (single junction) variant.

	This combines pathfinding and scoring, using:
	- Bitmask states for visited nodes + used edges
	- Precomputed edge intersection conflicts
	- Branch-and-bound to avoid enumerating all solutions

	Returns [(score, path1, path2)], where path2 may be [].
	"""
	if allow_revisit:
		# This fast solver is tuned for the default puzzle constraint; fall back to slow enumerator.
		solutions = all_two_path_solutions_unordered(
			start=start,
			actions=actions,
			adjacency=adjacency,
			node_types=node_types,
			allow_revisit=True,
			can_skip_action=can_skip_action,
			forbid_self_intersections=forbid_self_intersections,
		)
		scored = [
			(score_two_paths(p1, p2 if p2 else None, bonus_edges=bonus_edges), p1, p2)
			for (p1, p2) in solutions
		]
		scored.sort(key=lambda t: (-t[0], t[1], t[2]))
		return [(s, p1, p2) for (s, p1, p2) in scored[:top_k]]

	junction_count = sum(1 for a in actions if a.strip().lower() == "junction")
	if junction_count > 1:
		raise ValueError("Only one 'junction' action is supported")

	graph = _build_precomputed_graph(
		points=set(node_types.keys()),
		adjacency=adjacency,
		node_types=node_types,
		bonus_edges=bonus_edges,
	)
	if start not in graph.point_to_idx:
		raise ValueError(f"Start node {start} not present in nodes")

	counts = _actions_to_counts([a for a in actions if a.strip().lower() != "junction"])
	junction_left = junction_count == 1

	start_idx = graph.point_to_idx[start]
	start_visited = 1 << start_idx

	max_bonus = max(graph.bonus_value_by_edge) if graph.bonus_value_by_edge else 0

	@lru_cache(maxsize=None)
	def base_score(visited_mask: int) -> int:
		return _base_score_for_visited(visited_mask, graph.region_masks)

	@lru_cache(maxsize=None)
	def bonus_sum(used_edges_mask: int) -> int:
		return _bonus_sum_for_used_edges(used_edges_mask & graph.bonus_edge_mask, graph.bonus_value_by_edge)

	@lru_cache(maxsize=None)
	def upper_bound(
		visited_mask: int,
		used_edges_mask: int,
		remaining_counts: tuple[int, int, int, int, int],
		junction_left_local: bool,
	) -> int:
		# Very cheap optimistic bound: assume each remaining move can add a new region
		# and also add to the current max region occupancy; bonus uses max per move.
		rem = _counts_total(remaining_counts)
		if rem == 0 and not junction_left_local:
			return base_score(visited_mask) + bonus_sum(used_edges_mask)
		# Compute current region coverage + max occupancy
		regions_visited = 0
		max_in_one = 0
		for rm in graph.region_masks:
			c = _popcount(visited_mask & rm)
			if c:
				regions_visited += 1
				if c > max_in_one:
					max_in_one = c
		regions_max = min(len(graph.region_masks), regions_visited + rem)
		max_one_max = max_in_one + rem
		ub_base = regions_max * max_one_max
		return ub_base + bonus_sum(used_edges_mask) + rem * max_bonus

	# Keep best K in a min-heap of (score, path1_tuple, path2_tuple)
	heap: list[tuple[int, tuple[Point, ...], tuple[Point, ...]]] = []
	best_threshold = -10**18

	seen: set[tuple[int, int, int, tuple[int, int, int, int, int], bool, bool]] = set()

	def push_solution(path1: list[int], path2: Optional[list[int]], used_edges_mask: int, visited_mask: int) -> None:
		nonlocal best_threshold
		score = base_score(visited_mask) + bonus_sum(used_edges_mask)
		p1_pts = tuple(graph.points[i] for i in path1)
		p2_pts = tuple(graph.points[i] for i in (path2 or []))
		if len(heap) < top_k:
			heapq.heappush(heap, (score, p1_pts, p2_pts))
		else:
			if score <= heap[0][0]:
				return
			heapq.heapreplace(heap, (score, p1_pts, p2_pts))
		best_threshold = heap[0][0]

	def dfs(
		path1: list[int],
		path2: Optional[list[int]],
		used_edges_mask: int,
		visited_mask: int,
		remaining_counts: tuple[int, int, int, int, int],
		junction_left_local: bool,
		force_next_on_path2: bool,
	) -> None:
		nonlocal best_threshold

		# Prune by optimistic bound
		if len(heap) == top_k:
			ub = upper_bound(visited_mask, used_edges_mask, remaining_counts, junction_left_local)
			if ub <= best_threshold:
				return

		state_key = (
			path1[-1],
			path2[-1] if path2 else -1,
			used_edges_mask,
			remaining_counts,
			junction_left_local,
			force_next_on_path2,
		)
		if state_key in seen:
			return
		seen.add(state_key)

		if _counts_total(remaining_counts) == 0 and not junction_left_local:
			push_solution(path1, path2, used_edges_mask, visited_mask)
			return

		# Optionally skip an action.
		if can_skip_action:
			# Skip consumes one remaining type without moving.
			for ai, c in enumerate(remaining_counts):
				if c <= 0:
					continue
				dfs(
					path1,
					path2,
					used_edges_mask,
					visited_mask,
					_dec_counts(remaining_counts, ai),
					junction_left_local,
					force_next_on_path2,
				)
			if junction_left_local:
				# Skip junction (consume it with no effect)
				dfs(path1, path2, used_edges_mask, visited_mask, remaining_counts, False, force_next_on_path2)

		# Consume junction: create path2 from any node visited so far in path1.
		if junction_left_local and path2 is None:
			# Only allow join points on path1 (mirrors the original rules)
			for join_idx in sorted(set(path1)):
				dfs(path1, [join_idx], used_edges_mask, visited_mask, remaining_counts, False, True)

		# No moves possible if no remaining actions.
		if _counts_total(remaining_counts) == 0:
			return

		# Decide which path(s) we are allowed to extend.
		if path2 is None:
			allowed_paths = (1,)
		elif force_next_on_path2:
			allowed_paths = (2,)
		else:
			allowed_paths = (1, 2)

		# Expand in a heuristic order: prefer consuming specific actions before 'any'.
		action_indices = [0, 1, 2, 3, 4]
		for ai in action_indices:
			if remaining_counts[ai] <= 0:
				continue
			action_type = _ACTION_ORDER[ai]
			for which in allowed_paths:
				current_path = path1 if which == 1 else path2
				if current_path is None:
					continue
				tail = current_path[-1]
				for nxt, edge_idx in graph.adj[tail]:
					nxt_type = graph.node_type_by_idx[nxt]
					if not _node_matches_action(action_type, nxt_type):
						continue
					if not allow_revisit:
						# Disallow visiting any node already used by either path,
						# except that path2 may start from a path1 node (handled above).
						if (visited_mask >> nxt) & 1:
							continue
					if forbid_self_intersections:
						if (used_edges_mask >> edge_idx) & 1:
							continue
						if (used_edges_mask & graph.intersect_mask_by_edge[edge_idx]) != 0:
							continue

					next_counts = _dec_counts(remaining_counts, ai)
					next_used_edges = used_edges_mask | (1 << edge_idx)
					next_visited = visited_mask | (1 << nxt)

					current_path.append(nxt)
					next_force = force_next_on_path2
					if which == 2 and force_next_on_path2:
						next_force = False
					dfs(
						path1,
						path2,
						next_used_edges,
						next_visited,
						next_counts,
						junction_left_local,
						next_force,
					)
					current_path.pop()

	dfs([start_idx], None, 0, start_visited, counts, junction_left, False)

	# Emit results sorted like the old printing logic
	results = sorted(heap, key=lambda t: (-t[0], t[1], t[2]))
	return [(s, list(p1), list(p2)) for (s, p1, p2) in results]


def main(*, plot: bool, highlight_best: bool) -> None:
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
	START_NODE: Optional[Point] = (5, 2)
	ACTIONS: list[str] = ["square", "square", "pentagon", "pentagon", "triangle", "triangle", "any", "any", "junction"]

	# Hardcode bonus edges here.
	# Format: {((x1,y1),(x2,y2)): bonus_points, ...}
	# The edge is treated as undirected; order of endpoints does not matter.
	BONUS_EDGES: dict[tuple[Point, Point], int] = {
		((0, 4), (0, 6)): 2,
		((0, 6), (1, 5)): 2,
		((1, 8), (1, 5)): 2,
		((0, 7), (2, 5)): 2,
		((2, 6), (2, 5)): 2,
		((2, 5), (4, 5)): 2,
		((2, 4), (4, 6)): 2,
		((1, 5), (2, 6)): 2,
		((3, 3), (3, 7)): 2,
		((3, 3), (4, 4)): 2,
		((4, 3), (4, 4)): 2,
		((4, 3), (7, 6)): 2,
		((4, 5), (6, 3)): 2,
		((5, 2), (5, 5)): 2,
		((5, 5), (7, 3)): 2,
		((6, 3), (6, 6)): 2,
		((4, 4), (7, 4)): 2,
		((5, 6), (7, 4)): 2,
		((2, 4), (4, 4)): 2,
		((7, 4), (7, 6)): 2,
		((6, 6), (9, 3)): 2,
		((7, 4), (8, 5)): 2,
		((8, 2), (8, 5)): 2,
		((9, 3), (9, 6)): 2,
	}

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

	bonus_edges = {normalize_edge(a, b): pts for (a, b), pts in BONUS_EDGES.items()}
	unknown_bonus_edges = set(bonus_edges.keys()) - edges
	if unknown_bonus_edges:
		print(
			"Warning: some BONUS_EDGES are not present in the generated graph and will be ignored: "
			+ ", ".join(map(str, sorted(unknown_bonus_edges)))
		)

	best_paths: list[list[Point]] = []
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
				# Fast path: combine search + scoring with pruning (no full enumeration)
				scored = best_two_path_solutions_unordered_topk(
					start=START_NODE,
					actions=ACTIONS,
					adjacency=adjacency,
					node_types=node_types,
					bonus_edges=bonus_edges,
					allow_revisit=False,
					can_skip_action=CAN_SKIP_ACTION,
					forbid_self_intersections=FORBID_SELF_INTERSECTIONS,
					top_k=TOP_K,
				)
			print(f"Paths found (top {TOP_K}): {len(scored)}")
			if scored:
				_, best_p1, best_p2 = scored[0]
				best_paths = [best_p1] + ([best_p2] if best_p2 else [])
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
			scored_paths = [(score_path(p, bonus_edges=bonus_edges), p) for p in paths]
			scored_paths.sort(key=lambda sp: (-sp[0], sp[1]))
			if scored_paths:
				best_paths = [scored_paths[0][1]]
			for score, path in scored_paths[:TOP_K]:
				print(f"{path} -> score={score}")
	else:
		best_paths = []

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

	bonus_edge_set = set(bonus_edges.keys())

	# Draw edges (grey lines)
	for (x1, y1), (x2, y2) in edges:
		if ((x1, y1), (x2, y2)) in bonus_edge_set:
			continue
		ax.plot([x1, x2], [y1, y2], color="0.7", linewidth=1.0, zorder=1)

	# Highlight bonus edges on top
	for (x1, y1), (x2, y2) in sorted(bonus_edge_set & edges):
		# Keep the same base edge color; emphasize via thickness only.
		ax.plot([x1, x2], [y1, y2], color="0.7", linewidth=3.0, zorder=1.5)

	# Optionally highlight the best-scoring path(s) (thicker line, same color)
	if highlight_best and best_paths:
		for path in best_paths:
			if len(path) < 2:
				continue
			for i in range(len(path) - 1):
				(x1, y1) = path[i]
				(x2, y2) = path[i + 1]
				ax.plot([x1, x2], [y1, y2], color="tab:red", linewidth=5.0, zorder=1.6)

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
	parser.add_argument(
		"--highlight-best",
		action="store_true",
		help="Overlay the best-scoring path on the plot",
	)
	args = parser.parse_args()
	main(plot=args.plot, highlight_best=args.highlight_best)
