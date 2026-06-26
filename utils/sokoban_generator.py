import argparse
import math
import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = Path(__file__).resolve().with_name("template.txt")
GENERATED_SPECS_DIR = PROJECT_ROOT / "generated_sokoban_specs"


def parse_board(board: Sequence[Sequence[str]]) -> Tuple[int, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]], Tuple[int, int]]:
    """Parse a board definition into the coordinates required by the LTS template."""
    rows = len(board)
    if rows == 0:
        raise ValueError("Board cannot be empty")

    cols = len(board[0])
    if rows != cols:
        raise ValueError(f"Board must be square. Got {rows}x{cols}")
    if any(len(row) != cols for row in board):
        raise ValueError("All board rows must have the same length")

    brick_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    obstacles: List[Tuple[int, int]] = []
    valid_cells = {"-", "B", "F", "O"}

    for r in range(rows):
        for c in range(cols):
            cell = str(board[r][c]).upper().strip()
            if cell not in valid_cells:
                raise ValueError(f"Invalid cell '{cell}' at ({r + 1}, {c + 1})")
            if cell == "B":
                if brick_pos is not None:
                    raise ValueError("Board must have exactly one brick 'B'")
                brick_pos = (r + 1, c + 1)
            elif cell == "F":
                if goal_pos is not None:
                    raise ValueError("Board must have exactly one goal 'F'")
                goal_pos = (r + 1, c + 1)
            elif cell == "O":
                obstacles.append((r + 1, c + 1))

    if brick_pos is None:
        raise ValueError("No brick 'B' found in board")
    if goal_pos is None:
        raise ValueError("No goal 'F' found in board")

    player_start = (1, 1)  # Fixed to match existing controller assumptions.
    if brick_pos == player_start or goal_pos == player_start:
        raise ValueError("Player start (1,1) cannot overlap with brick or goal")
    if player_start in obstacles:
        raise ValueError("Player start (1,1) cannot be an obstacle")

    return rows, player_start, brick_pos, obstacles, goal_pos


def read_board_csv(board_csv: str) -> List[List[str]]:
    """Read a board CSV file into a list-of-lists representation."""
    with open(board_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [[cell.strip() for cell in row] for row in reader if row]


def board_to_comment_lines(board: Sequence[Sequence[str]]) -> List[str]:
    """Build the board diagram comments used in the generated LTS file."""
    size = len(board)
    header = "//  " + " ".join(str(i) for i in range(1, size + 1))
    rows = [f"//{idx} " + " ".join(str(cell).strip() for cell in row) for idx, row in enumerate(board, start=1)]
    return [header, *rows]


def board_to_console_string(board: Sequence[Sequence[str]]) -> str:
    return "\n".join(line[3:] for line in board_to_comment_lines(board))


def create_random_board(size: int, obstacle_count: int = 3, rng: Optional[random.Random] = None) -> List[List[str]]:
    """Create a random valid board definition with one brick, one goal and obstacles."""
    if size < 3:
        raise ValueError("Board size must be at least 3 to keep the brick off the borders")

    rng = rng or random.Random()
    board = [["-" for _ in range(size)] for _ in range(size)]

    brick_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    if not brick_positions:
        raise ValueError("Board is too small to place a brick away from the borders")

    all_positions = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    max_obstacles = len(all_positions) - 2  # Reserve space for B and F.
    if max_obstacles < 0:
        raise ValueError("Board is too small to place brick and goal")
    obstacle_count = max(0, min(obstacle_count, max_obstacles))

    brick = rng.choice(brick_positions)
    remaining_positions = [pos for pos in all_positions if pos != brick]
    chosen = rng.sample(remaining_positions, 1 + obstacle_count)
    goal = chosen[0]
    obstacles = chosen[1:]

    board[brick[0]][brick[1]] = "B"
    board[goal[0]][goal[1]] = "F"
    for r, c in obstacles:
        board[r][c] = "O"

    return board


def board_signature(board: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    """Return a hashable signature so board uniqueness can be enforced."""
    return tuple(tuple(str(cell).strip().upper() for cell in row) for row in board)


def max_unique_configurations(size: int, obstacle_count: int) -> int:
    """Count the max unique boards with fixed player at (1,1), 1 B, 1 F, and N obstacles."""
    if size < 3:
        return 0
    available = (size * size) - 1  # Exclude fixed player cell (1,1).
    brick_positions = (size - 2) * (size - 2)
    if brick_positions < 1 or available < 2:
        return 0
    if obstacle_count < 0 or obstacle_count > available - 2:
        return 0
    return brick_positions * (available - 1) * math.comb(available - 2, obstacle_count)


def load_template() -> str:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return f.read()


def generate_lts(board: Sequence[Sequence[str]], output_file: Optional[Path] = None) -> str:
    """Generate full LTS specification from a board list-of-lists."""
    size, player_start, brick_pos, obstacles, goal_pos = parse_board(board)

    # Keep the board string representation in the diagram comment section.
    diagram_comment = "\n".join(board_to_comment_lines(board))

    # Build obstacles set
    obs_str = ", ".join(f"setPos[{r}][{c}]" for r, c in obstacles)
    if len(obstacles) > 1:
        obstacles_set = f"set Obstacles  = {{ {obs_str} }}"
        safe_positions = "Sets\{Obstacles, OutOfBoard}"
        illegal_position = "([p:Pieces].Obstacles -> IP | [p:Pieces].SafePositions -> IllegalPosition)"
    else:
        obstacles_set = ""
        safe_positions = "Sets\{OutOfBoard}"
        illegal_position = "([p:Pieces].SafePositions -> IllegalPosition)"

    template = load_template()
    # Use simple string replacement instead of .format() to avoid brace escaping issues
    filled = template
    filled = filled.replace("{diagram_comment}", diagram_comment)
    filled = filled.replace("{obstacles_set}", obstacles_set)
    filled = filled.replace("{safe_positions}", safe_positions)
    filled = filled.replace("{illegal_position}", illegal_position)
    filled = filled.replace("{goal_row}", str(goal_pos[0]))
    filled = filled.replace("{goal_col}", str(goal_pos[1]))
    filled = filled.replace("{size}", str(size))
    filled = filled.replace("{player_row}", str(player_start[0]))
    filled = filled.replace("{player_col}", str(player_start[1]))
    filled = filled.replace("{brick_row}", str(brick_pos[0]))
    filled = filled.replace("{brick_col}", str(brick_pos[1]))


    if output_file:
        with output_file.open("w", encoding="utf-8") as f:
            f.write(filled)
        print(f"Generated: {output_file}")

    return filled


def generate_board_defs_and_lts(board_size: int, configurations: int, obstacle_count: int = 3, seed: Optional[int] = None) -> None:
    """Generate random boards in memory and save matching LTS files in generated_sokoban_specs/."""
    if configurations <= 0:
        raise ValueError("Number of configurations must be greater than 0")

    GENERATED_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    size_folder = GENERATED_SPECS_DIR / f"sokoban-{board_size}-1"
    size_folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    max_unique = max_unique_configurations(board_size, obstacle_count)
    if configurations > max_unique:
        raise ValueError(
            f"Requested {configurations} configurations but only {max_unique} unique boards "
            f"exist for size={board_size}, obstacles={obstacle_count}."
        )

    # Enforce uniqueness: do not emit duplicated board layouts in the same batch.
    seen_signatures: set[Tuple[Tuple[str, ...], ...]] = set()

    idx = 1
    while idx <= configurations:
        board = create_random_board(board_size, obstacle_count=obstacle_count, rng=rng)
        signature = board_signature(board)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        # print(f"\n--- Configuration {idx}/{configurations} ---")
        # print(board_to_console_string(board))

        lts_file = size_folder / f"sokoban-{board_size}-1_{obstacle_count}obstacles_{idx}.lts"
        generate_lts(board, lts_file)
        idx += 1


def generate_default_batch() -> None:
    """Generate the default benchmark batch when no CLI arguments are provided."""
    for size in range(4, 8):
        for obstacles in range(0, 5):
            print(f"\n=== Size {size} | Obstacles {obstacles} | Configurations 10 ===")
            generate_board_defs_and_lts(
                board_size=size,
                configurations=10,
                obstacle_count=obstacles,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random Sokoban board defs and LTS specs.")
    parser.add_argument("board_size", type=int, nargs="?", help="Size N for an NxN board")
    parser.add_argument("configurations", type=int, nargs="?", help="Number of board configurations to generate")
    parser.add_argument("--obstacles", type=int, default=3, help="Number of obstacles per board (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible generation")
    args = parser.parse_args()

    print("Sokoban Board/LTS Generator")
    if args.board_size is None and args.configurations is None:
        print("No CLI input detected. Running default batch: sizes 4..7, obstacles 0..4, 10 configurations each.")
        generate_default_batch()
        print(f"\nDone. Files are in: {GENERATED_SPECS_DIR}")
        return

    if args.board_size is None or args.configurations is None:
        parser.error("Please provide both board_size and configurations, or provide no positional arguments for the default batch.")

    generate_board_defs_and_lts(
        board_size=args.board_size,
        configurations=args.configurations,
        obstacle_count=args.obstacles,
        seed=args.seed,
    )
    print(f"\nDone. Files are in: {GENERATED_SPECS_DIR}")


if __name__ == "__main__":
    main()