#!/usr/bin/env python3
"""
Screw Sequence Monitor - Reads and displays screw sequence tracking from the 3D scene.

This script polls the 3D scene backend to monitor the screw sequence in real-time.
It displays which screw is being worked on, the sequence progress, and validates
whether the correct order is being followed.

Expected order: BL (bottom-left) → TR (top-right) → BR (bottom-right) → TL (top-left)

Usage:
    # Start the 3D scene backend first:
    cd 3d_scene && python 3dscene.py

    # Then in another terminal, run this monitor:
    python 3d_scene/sequence_from_distance_tool.py

Press Ctrl+C to stop monitoring.
"""

import time
import requests
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distance_tool_screw import get_distance_sync, BACKEND_URL


# Short names for display
SHORT_NAMES = {
    "top_left": "TL",
    "top_right": "TR",
    "bottom_left": "BL",
    "bottom_right": "BR",
}

# Expected sequence
EXPECTED_ORDER = ["bottom_left", "top_right", "bottom_right", "top_left"]


def get_screw_status():
    """Get current screw sequence status from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/screw/status", timeout=1.0)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def reset_sequence():
    """Reset the screw sequence on the backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/screw/reset", timeout=1.0)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def print_status_header():
    """Print the status display header."""
    print("\n" + "=" * 70)
    print("  🔩 SCREW SEQUENCE MONITOR")
    print("  Expected order: BL → TR → BR → TL (diagonal pattern)")
    print("=" * 70)


def format_sequence(sequence):
    """Format a sequence list for display."""
    if not sequence:
        return "None"
    return " → ".join([SHORT_NAMES.get(s, s) for s in sequence])


def print_screw_diagram(status):
    """Print a visual diagram of the case with screw states."""
    screws = status.get("screws", {})
    active = status.get("active_screw", None)
    next_expected = status.get("next_expected", None)

    def get_screw_char(pos):
        screw = screws.get(pos, {})
        if screw.get("is_tightened", False):
            return "✓"  # Completed
        elif pos == active:
            return "●"  # Active/being worked on
        elif pos == next_expected:
            return "◎"  # Next expected
        else:
            return "○"  # Pending

    def get_screw_color(pos):
        screw = screws.get(pos, {})
        if screw.get("is_tightened", False):
            return "\033[92m"  # Green
        elif pos == active:
            return "\033[93m"  # Yellow
        elif pos == next_expected:
            return "\033[94m"  # Blue
        else:
            return "\033[90m"  # Gray

    reset = "\033[0m"

    tl = f"{get_screw_color('top_left')}{get_screw_char('top_left')}{reset}"
    tr = f"{get_screw_color('top_right')}{get_screw_char('top_right')}{reset}"
    bl = f"{get_screw_color('bottom_left')}{get_screw_char('bottom_left')}{reset}"
    br = f"{get_screw_color('bottom_right')}{get_screw_char('bottom_right')}{reset}"

    print(f"  ┌─────────────────┐")
    print(f"  │  {tl} TL     TR {tr}  │")
    print(f"  │                 │")
    print(f"  │      CASE       │")
    print(f"  │                 │")
    print(f"  │  {bl} BL     BR {br}  │")
    print(f"  └─────────────────┘")


def print_status(status, distance):
    """Print current status in a formatted way."""
    # Clear screen (optional - comment out if you prefer scrolling output)
    # print("\033[H\033[J", end="")

    print("\r" + " " * 80, end="\r")  # Clear line

    state = status.get("current_state", "unknown")
    step = status.get("current_step", 0)
    active = status.get("active_screw", None)
    next_expected = status.get("next_expected", None)
    actual_seq = status.get("actual_sequence", [])
    is_correct = status.get("is_correct", True)
    completed = status.get("completed", False)
    errors = status.get("errors", [])
    frames_near = status.get("frames_near_3d", 0)
    frames_to_complete = status.get("frames_to_complete_3d", 40)

    # Color codes
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    # State indicator
    state_color = {
        "idle": "\033[90m",
        "approaching": CYAN,
        "screwing": YELLOW,
        "completed": GREEN,
    }.get(state, RESET)

    # Print compact status line
    active_str = SHORT_NAMES.get(active, "-") if active else "-"
    next_str = SHORT_NAMES.get(next_expected, "-") if next_expected else "Done"
    seq_str = format_sequence(actual_seq) if actual_seq else "None"
    dist_str = f"{distance:.1f}cm" if distance else "N/A"

    # Progress bar for screwing
    progress = ""
    if state == "screwing" and frames_to_complete > 0:
        pct = min(1.0, frames_near / frames_to_complete)
        bar_len = 10
        filled = int(pct * bar_len)
        progress = f" [{('█' * filled) + ('░' * (bar_len - filled))}]"

    # Status line
    if completed:
        if is_correct:
            print(f"{GREEN}✓ SEQUENCE COMPLETE - CORRECT ORDER!{RESET}")
        else:
            print(f"{RED}✗ SEQUENCE COMPLETE - WRONG ORDER!{RESET}")
        print(f"  Actual: {seq_str}")
        if errors:
            print(f"  Errors: {', '.join(errors)}")
    else:
        print(
            f"  State: {state_color}{state.upper():10}{RESET} | "
            f"Step: {step+1}/4 | "
            f"Active: {YELLOW}{active_str:3}{RESET} | "
            f"Next: {CYAN}{next_str:3}{RESET} | "
            f"Dist: {dist_str}{progress}"
        )

        if actual_seq:
            correctness = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
            print(f"  Progress: {seq_str} {correctness}")

        if errors and not is_correct:
            print(f"  {RED}Errors: {', '.join(errors[-3:])}{RESET}")


def main():
    """Main monitoring loop."""
    print_status_header()
    print("\nConnecting to backend at", BACKEND_URL, "...")

    # Check connection
    status = get_screw_status()
    if status is None:
        print(f"\n\033[91mError: Cannot connect to backend at {BACKEND_URL}\033[0m")
        print("Make sure the 3D scene server is running:")
        print("  cd 3d_scene && python 3dscene.py")
        return

    print("Connected! Monitoring screw sequence...\n")
    print("Legend: ○ pending  ◎ next expected  ● active  ✓ done")
    print("-" * 70)

    last_status = None
    last_diagram_time = 0

    try:
        while True:
            status = get_screw_status()
            distance = get_distance_sync()

            if status:
                # Print diagram when state changes significantly
                current_time = time.time()
                state_changed = (
                    last_status is None
                    or status.get("current_step") != last_status.get("current_step")
                    or status.get("current_state") != last_status.get("current_state")
                    or status.get("completed") != last_status.get("completed")
                )

                if state_changed or current_time - last_diagram_time > 5:
                    print()
                    print_screw_diagram(status)
                    last_diagram_time = current_time

                print_status(status, distance)
                last_status = status

                # Stop polling as frequently if completed
                if status.get("completed"):
                    time.sleep(2.0)
                else:
                    time.sleep(0.2)  # Poll every 200ms
            else:
                print("\r  Waiting for backend...", end="")
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

        # Print final summary
        if last_status and last_status.get("actual_sequence"):
            print("\n" + "=" * 70)
            print("FINAL SUMMARY")
            print("=" * 70)
            print(f"  Expected: {format_sequence(EXPECTED_ORDER)}")
            print(
                f"  Actual:   {format_sequence(last_status.get('actual_sequence', []))}"
            )
            print(f"  Correct:  {'Yes ✓' if last_status.get('is_correct') else 'No ✗'}")
            if last_status.get("errors"):
                print(f"  Errors:   {', '.join(last_status['errors'])}")
            print("=" * 70)


if __name__ == "__main__":
    main()
