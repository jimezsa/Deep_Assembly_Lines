"""
Battery insertion sequence FSM module.

This module provides finite state machine logic for tracking battery insertion
order and detecting errors in the assembly sequence.
"""

import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

EXPECTED_ORDER = [1, 2, 3, 4, 5, 6]

# FSM Parameters
IOU_THRESHOLD = 0.35
PROGRESS_THRESHOLD = 0.55
ENTER_MIN_FRAMES = 3
INSERT_MIN_FRAMES = 3  # Frames required to commit an insertion
EXIT_MAX_MISSES = 10  # Tolerance for brief occlusion

# Case/Battery class IDs from YOLO
CASE_ID = 1
BATTERY_ID = 3


# =============================================================================
# Helper Functions
# =============================================================================

def case_bbox(case_poly):
    """Get bounding box from case polygon."""
    x = case_poly[:, 0]
    y = case_poly[:, 1]
    return min(x), min(y), max(x), max(y)


def point_inside_polygon(point, polygon):
    """Check if point is inside polygon."""
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


def insertion_progress(centroid, case_poly):
    """Calculate insertion depth (0-1) based on Y position in case."""
    x1, y1, x2, y2 = case_bbox(case_poly)
    depth = (centroid[1] - y1) / (y2 - y1)
    return np.clip(depth, 0.0, 1.0)


def poly_to_mask(poly, frame_size):
    """Convert polygon to binary mask."""
    mask = np.zeros(frame_size, dtype=np.uint8)
    pts = poly.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_iou(a, b):
    """Calculate IoU between two binary masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0
    return inter / union


def split_case_into_slots(case_poly, frame_size):
    """Split case polygon into 6 slot regions.
    
    Physical layout:
    2 4 6  (top row)
    1 3 5  (bottom row)
    
    Returns:
        tuple: (slots dict, slot_masks dict)
    """
    x1, y1 = case_poly.min(axis=0)
    x2, y2 = case_poly.max(axis=0)

    slots = {}
    cols = np.linspace(x1, x2, 4)  # 4 points = 3 cols
    rows = np.linspace(y1, y2, 3)  # 3 points = 2 rows
    
    # Mapping: (col, row) -> slot_id
    layout = [
        (0, 1, 1),  # col 0, bottom row -> slot 1
        (0, 0, 2),  # col 0, top row -> slot 2
        (1, 1, 3),  # col 1, bottom row -> slot 3
        (1, 0, 4),  # col 1, top row -> slot 4
        (2, 1, 5),  # col 2, bottom row -> slot 5
        (2, 0, 6),  # col 2, top row -> slot 6
    ]

    for c, r, slot_id in layout:
        poly = np.array([
            [cols[c], rows[r]],      # top-left
            [cols[c+1], rows[r]],    # top-right
            [cols[c+1], rows[r+1]],  # bottom-right
            [cols[c], rows[r+1]],    # bottom-left
        ], dtype=np.float32)
        
        slots[slot_id] = poly

    slot_masks = {sid: poly_to_mask(poly, frame_size) for sid, poly in slots.items()}
    return slots, slot_masks


# =============================================================================
# FSM Classes
# =============================================================================

class SlotFSM:
    """Finite state machine for a single battery slot."""
    
    def __init__(self, slot_id):
        self.sid = slot_id
        self.state = "EMPTY"
        self.counter = 0
        self.miss = 0
        self.committed = False  # Once committed, immutable

    def update(self, present, progress, frame_idx):
        """Update FSM state.
        
        Args:
            present: Whether battery is detected in this slot
            progress: Insertion progress (0-1)
            frame_idx: Current frame index
            
        Returns:
            str: Event type ('INSERT', 'REMOVE', or None)
        """
        # Once committed, ignore all activity
        if self.committed:
            return None
        
        event = None

        if self.state == "EMPTY":
            if present:
                self.state = "ENTERING"
                self.counter = 1

        elif self.state == "ENTERING":
            if present:
                self.counter += 1
                if self.counter >= INSERT_MIN_FRAMES:
                    self.state = "INSERTED"
                    event = "INSERT"
            else:
                # Battery removed before commit
                self.state = "EMPTY"
                self.counter = 0

        elif self.state == "INSERTED":
            if not present:
                self.state = "EXITING"
                self.miss = 1
            else:
                self.miss = 0

        elif self.state == "EXITING":
            if present:
                self.state = "INSERTED"
                self.miss = 0
            else:
                self.miss += 1
                if self.miss > EXIT_MAX_MISSES:
                    self.state = "EMPTY"
                    self.counter = 0
                    event = "REMOVE"

        return event

    def mark_committed(self):
        """Lock this slot - no more state changes."""
        self.committed = True


class OrderFSM:
    """Finite state machine for tracking insertion order."""
    
    def __init__(self):
        self.expected_idx = 0
        self.sequence = []
        self.wrong_inserts = {}  # slot_id -> frame_idx
        self.current_error_msg = None
        self.success = False
        self.success_timestamp = None  # track when success was achieved
        self.success_duration = 5.0  # show success for 5 seconds

    def on_insert(self, slot_id, frame_idx):
        """Handle insertion event.
        
        Args:
            slot_id: Slot where battery was inserted
            frame_idx: Current frame index
            
        Returns:
            str: Result type ('COMMIT', 'WRONG', or None)
        """
        if self.expected_idx >= len(EXPECTED_ORDER):
            return None  # Sequence complete
        
        expected = EXPECTED_ORDER[self.expected_idx]

        if slot_id == expected:
            self.sequence.append(slot_id)
            self.expected_idx += 1
            self.current_error_msg = None  # Clear error on correct insertion
            
            # Check if sequence is complete
            if self.sequence == EXPECTED_ORDER:
                self.success = True
                if self.success_timestamp is None:  
                    import time  
                    self.success_timestamp = time.time()  # record when success happened
            
            return "COMMIT"
        else:
            # Wrong slot
            if slot_id not in self.wrong_inserts:
                self.wrong_inserts[slot_id] = frame_idx
                self.current_error_msg = f"WRONG SLOT: {slot_id} inserted (expected {expected})"
            return "WRONG"

    def on_remove(self, slot_id, frame_idx):
        """Handle removal event.
        
        Args:
            slot_id: Slot where battery was removed
            frame_idx: Current frame index
            
        Returns:
            str: Result type ('BACKTRACK' or None)
        """
        # Only care if it was the last committed slot
        if self.sequence and self.sequence[-1] == slot_id:
            self.sequence.pop()
            self.expected_idx -= 1
            self.success = False  # Reset success flag
            return "BACKTRACK"
        
        # Clear wrong insertion tracking
        if slot_id in self.wrong_inserts:
            self.wrong_inserts.pop(slot_id)
            # Clear error message if no more wrong inserts
            if not self.wrong_inserts:
                self.current_error_msg = None
        
        return None

    def get_slot_states(self):
        """Get current state of all slots for visualization.
        
        Returns:
            dict: slot_id -> state ('correct', 'wrong', 'empty')
        """
        states = {}
        
        # Mark correctly inserted slots as green
        for slot_id in self.sequence:
            states[slot_id] = 'correct'
        
        # Mark wrongly inserted slots as red
        for slot_id in self.wrong_inserts:
            states[slot_id] = 'wrong'
        
        # All other slots are empty
        for slot_id in range(1, 7):
            if slot_id not in states:
                states[slot_id] = 'empty'
        
        return states

    def should_show_success(self):
        """Check if success message should still be displayed (within 5 seconds).
        
        Returns:
            bool: True if success and within time window
        """
        if not self.success or self.success_timestamp is None:
            return False
        
        import time
        elapsed = time.time() - self.success_timestamp
        return elapsed < self.success_duration
# =============================================================================
# Battery Sequence Tracker
# =============================================================================

class BatterySequenceTracker:
    """Main tracker that processes YOLO results and maintains FSM state."""
    
    def __init__(self, frame_size=(720, 1280)):
        self.frame_size = frame_size
        self.slot_fsms = {sid: SlotFSM(sid) for sid in range(1, 7)}
        self.order_fsm = OrderFSM()
        self.slots = None
        self.slot_masks = None
        
    def process_yolo_frame(self, yolo_results, frame_idx):
        """Process YOLO detection results for a single frame.
        
        Args:
            yolo_results: YOLO results object from ultralytics
            frame_idx: Current frame index
            
        Returns:
            dict: Tracking state including slot states and error messages
        """
        # Extract case and battery detections
        case_poly = None
        batteries = []
        
        if yolo_results is not None:
            for r in yolo_results:
                if r.masks is None:
                    continue
                
                for j, poly in enumerate(r.masks.xy):
                    class_id = int(r.boxes.cls[j])
                    poly = np.asarray(poly, dtype=np.float32)
                    
                    if class_id == CASE_ID:
                        case_poly = poly
                    elif class_id == BATTERY_ID:
                        cx = float(np.mean(poly[:, 0]))
                        cy = float(np.mean(poly[:, 1]))
                        batteries.append({
                            "polygon": poly,
                            "centroid": (cx, cy)
                        })
        
        # Skip if no case detected
        if case_poly is None:
            return self._get_state()
        
        # Initialize slots on first case detection
        if self.slots is None:
            self.slots, self.slot_masks = split_case_into_slots(case_poly, self.frame_size)
        
        # Phase 1: Determine which batteries are inside case and their progress
        inside_batteries = []
        for b in batteries:
            centroid = b["centroid"]
            inside = point_inside_polygon(centroid, case_poly)
            if inside:
                progress = insertion_progress(centroid, case_poly)
                inside_batteries.append({
                    "poly": b["polygon"],
                    "progress": progress
                })
        
        # Phase 2: Assign batteries to slots using IoU
        frame_slots = {}
        for b in inside_batteries:
            battery_mask = poly_to_mask(b["poly"], self.frame_size)
            best_sid = None
            best_iou = 0.0
            
            for sid, smask in self.slot_masks.items():
                iou = mask_iou(battery_mask, smask)
                if iou > best_iou:
                    best_iou = iou
                    best_sid = sid
            
            if best_sid is None or best_iou < IOU_THRESHOLD:
                continue
            
            # Keep higher IoU if collision
            if best_sid in frame_slots:
                if best_iou <= frame_slots[best_sid]["iou"]:
                    continue
            
            frame_slots[best_sid] = {
                "progress": b["progress"],
                "iou": best_iou
            }
        
        # Phase 3: Update FSMs
        frame_events = []
        
        for sid, fsm in self.slot_fsms.items():
            present = sid in frame_slots
            progress = frame_slots[sid]["progress"] if present else 0.0
            
            event = fsm.update(present, progress, frame_idx)
            
            if event:
                frame_events.append((sid, event))
        
        # Process insertion events (correct ones first)
        insert_events = [(sid, ev) for sid, ev in frame_events if ev == "INSERT"]
        
        if insert_events:
            expected = EXPECTED_ORDER[self.order_fsm.expected_idx] if self.order_fsm.expected_idx < len(EXPECTED_ORDER) else None
            
            correct_insert = None
            wrong_inserts = []
            
            for sid, ev in insert_events:
                if sid == expected:
                    correct_insert = sid
                else:
                    wrong_inserts.append(sid)
            
            # Process correct insert first
            if correct_insert:
                result = self.order_fsm.on_insert(correct_insert, frame_idx)
                if result == "COMMIT":
                    self.slot_fsms[correct_insert].mark_committed()
            
            # Process wrong inserts
            for sid in wrong_inserts:
                if sid not in self.order_fsm.sequence:
                    self.order_fsm.on_insert(sid, frame_idx)
        
        # Process removal events
        remove_events = [(sid, ev) for sid, ev in frame_events if ev == "REMOVE"]
        for sid, ev in remove_events:
            self.order_fsm.on_remove(sid, frame_idx)
        
        return self._get_state()
    
    def _get_state(self):
        """Get current tracking state for visualization."""
        return {
            'slot_states': self.order_fsm.get_slot_states(),
            'error_msg': self.order_fsm.current_error_msg,
            'success': self.order_fsm.should_show_success(),
            'sequence': self.order_fsm.sequence.copy()
        }
    
    def reset(self):
        """Reset tracker to initial state."""
        self.slot_fsms = {sid: SlotFSM(sid) for sid in range(1, 7)}
        self.order_fsm = OrderFSM()
        self.slots = None
        self.slot_masks = None
        print("[BatteryFSM] Tracker reset")  # confirm reset
