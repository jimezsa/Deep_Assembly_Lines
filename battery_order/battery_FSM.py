    
import pickle
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
# YOLO_CACHE = "135122071615_yolo_cache.pkl" # wrong 5th slot before 4th; not detected (lateral view)
YOLO_CACHE = "137322071489_yolo_cache.pkl" # wrong 5th slot before 4th; detected correctly (top view)
# YOLO_CACHE = "wrong_top_yolo_cache.pkl" # wrong 3rd slot before 2nd; detected correctly
# YOLO_CACHE = "rec7-89_yolo_cache.pkl" # correct full sequence; detected correctly

EXPECTED_ORDER = [1, 2, 3, 4, 5, 6]

FPS = 1.7
T_INSERT = 1.0
DT = 1.0 / FPS

IOU_THRESHOLD = 0.35
PROGRESS_THRESHOLD = 0.55

ENTER_MIN_FRAMES = 3     # debounce for hovering
INSERT_MIN_FRAMES = int(T_INSERT * FPS)  # must stay long enough
EXIT_MAX_MISSES = 10 # 2      # tolerate 'brief' occlusion

# =========================
# HELPERS
# =========================

def log(frame_idx, msg):
    print(f"[Frame {frame_idx:04d}] {msg}")

def case_bbox(case_poly):
    x = case_poly[:, 0]
    y = case_poly[:, 1]
    return min(x), min(y), max(x), max(y)

def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0

def insertion_progress(centroid, case_poly):
    x1, y1, x2, y2 = case_bbox(case_poly)
    depth = (centroid[1] - y1) / (y2 - y1)
    return np.clip(depth, 0.0, 1.0)

def poly_to_mask(poly, frame_size):
    mask = np.zeros(frame_size, dtype=np.uint8)
    pts = poly.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0
    return inter / union

def split_case_into_slots(case_poly, frame_size):
    x1, y1 = case_poly.min(axis=0)
    x2, y2 = case_poly.max(axis=0)

    slots = {}
    cols = np.linspace(x1, x2, 4)  # 4 points = 3 cols
    rows = np.linspace(y1, y2, 3)  # 3 points = 2 rows

    # Physical layout :
    # 2 4 6  (top row, rows[0] a rows[1])
    # 1 3 5  (bottom row, rows[1] a rows[2])
    
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

# =========================
# PHASE 1 - YOLO CACHE READ
# =========================

data = pickle.load(open(YOLO_CACHE, "rb"))
frame_size = (720, 1280)

phase1_data = []

for frame in data:
    case = frame.get("case", None)
    if case is None:
        phase1_data.append(None)
        continue

    case_poly = case["polygon"]
    batteries = []

    for b in frame.get("batteries", []):
        centroid = b["centroid"]
        poly = b["polygon"]
        inside = point_inside_polygon(centroid, case_poly)
        progress = insertion_progress(centroid, case_poly)

        batteries.append({
            "centroid": centroid,
            "poly": poly,
            "inside": inside,
            "progress": progress
        })

    phase1_data.append({
        "case_poly": case_poly,
        "batteries": batteries
    })

# =========================
# PHASE 2 - SLOT ASSIGNMENT
# =========================

phase2_data = []

for frame in phase1_data:
    if frame is None:
        phase2_data.append(None)
        continue

    case_poly = frame["case_poly"]
    batteries = frame["batteries"]
    slots, slot_masks = split_case_into_slots(case_poly, frame_size)

    frame_slots = {}

    for b in batteries:
        if not b["inside"]:
            continue

        battery_mask = poly_to_mask(b["poly"], frame_size)
        best_sid = None
        best_iou = 0.0

        for sid, smask in slot_masks.items():
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

    phase2_data.append(frame_slots)

# =========================
# PHASE 3 - FSM WITH IMMUTABLE COMMITS
# =========================

class SlotFSM:
    def __init__(self, slot_id):
        self.sid = slot_id
        self.state = "EMPTY"
        self.counter = 0
        self.miss = 0
        self.committed = False  # Once committed, immutable

    def update(self, present, progress, frame_idx):
        """Returns event: 'INSERT', 'REMOVE', or None"""
        
        # IMMUTABLE: Once committed, ignore all activity!!!
        if self.committed:
            return None
        
        event = None

        if self.state == "EMPTY":
            # IoU threshold in Phase 2 is enough for ENTERING detection
            if present:
                self.state = "ENTERING"
                self.counter = 1
                log(frame_idx, f"Slot {self.sid}: ENTERING")

        elif self.state == "ENTERING":
            if present:
                self.counter += 1
                if self.counter >= INSERT_MIN_FRAMES:
                    self.state = "INSERTED"
                    event = "INSERT"
                    log(frame_idx, f"Slot {self.sid}: INSERTED ✔")
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
                    log(frame_idx, f"↩️  Slot {self.sid}: REMOVED")

        return event

    def mark_committed(self):
        """Lock this slot - no more state changes"""
        self.committed = True


class OrderFSM:
    def __init__(self):
        self.expected_idx = 0
        self.sequence = []
        self.wrong_inserts = {}  # track wrong insertions

    def on_insert(self, slot_id, frame_idx):
        """Handle insertion event"""
        if self.expected_idx >= len(EXPECTED_ORDER):
            return None  # Sequence complete
        
        expected = EXPECTED_ORDER[self.expected_idx]

        if slot_id == expected:
            self.sequence.append(slot_id)
            self.expected_idx += 1
            log(frame_idx, f"✅ COMMITTED: Slot {slot_id} | Order={self.sequence}")
            return "COMMIT"
        else:
            # Wrong slot
            if slot_id not in self.wrong_inserts:
                log(frame_idx, f"⚠️  WRONG SLOT: {slot_id} inserted (expected {expected})")
                self.wrong_inserts[slot_id] = frame_idx
            return "WRONG"

    def on_remove(self, slot_id, frame_idx):
        """Handle removal event - check for backtracking"""
        # Only care if it was the last committed slot
        if self.sequence and self.sequence[-1] == slot_id:
            self.sequence.pop()
            self.expected_idx -= 1
            log(frame_idx, f"↩️ BACKTRACK: Slot {slot_id} removed | Order={self.sequence}")
            return "BACKTRACK"
        
        # Clear wrong insertion tracking
        if slot_id in self.wrong_inserts:
            log(frame_idx, f"✅ Correction: Slot {slot_id} cleared")
            self.wrong_inserts.pop(slot_id)
        
        return None


# Initialize FSMs
slot_fsms = {sid: SlotFSM(sid) for sid in range(1, 7)}
order_fsm = OrderFSM()

# Process frames
for frame_idx, frame_slots in enumerate(phase2_data):
    if frame_slots is None:
        continue

    # Collect all events for this frame
    frame_events = []  # [(slot_id, event_type)]

    for sid, fsm in slot_fsms.items():
        present = sid in frame_slots
        progress = frame_slots[sid]["progress"] if present else 0.0
        
        event = fsm.update(present, progress, frame_idx)
        
        if event:
            frame_events.append((sid, event))

    # PROCESS INSERTS: Handle correct one first, suppress already-committed
    insert_events = [(sid, ev) for sid, ev in frame_events if ev == "INSERT"]
    
    if insert_events:
        # Check if any is the expected slot
        expected = EXPECTED_ORDER[order_fsm.expected_idx] if order_fsm.expected_idx < len(EXPECTED_ORDER) else None
        
        correct_insert = None
        wrong_inserts = []
        
        for sid, ev in insert_events:
            if sid == expected:
                correct_insert = sid
            else:
                wrong_inserts.append(sid)
        
        # Process correct insert first
        if correct_insert:
            result = order_fsm.on_insert(correct_insert, frame_idx)
            if result == "COMMIT":
                slot_fsms[correct_insert].mark_committed()  # Lock it
        
        # Process wrong inserts (only if not already in sequence)
        for sid in wrong_inserts:
            if sid not in order_fsm.sequence:  # Don't warn about already-committed slots...
                order_fsm.on_insert(sid, frame_idx)

    # Process removals
    remove_events = [(sid, ev) for sid, ev in frame_events if ev == "REMOVE"]
    for sid, ev in remove_events:
        order_fsm.on_remove(sid, frame_idx)

# =========================
# FINAL REPORT
# =========================

print("\n" + "="*60)
print("FINAL INSERTION ORDER:", order_fsm.sequence)
print("="*60)

if order_fsm.sequence == EXPECTED_ORDER:
    print("✅ SUCCESS! All batteries inserted in correct order!")
else:
    print(f"⚠️  INCOMPLETE or INCORRECT")
    print(f"   Expected: {EXPECTED_ORDER}")
    print(f"   Got:      {order_fsm.sequence}")