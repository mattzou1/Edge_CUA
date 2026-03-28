import os
import json
from difflib import SequenceMatcher

os.environ["VLLM_PLUGINS"] = "none"
from vllm import LLM, SamplingParams


AVAILABLE_ACTIONS = """
    Action Type | Parameters | Description
    MOVE_TO | x, y | Move the cursor to the specified position
    CLICK | button, x, y, num_clicks | Click the left button if not specified, otherwise click the specified button; click at current position if x,y not specified, otherwise click at specified position
    MOUSE_DOWN | button | Press the left button if not specified, otherwise press the specified button
    MOUSE_UP | button | Release the left button if not specified, otherwise release the specified button
    RIGHT_CLICK | x, y | Right click at current position if x,y not specified, otherwise right click at specified position
    DOUBLE_CLICK | x, y | Double click at current position if x,y not specified, otherwise double click at specified position
    DRAG_TO | x, y | Drag the cursor to the specified position with the left button pressed
    SCROLL | dx, dy | Scroll the mouse wheel up or down
    TYPING | text | Type the specified text
    PRESS | key | Press the specified key and release it
    KEY_DOWN | key | Press the specified key
    KEY_UP | key | Release the specified key
    HOTKEY | keys | Press the specified key combination
    WAIT | - | Wait until the next action
    FAIL | - | Decide the task cannot be performed
    DONE | - | Decide the task is done
"""


# ---------------------------------------------------------------------------
# LLM singleton -- loaded once at server startup (main thread), passed to
# generate_step() so we never reload a 1B model mid-demo.
# ---------------------------------------------------------------------------

_llm_instance = None
_llm_model_path = None


def get_llm(model_path: str) -> LLM:
    global _llm_instance, _llm_model_path
    if _llm_instance is None or _llm_model_path != model_path:
        print(f"[STARTUP] Loading LLM from {model_path}... this takes ~60s")
        _llm_instance = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.75,
            max_model_len=4096,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            enforce_eager=True,
        )
        _llm_model_path = model_path
        print("[STARTUP] LLM ready.")
    return _llm_instance


# ---------------------------------------------------------------------------
# Element utilities
# ---------------------------------------------------------------------------

def is_contained(bbox_a, bbox_b):
    return (bbox_a[0] >= bbox_b[0] and
            bbox_a[1] >= bbox_b[1] and
            bbox_a[2] <= bbox_b[2] and
            bbox_a[3] <= bbox_b[3])


def build_spatial_tree(som_elements):
    hierarchy = {}

    for elem_id, elem_data in som_elements.items():
        hierarchy[elem_id] = {'parent': None, 'children': [], 'depth': 0}

    for child_id, child_data in som_elements.items():
        potential_parents = []

        for parent_id, parent_data in som_elements.items():
            if child_id == parent_id:
                continue
            if is_contained(child_data['bbox'], parent_data['bbox']):
                parent_area = ((parent_data['bbox'][2] - parent_data['bbox'][0]) *
                               (parent_data['bbox'][3] - parent_data['bbox'][1]))
                potential_parents.append((parent_id, parent_area))

        if potential_parents:
            potential_parents.sort(key=lambda x: x[1])
            direct_parent = potential_parents[0][0]
            hierarchy[child_id]['parent'] = direct_parent
            hierarchy[direct_parent]['children'].append(child_id)

    def calculate_depth(elem_id, visited=None):
        if visited is None:
            visited = set()
        if elem_id in visited:
            return 0
        visited.add(elem_id)
        parent_id = hierarchy[elem_id]['parent']
        if parent_id is None:
            hierarchy[elem_id]['depth'] = 0
            return 0
        depth = 1 + calculate_depth(parent_id, visited)
        hierarchy[elem_id]['depth'] = depth
        return depth

    for elem_id in hierarchy:
        calculate_depth(elem_id)

    return hierarchy


def create_som_elements(vision_data):
    som_elements = {}
    elements = vision_data.get("elements", [])
    window_dims = vision_data.get("window dims")
    screen_center_x = window_dims[0] / 2
    screen_center_y = window_dims[1] / 2

    element_id = 1
    for el in elements:
        text = el.get("text", "").strip()
        bbox = el.get("bbox", [0, 0, 0, 0])
        el_type = el.get("type", "unknown")

        if text and len(text) > 1:
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            distance_from_center = ((center_x - screen_center_x)**2 +
                                    (center_y - screen_center_y)**2)**0.5
            max_distance = ((screen_center_x)**2 + (screen_center_y)**2)**0.5
            centrality = 1.0 - (distance_from_center / max_distance)

            confidence = el.get("confidence", el.get("det_confidence", 0.5))

            som_elements[element_id] = {
                "id": element_id,
                "text": text,
                "type": el_type,
                "bbox": bbox,
                "center": (center_x, center_y),
                "confidence": confidence,
                "centrality": centrality,
                "ui_class": el.get("ui_class", None),
            }
            element_id += 1

    return som_elements


def robust_find_element(user_query, som_elements):
    best_id = None
    highest_score = 0.0
    match_details = {}

    for eid, data in som_elements.items():
        text_score = SequenceMatcher(None, user_query.lower(), data['text'].lower()).ratio()
        type_boost = 1.2 if data['type'] == 'ui' else 1.0
        ui_boost = 1.15 if data.get('ui_class') == 'icon' else 1.0

        final_score = (
            text_score * 0.6 +
            data['confidence'] * 0.2 +
            data['centrality'] * 0.2
        ) * type_boost * ui_boost

        if final_score > highest_score and text_score > 0.4:
            highest_score = final_score
            best_id = eid
            match_details = {
                'text': data['text'],
                'text_score': text_score,
                'confidence': data['confidence'],
                'centrality': data['centrality'],
                'final_score': final_score,
                'type': data['type'],
            }

    if best_id is not None:
        print(f"[Fuzzy Match] Query='{user_query}' -> Element {best_id}: '{match_details['text']}' "
              f"(score={match_details['final_score']:.3f})")

    return best_id, highest_score, match_details


def is_valid_action(line):
    line = line.strip()
    if not line:
        return False
    valid_actions = [
        "MOVE_TO", "CLICK", "MOUSE_DOWN", "MOUSE_UP",
        "RIGHT_CLICK", "DOUBLE_CLICK", "DRAG_TO",
        "SCROLL", "TYPING", "PRESS", "KEY_DOWN",
        "KEY_UP", "HOTKEY", "WAIT", "FAIL", "DONE",
        "CLICK_ELEMENT",
    ]
    first_word = line.split()[0] if line.split() else ""
    return first_word in valid_actions


def translate_som_to_coordinates(actions, som_elements):
    translated_actions = []

    for action in actions:
        parts = action.strip().split(maxsplit=1)
        if not parts:
            continue

        action_type = parts[0]

        if action_type == "CLICK_ELEMENT":
            if len(parts) > 1:
                try:
                    elem_id = int(parts[1])
                    if elem_id in som_elements:
                        elem = som_elements[elem_id]
                        center_x, center_y = elem["center"]
                        translated_actions.append(f"CLICK {center_x} {center_y}")
                    else:
                        print(f"Warning: Element ID {elem_id} not found")
                except ValueError:
                    query_text = parts[1].strip()
                    best_id, score, details = robust_find_element(query_text, som_elements)
                    if best_id is not None:
                        elem = som_elements[best_id]
                        center_x, center_y = elem["center"]
                        translated_actions.append(f"CLICK {center_x} {center_y}")
                    else:
                        print(f"Warning: No fuzzy match found for '{query_text}'")
            else:
                print(f"Warning: CLICK_ELEMENT missing ID: {action}")
        else:
            translated_actions.append(action)

    return translated_actions


def build_step_prompt(task, som_elements, previous_actions=None):
    ui_elements = []
    for elem_id, elem_data in sorted(som_elements.items()):
        text = elem_data["text"]
        elem_type = elem_data["type"]
        ui_class = elem_data.get("ui_class", "")
        ui_class_str = f" ({ui_class})" if ui_class else ""
        ui_elements.append(f"  [{elem_id}] {text} - {elem_type}{ui_class_str}")

    ui_context = "\n".join(ui_elements)

    history_context = ""
    if previous_actions and len(previous_actions) > 0:
        history_context = "\nPrevious actions:\n" + "\n".join(
            [f"  {i+1}. {action}" for i, action in enumerate(previous_actions)]
        )
    else:
        history_context = "\nThis is the first action."

    prompt = f"""You are controlling a computer to complete a task. You must think step-by-step and perform ONE action at a time.

TASK: {task}
{history_context}

SCREEN ELEMENTS AVAILABLE:
{ui_context}

AVAILABLE ACTIONS:
- CLICK_ELEMENT <id>  : Click on a UI element by its ID number
- TYPING "text"       : Type text (use after clicking input field)
- HOTKEY <keys>       : Press keyboard shortcut (e.g., ctrl+c, ctrl+v)
- DONE                : Task is complete
- FAIL                : Task cannot be completed

INSTRUCTIONS:
1. Read the TASK carefully
2. Look at what actions have been completed (if any)
3. Determine what the NEXT STEP should be to make progress on the task
4. Find the appropriate UI element from the list above (if needed)
5. Output EXACTLY ONE action in the correct format

EXAMPLES:

Example 1:
TASK: Click on Gmail
Elements: [1] Google - ui, [2] Gmail - ui (icon), [3] Search - ui
Previous actions: (none)
Action: CLICK_ELEMENT 2

Example 2:
TASK: Search for python tutorial
Elements: [1] Google - ui, [2] Search - ui, [3] Settings - ui
Previous actions: 1. CLICK_ELEMENT 2
Action: TYPING "python tutorial"

Example 3:
TASK: Open terminal
Elements: [1] File Manager - ui (icon), [2] Terminal - ui (icon)
Previous actions: (none)
Action: CLICK_ELEMENT 2

Now complete YOUR task:

TASK: {task}
{history_context}

Output EXACTLY ONE action:
Action:"""

    return prompt


# ---------------------------------------------------------------------------
# Main entry point: generate one action step
# ---------------------------------------------------------------------------

def generate_step(
    llm,
    task: str,
    vision_data: dict,
    previous_actions: list = None,
    temperature: float = 0.3,
    max_tokens: int = 128,
) -> tuple:
    """
    Generate a single action step using the pre-loaded LLM.

    Args:
        llm: Pre-loaded vllm.LLM instance (from get_llm())
        task: Task description in plain English
        vision_data: Vision output dict from vision_CPU.py
        previous_actions: List of raw action strings already executed
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        tuple: (raw_action: str, coordinate_action: str | None)
            raw_action -- semantic LLM output, e.g. "CLICK_ELEMENT 5"
            coordinate_action -- pixel-resolved, e.g. "CLICK 640 400" (None for DONE/FAIL)
    """
    som_elements = create_som_elements(vision_data)
    prompt = build_step_prompt(task, som_elements, previous_actions)

    print(f"[PLANNER] Generating action for: {task!r}")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n", "Action:", "TASK:"],
    )
    outputs = llm.generate([prompt], sampling_params)

    raw_action = outputs[0].outputs[0].text.strip().strip('"\'').strip()

    print(f"[PLANNER] raw_action={raw_action!r}")

    if not is_valid_action(raw_action):
        print(f"[PLANNER] Invalid action format: {raw_action!r} -- scanning for DONE/FAIL")
        for word in raw_action.split():
            if word.upper() in ("DONE", "FAIL"):
                raw_action = word.upper()
                break
        else:
            print(f"[PLANNER] No recognizable action found, returning FAIL")
            return "FAIL", None

    is_terminal = raw_action.strip().upper() in ("DONE", "FAIL")

    coordinate_action = None
    if is_valid_action(raw_action) and not is_terminal:
        coord_list = translate_som_to_coordinates([raw_action], som_elements)
        coordinate_action = coord_list[0] if coord_list else None

    print(f"[PLANNER] coordinate_action={coordinate_action!r}")
    return raw_action, coordinate_action
