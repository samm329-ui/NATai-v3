import threading
from typing import Optional, Tuple

_key_counter = 0
_counter_lock = threading.Lock()


def get_next_key_pair(
    num_keys: int, need_brain: bool = True
) -> Tuple[Optional[int], int]:
    if not need_brain:
        return (None, 0)

    if num_keys <= 1:
        return (0, 0)

    global _key_counter

    with _counter_lock:
        brain_idx = _key_counter % num_keys
        chat_idx = (_key_counter + 1) % num_keys
        _key_counter += 1

    return (brain_idx, chat_idx)
