"""Profile completion and process liveness are evidence, not diagnosis rules."""

import os


def process_state(pid):
    try:
        with open(f"/proc/{pid}/stat") as stream:
            text = stream.read()
        state = text[text.rfind(")") + 2 :].split()[0]
        return dict(
            process_exists=True,
            process_state=state,
            process_alive=state not in ["Z", "X"],
        )
    except FileNotFoundError:
        return dict(process_exists=False, process_state=None, process_alive=False)
    except (PermissionError, OSError):
        return dict(process_exists=None, process_state=None, process_alive=None)


def complete_nodes(blobs, epoch, nodes=("0", "1")):
    valid = [
        b
        for n, b in zip(nodes, blobs)
        if b
        and b.get("node") == n
        and b.get("epoch") == epoch
        and b.get("complete") is True
    ]
    return valid, len(valid) == len(nodes)
