"""Task-scoped SSH; credentials are read from the user's local server file."""

import argparse, concurrent.futures, pathlib, re, sys, time
import paramiko

ROOT = pathlib.Path(__file__).resolve().parents[1]


def connect(host):
    secret = re.search(r"密码\s+(\S+)", (ROOT / "服务器.md").read_text()).group(1)
    for attempt in range(3):
        c = paramiko.SSHClient()
        c.load_system_host_keys()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(
                host,
                username="test",
                password=secret,
                timeout=12,
                auth_timeout=15,
                banner_timeout=15,
            )
            return c
        except (paramiko.SSHException, OSError):
            c.close()
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


def run(host, command, timeout=120):
    c = connect(host)
    try:
        _, out, err = c.exec_command(command, timeout=timeout)
        s = out.read().decode(errors="replace") + err.read().decode(errors="replace")
        return host, out.channel.recv_exit_status(), s
    finally:
        c.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--host", default="both")
    p.add_argument("--timeout", type=int, default=120)
    a = p.parse_args()
    hosts = ["10.10.4.1", "10.10.4.2"] if a.host == "both" else [a.host]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for h, rc, s in pool.map(lambda h: run(h, a.command, a.timeout), hosts):
            print(f"HOST {h} EXIT {rc}\n{s}")


def sudo_run(host, command, timeout=120):
    c = connect(host)
    try:
        import shlex

        sin, out, err = c.exec_command(
            'sudo -k -S -p "" bash -lc ' + shlex.quote(command), timeout=timeout
        )
        try:
            sin.write(
                re.search(r"密码\s+(\S+)", (ROOT / "服务器.md").read_text()).group(1)
                + "\n"
            )
            sin.flush()
            sin.channel.shutdown_write()
        except OSError:
            if not out.channel.exit_status_ready():
                raise
        result = out.read().decode(errors="replace") + err.read().decode(
            errors="replace"
        )
        return host, out.channel.recv_exit_status(), result
    finally:
        c.close()
