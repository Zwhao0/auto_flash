import os, json, time
import redis
from .config import load


class Store:
    def __init__(self, run=None):
        self.config = load()
        self.run = run or os.environ["FLASH_RUN_ID"]
        self.prefix = "flash:" + self.run + ":"
        self.r = redis.Redis(
            host=os.getenv("FLASH_REDIS_HOST", self.config["redis_host"]),
            port=int(os.getenv("FLASH_REDIS_PORT", str(self.config["redis_port"]))),
            decode_responses=True,
            socket_timeout=3,
        )

    def k(self, key):
        return self.prefix + key

    def get(self, key, default=None):
        v = self.r.get(self.k(key))
        return json.loads(v) if v is not None else default

    def set(self, key, value):
        self.r.set(self.k(key), json.dumps(value, allow_nan=False))

    def push(self, key, value, limit=500):
        with self.r.pipeline() as p:
            p.rpush(self.k(key), json.dumps(value, allow_nan=False))
            p.ltrim(self.k(key), -limit, -1)
            p.execute()

    def items(self, key):
        return [json.loads(v) for v in self.r.lrange(self.k(key), 0, -1)]

    def event(self, kind, **kw):
        self.push("events", dict(kw, kind=kind, time=time.time()), 10000)
