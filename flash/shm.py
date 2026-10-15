import mmap, struct, pathlib

HEADER = 128
NCOMM = 512
COMM = struct.Struct("<8Q")
RECORD = struct.Struct("<10Q")
CAP = 65536
MAGIC = 0x464C415348303031


class Reader:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "r+b")
        self.m = mmap.mmap(self.file.fileno(), 0)
        self.cursor = 0
        self.lost = 0
        if self.header()[0] != MAGIC:
            raise ValueError("uninitialized FLASH shm")

    def header(self):
        return struct.unpack_from("<16Q", self.m, 0)

    def mode(self, mode, epoch):
        struct.pack_into("<Q", self.m, 16, epoch)
        struct.pack_into("<Q", self.m, 8, mode)

    def metadata(self):
        n = min(self.header()[11], NCOMM)
        return {
            c[0]: dict(comm=c[0], hash=str(c[1]), rank=c[2], local=c[3], size=c[4])
            for c in (
                COMM.unpack_from(self.m, HEADER + i * COMM.size) for i in range(n)
            )
        }

    def records(self):
        end = self.header()[3]
        if end - self.cursor > CAP:
            self.lost += end - self.cursor - CAP
            self.cursor = end - CAP
        while self.cursor < end:
            expected = self.cursor + 1
            pos = HEADER + NCOMM * COMM.size + (self.cursor % CAP) * RECORD.size
            r = RECORD.unpack_from(self.m, pos)
            if (
                r[0] == expected
                and struct.unpack_from("<Q", self.m, pos)[0] == expected
            ):
                yield r
            else:
                self.lost += 1
            self.cursor += 1

    def close(self):
        self.m.close()
        self.file.close()


def discover(run):
    return pathlib.Path("/dev/shm").glob("flash_" + run + "_*")
