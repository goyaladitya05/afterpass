"""Link compiled nodes to source nodes, grading every link.

explicit: the runtime named the source node exactly.
inferred: matched through a tensor name or a name prefix, or the name is ambiguous.
unknown:  nothing matched; recorded as a link with source None.
"""

RANK = {"explicit": 0, "inferred": 1}


class Linker:
    def __init__(self, source: dict):
        self.by_name: dict[str, list[int]] = {}
        self.producer: dict[str, int] = {}
        for n in source["nodes"]:
            if n["name"]:
                self.by_name.setdefault(n["name"], []).append(n["id"])
            for t in n["outputs"]:
                self.producer[t] = n["id"]

    def _resolve(self, name: str) -> list[tuple[int, str, str]]:
        """Return (source id, grade, via) for one name the runtime claims."""
        ids = self.by_name.get(name)
        if ids:
            grade = "explicit" if len(ids) == 1 else "inferred"
            return [(i, grade, "name") for i in ids]
        if name in self.producer:
            return [(self.producer[name], "inferred", "tensor")]
        prefix = name
        while "/" in prefix.strip("/"):
            prefix = prefix.rstrip("/").rsplit("/", 1)[0]
            if prefix in self.by_name:
                return [(i, "inferred", "prefix") for i in self.by_name[prefix]]
            if prefix in self.producer:
                return [(self.producer[prefix], "inferred", "prefix")]
        return []

    def link(self, compiled_id: int, claimed: list[str], out_tensors: list[str]) -> list[dict]:
        best: dict[int, dict] = {}

        def add(sid, grade, via, claim):
            cur = best.get(sid)
            if cur is None or RANK[grade] < RANK[cur["grade"]]:
                best[sid] = {"compiled": compiled_id, "source": sid, "grade": grade, "via": via, "claim": claim}

        for name in claimed:
            for sid, grade, via in self._resolve(name):
                add(sid, grade, via, name)
        if not best:
            for t in out_tensors:
                if t in self.producer:
                    add(self.producer[t], "inferred", "tensor", t)
        if not best:
            return [{"compiled": compiled_id, "source": None, "grade": "unknown"}]
        return list(best.values())
