import csv
import os

SOURCE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source_data")


class UnitTable:
    def __init__(self, data_dir=SOURCE_DATA_DIR):
        self.root = {}
        self.units = []
        for r in csv.DictReader(open(os.path.join(data_dir, "units.csv"), encoding="utf-8")):
            cids = tuple(map(int, r["char_ids"].split()))
            assert int(r["id"]) == len(self.units)
            self.units.append((r["unit"], cids, int(r["count"])))
            node = self.root
            for cid in cids:
                node = node.setdefault(cid, {})
            node.setdefault(None, []).append((r["unit"], int(r["count"]), int(r["id"])))
        self.n_units = len(self.units)
