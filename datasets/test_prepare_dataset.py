"""Minimal regression tests for the file conversion; uses temp files and does not touch the real dataset."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from prepare_dataset import (
    apply_changes, apply_preparation, convert_csv, plan_changes, verify_source_paths,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.mapping = []
        for index in range(2):
            data = f"image contents {index}".encode()
            old = f"train/class/{index}wet-class.jpg"
            new = f"train/class/{index}-class.jpg"
            path = self.root / old
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.mapping.append({"old_path": old, "new_path": new,
                                 "sha256": hashlib.sha256(data).hexdigest()})

    def test_conversion_preserves_bytes_and_rerun_is_noop(self):
        changes = plan_changes(self.root, self.mapping)
        self.assertEqual(len(changes), 2)
        apply_changes(changes)
        for row in self.mapping:
            self.assertFalse((self.root / row["old_path"]).exists())
            data = (self.root / row["new_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
        self.assertEqual(plan_changes(self.root, self.mapping), [])

    def test_preflight_does_not_rename(self):
        plan_changes(self.root, self.mapping)
        for row in self.mapping:
            self.assertTrue((self.root / row["old_path"]).is_file())
            self.assertFalse((self.root / row["new_path"]).exists())

    def test_partially_converted_dataset(self):
        first = self.mapping[0]
        (self.root / first["old_path"]).rename(self.root / first["new_path"])
        changes = plan_changes(self.root, self.mapping)
        self.assertEqual(len(changes), 1)
        apply_changes(changes)
        self.assertEqual(plan_changes(self.root, self.mapping), [])

    def test_conflict_never_overwrites(self):
        target = self.root / self.mapping[-1]["new_path"]
        target.write_bytes(b"keep existing file")
        with self.assertRaisesRegex(ValueError, "Both source and destination exist"):
            plan_changes(self.root, self.mapping)
        self.assertEqual(target.read_bytes(), b"keep existing file")
        self.assertTrue((self.root / self.mapping[0]["old_path"]).exists())

    def test_missing_file_stops_before_any_rename(self):
        (self.root / self.mapping[-1]["old_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Neither source nor destination exists"):
            plan_changes(self.root, self.mapping)
        self.assertTrue((self.root / self.mapping[0]["old_path"]).exists())

    def test_changed_image_stops_before_any_rename(self):
        (self.root / self.mapping[-1]["old_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            plan_changes(self.root, self.mapping)
        self.assertTrue((self.root / self.mapping[0]["old_path"]).exists())

    def test_target_created_after_preflight_is_not_overwritten(self):
        changes = plan_changes(self.root, self.mapping)
        target = changes[0][1]
        target.write_bytes(b"created after preflight")
        with self.assertRaises(FileExistsError):
            apply_changes(changes)
        self.assertEqual(target.read_bytes(), b"created after preflight")
        self.assertTrue(changes[0][0].exists())

    def csv_plan(self):
        original = b'image,label,class\r\ntrain/class/0wet-class.jpg,7,class\r\n'
        path = self.root / "train.csv"
        path.write_bytes(original)
        data, count = convert_csv(original, self.mapping)
        return {"path": path, "original": original, "data": data, "count": count,
                "original_hash": hashlib.sha256(original).hexdigest(),
                "target_hash": hashlib.sha256(data).hexdigest(), "rows": 1}

    def test_csv_conversion_only_changes_image_field(self):
        original = (b'image,label,class\r\ntrain/class/0wet-class.jpg,7,"class,quoted"\r\n'
                    b'other.jpg,03,another\r\n')
        converted, count = convert_csv(original, self.mapping)
        self.assertEqual(count, 1)
        self.assertEqual(converted, original.replace(b'0wet-class.jpg', b'0-class.jpg'))

    def test_csv_backup_and_image_conversion(self):
        plan = self.csv_plan()
        apply_preparation(plan_changes(self.root, self.mapping), [plan])
        self.assertEqual(plan["path"].read_bytes(), plan["data"])
        self.assertEqual((self.root / "train.csv.bak").read_bytes(), plan["original"])
        self.assertEqual(plan_changes(self.root, self.mapping), [])
        self.assertEqual(list(self.root.glob(".train.csv.*")), [])

    def test_conflicting_backup_stops_before_image_conversion(self):
        plan = self.csv_plan()
        backup = self.root / "train.csv.bak"
        backup.write_bytes(b"existing backup")
        with self.assertRaisesRegex(ValueError, "backup conflict"):
            apply_preparation(plan_changes(self.root, self.mapping), [plan])
        self.assertEqual(backup.read_bytes(), b"existing backup")
        self.assertEqual(plan["path"].read_bytes(), plan["original"])
        self.assertEqual(len(plan_changes(self.root, self.mapping)), 2)
        self.assertEqual(list(self.root.glob(".train.csv.*")), [])

    def test_matching_backup_allows_resume(self):
        plan = self.csv_plan()
        backup = self.root / "train.csv.bak"
        backup.write_bytes(plan["original"])
        apply_preparation([], [plan])
        self.assertEqual(backup.read_bytes(), plan["original"])
        self.assertEqual(plan["path"].read_bytes(), plan["data"])

    def test_changed_csv_stops_before_image_conversion(self):
        plan = self.csv_plan()
        plan["path"].write_bytes(b"edited after preflight")
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            apply_preparation(plan_changes(self.root, self.mapping), [plan])
        self.assertEqual(plan["path"].read_bytes(), b"edited after preflight")
        self.assertEqual(len(plan_changes(self.root, self.mapping)), 2)
        self.assertFalse((self.root / "train.csv.bak").exists())

    def test_source_validation_without_subsets_before_and_after_conversion(self):
        plan = self.csv_plan()
        changes = plan_changes(self.root, self.mapping)
        self.assertFalse((self.root / "subsets").exists())
        self.assertEqual(verify_source_paths(self.root, changes, [plan]), 1)
        apply_preparation(changes, [plan])
        self.assertEqual(verify_source_paths(self.root, [], [plan]), 1)
        self.assertFalse((self.root / "subsets").exists())

    def test_source_validation_rejects_missing_image(self):
        plan = self.csv_plan()
        changes = plan_changes(self.root, self.mapping)
        changes[0][0].unlink()
        with self.assertRaisesRegex(ValueError, "Missing images"):
            verify_source_paths(self.root, changes, [plan])

    def test_source_validation_rejects_wrong_row_count(self):
        plan = self.csv_plan()
        plan["rows"] = 2
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            verify_source_paths(self.root, plan_changes(self.root, self.mapping), [plan])


if __name__ == "__main__":
    unittest.main()
