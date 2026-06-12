# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import pytest

from floudsonnx.utils.path_guard import safe_join, safe_model_folder


class TestSafeModelFolder:
    def test_strips_org_prefix(self):
        assert safe_model_folder("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"

    def test_no_org_prefix(self):
        assert safe_model_folder("t5-small") == "t5-small"

    def test_strips_unsafe_chars(self):
        result = safe_model_folder("org/model<name>")
        assert "<" not in result and ">" not in result

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            safe_model_folder("/")


class TestSafeJoin:
    def test_safe_join_inside_base(self, tmp_path):
        result = safe_join(str(tmp_path), "subdir", "file.onnx")
        assert result.startswith(str(tmp_path))

    def test_path_traversal_raises(self, tmp_path):
        with pytest.raises(ValueError, match="traversal"):
            safe_join(str(tmp_path), "../../etc/passwd")
