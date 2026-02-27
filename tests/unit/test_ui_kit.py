"""Unit tests for UI-Kit components (Step 28)."""
import os
import pytest

UIKIT = os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ui-kit", "src")


class TestUIKitComponents:
    def test_modal_exists(self):
        assert os.path.isfile(os.path.join(UIKIT, "Modal.tsx"))

    def test_dropdown_exists(self):
        assert os.path.isfile(os.path.join(UIKIT, "Dropdown.tsx"))

    def test_table_exists(self):
        assert os.path.isfile(os.path.join(UIKIT, "Table.tsx"))

    def test_toast_exists(self):
        assert os.path.isfile(os.path.join(UIKIT, "Toast.tsx"))

    def test_index_exports(self):
        idx = os.path.join(UIKIT, "index.ts")
        assert os.path.isfile(idx)
        content = open(idx, encoding='utf-8').read()
        for comp in ["Modal", "Dropdown", "Table", "ToastProvider", "useToast"]:
            assert comp in content, f"Missing export: {comp}"

    def test_modal_has_accessibility(self):
        content = open(os.path.join(UIKIT, "Modal.tsx")).read()
        assert "aria-modal" in content
        assert "dialog" in content

    def test_dropdown_has_accessibility(self):
        content = open(os.path.join(UIKIT, "Dropdown.tsx")).read()
        assert "aria-haspopup" in content
        assert "listbox" in content

    def test_table_has_pagination(self):
        content = open(os.path.join(UIKIT, "Table.tsx")).read()
        assert "pageSize" in content
        assert "totalPages" in content

    def test_toast_has_types(self):
        content = open(os.path.join(UIKIT, "Toast.tsx")).read()
        for t in ["success", "error", "warning", "info"]:
            assert t in content
