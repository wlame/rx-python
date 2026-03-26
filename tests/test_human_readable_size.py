"""Tests for human_readable_size utility function."""

from rx.models import human_readable_size


class TestHumanReadableSize:
    def test_zero_bytes(self):
        assert human_readable_size(0) == '0.00 B'

    def test_bytes(self):
        assert human_readable_size(500) == '500.00 B'

    def test_one_byte(self):
        assert human_readable_size(1) == '1.00 B'

    def test_just_under_kb(self):
        assert human_readable_size(1023) == '1023.00 B'

    def test_exactly_one_kb(self):
        assert human_readable_size(1024) == '1.00 KB'

    def test_kilobytes(self):
        result = human_readable_size(1536)
        assert result == '1.50 KB'

    def test_megabytes(self):
        result = human_readable_size(5 * 1024 * 1024)
        assert result == '5.00 MB'

    def test_gigabytes(self):
        result = human_readable_size(2 * 1024 * 1024 * 1024)
        assert result == '2.00 GB'

    def test_terabytes(self):
        result = human_readable_size(3 * 1024 ** 4)
        assert result == '3.00 TB'

    def test_petabytes(self):
        result = human_readable_size(2 * 1024 ** 5)
        assert result == '2.00 PB'

    def test_large_value(self):
        result = human_readable_size(10 * 1024 ** 5)
        assert result == '10.00 PB'

    def test_format_has_two_decimals(self):
        result = human_readable_size(1500)
        assert '.46 KB' in result

    def test_importable_from_models(self):
        """Verify backward-compatible import from models.py still works."""
        from rx.models import human_readable_size as hrs
        assert hrs(1024) == '1.00 KB'

    def test_importable_from_web(self):
        """Verify backward-compatible import from web.py still works."""
        from rx.web import human_readable_size as hrs
        assert hrs(1024) == '1.00 KB'
