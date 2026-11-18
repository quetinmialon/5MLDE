import hashlib
import json

import pytest

from bank_marketing.validation import DataValidationError, validate_csv, validate_dataset


def test_valid_data_preserves_negative_balance_and_sentinel(make_csv):
    report = validate_csv(make_csv())
    assert report["valid"]
    assert report["row_count"] == 2
    assert report["class_counts"] == {"no": 1, "yes": 1}
    assert report["unknown_counts"] == {"poutcome": 2, "job": 1}
    assert report["error_count"] == 0


@pytest.mark.parametrize("field,value", [
    ("age", "0"), ("age", "-1"), ("age", "35.5"), ("age", "nan"),
    ("age", "inf"), ("age", "True"), ("age", "1_000"),
    ("balance", "NaN"), ("balance", "-inf"), ("balance", " "),
    ("balance", "9" * 400), ("balance", "9" * 5000),
    ("day", "0"), ("day", "32"), ("duration", "-1"),
    ("campaign", "0"), ("pdays", "-2"), ("previous", "-1"),
    ("job", "astronaut"), ("education", ""), ("loan", "unknown"),
    ("y", "maybe"), ("y", ""),
])
def test_invalid_values_block_processing(field, value, valid_rows, make_csv):
    rows = [{**valid_rows[0], field: value}]
    with pytest.raises(DataValidationError) as error:
        validate_csv(make_csv(rows))
    assert not error.value.report["valid"]
    assert any(item["field"] == field for item in error.value.report["errors"])


def test_documented_unknown_categories_are_valid(valid_rows, make_csv):
    row = {**valid_rows[0], "job": "unknown", "education": "unknown", "contact": "unknown"}
    report = validate_csv(make_csv([row]))
    assert report["valid"]
    assert report["unknown_counts"] == {"job": 1, "education": 1, "contact": 1, "poutcome": 1}


def test_observed_maxima_are_not_used_as_limits(valid_rows, make_csv):
    row = {**valid_rows[0], "age": "100", "balance": "200000", "pdays": "1000"}
    assert validate_csv(make_csv([row]))["valid"]


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_wrong_schema_is_rejected(change, valid_rows, make_csv):
    fields = list(valid_rows[0])
    if change == "missing":
        fields.remove("age")
    elif change == "extra":
        fields.append("unexpected")
    else:
        fields[0] = "job"
    with pytest.raises(DataValidationError):
        validate_csv(make_csv(fields=fields))


def test_column_order_is_not_required(valid_rows, make_csv):
    assert validate_csv(make_csv(fields=list(reversed(valid_rows[0]))))["valid"]


@pytest.mark.parametrize("kind", ["empty", "header_only", "comma_separator", "bad_encoding", "bad_quote"])
def test_unreadable_or_empty_dataset_is_rejected(kind, make_csv):
    content = make_csv()
    if kind == "empty":
        content = b""
    elif kind == "header_only":
        content = make_csv([])
    elif kind == "comma_separator":
        content = content.replace(b";", b",")
    elif kind == "bad_encoding":
        content += b"\xff"
    else:
        content += b'"unterminated'
    with pytest.raises(DataValidationError):
        validate_csv(content)


@pytest.mark.parametrize("extra_line", [b"\r\n", b"1;2\r\n", b";".join([b"1"] * 18) + b"\r\n"])
def test_wrong_width_or_blank_record_is_rejected(extra_line, make_csv):
    with pytest.raises(DataValidationError):
        validate_csv(make_csv() + extra_line)


def test_utf8_bom_is_supported(make_csv):
    assert validate_csv(b"\xef\xbb\xbf" + make_csv())["valid"]


def test_single_class_batch_is_allowed(valid_rows, make_csv):
    report = validate_csv(make_csv([valid_rows[0]]))
    assert report["class_counts"] == {"no": 1, "yes": 0}


def test_minimum_volume_is_blocking(make_csv):
    with pytest.raises(DataValidationError) as error:
        validate_csv(make_csv(), min_rows=3)
    assert error.value.report["row_count"] == 2


def test_duplicates_are_reported_without_removal(valid_rows, make_csv):
    report = validate_csv(make_csv([valid_rows[0], valid_rows[0]]))
    assert report["valid"]
    assert report["row_count"] == 2
    assert report["duplicate_count"] == 1
    assert report["warnings"]


def test_report_is_saved_when_validation_fails(tmp_path, valid_rows, make_csv):
    report_path = tmp_path / "reports" / "invalid.json"
    rows = [{**valid_rows[0], "y": "invalid"}] * 30
    with pytest.raises(DataValidationError) as error:
        validate_csv(make_csv(rows), report_path=report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == error.value.report
    assert saved["error_count"] == 30
    assert len(saved["errors"]) == 20


def test_report_identifies_exact_input_without_modifying_it(tmp_path, make_csv):
    content = make_csv()
    csv_path = tmp_path / "batch.csv"
    csv_path.write_bytes(content)
    report_path = tmp_path / "report.json"
    report = validate_dataset(csv_path, report_path=report_path)
    assert report["sha256"] == hashlib.sha256(content).hexdigest()
    assert report["source"] == str(csv_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"]
    assert csv_path.read_bytes() == content


def test_report_cannot_overwrite_source(tmp_path, make_csv):
    csv_path = tmp_path / "batch.csv"
    content = make_csv()
    csv_path.write_bytes(content)
    with pytest.raises(ValueError, match="rapport"):
        validate_dataset(csv_path, report_path=csv_path)
    assert csv_path.read_bytes() == content


def test_missing_file_blocks_processing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path / "missing.csv")
