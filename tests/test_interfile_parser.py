import pytest

from simind_python_connector.utils.interfile_parser import InterfileHeader


@pytest.mark.unit
def test_interfile_header_get_set(tmp_path):
    header_path = tmp_path / "test.hs"
    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "patient name := phantom",
                "!study ID := study123",
                "!END OF INTERFILE :=",
            ]
        )
    )

    header = InterfileHeader.from_file(header_path)
    assert header.get("patient name") == "phantom"

    header.set("patient name", "updated")
    header.set("new field", 42)
    header.write(header_path)

    text = header_path.read_text()
    assert "patient name := updated" in text
    assert "new field := 42" in text


@pytest.mark.unit
def test_interfile_header_insert(tmp_path):
    header_path = tmp_path / "insert.hs"
    header_path.write_text("a := 1\nc := 3\n")
    header = InterfileHeader.from_file(header_path)
    header.insert(1, "b", "{2, 2}")
    header.write(header_path)

    lines = header_path.read_text().splitlines()
    assert lines[1] == "b := {2, 2}"


@pytest.mark.unit
def test_interfile_header_set_changes_the_effective_duplicate_and_stays_before_end(
    tmp_path,
):
    header_path = tmp_path / "dup.hs"
    header_path.write_text(
        "!INTERFILE :=\nRadius := 100\nRadius := 200\n!END OF INTERFILE :="
    )
    header = InterfileHeader.from_file(header_path)

    header.set("Radius", 300)
    header.set("new field", 42)
    header.write(header_path)

    assert header.get("Radius") == "300"
    lines = header_path.read_text().splitlines()
    assert lines[-2:] == ["new field := 42", "!END OF INTERFILE :="]
