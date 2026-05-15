import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_rag.ingest import Chunk, _make_chunk_id, load_directory, load_file, load_pdf, load_text


# ---------------------------------------------------------------------------
# 1. test_chunk_dataclass
# ---------------------------------------------------------------------------

def test_chunk_dataclass():
    chunk = Chunk(text="hello world", source="/some/file.txt", page=1, chunk_id="abc123")
    assert chunk.text == "hello world"
    assert chunk.source == "/some/file.txt"
    assert chunk.page == 1
    assert chunk.chunk_id == "abc123"


# ---------------------------------------------------------------------------
# 2. test_chunk_id_is_md5
# ---------------------------------------------------------------------------

def test_chunk_id_is_md5():
    text = "some content to hash"
    expected = hashlib.md5(text.encode("utf-8")).hexdigest()
    assert _make_chunk_id(text) == expected
    assert len(_make_chunk_id(text)) == 32


# ---------------------------------------------------------------------------
# 3. test_load_text_basic
# ---------------------------------------------------------------------------

def test_load_text_basic(mocker, tmp_path):
    fake_file = tmp_path / "doc.txt"
    sample_text = "A" * 100
    mocker.patch.object(Path, "read_text", return_value=sample_text)
    chunks = load_text(fake_file)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source == str(fake_file)
    assert chunks[0].page == 1


# ---------------------------------------------------------------------------
# 4. test_load_text_chunking_overlap
# ---------------------------------------------------------------------------

def test_load_text_chunking_overlap(mocker, tmp_path):
    import local_rag.ingest as ingest_module

    original_chunk_size = ingest_module.CHUNK_SIZE
    original_overlap = ingest_module.CHUNK_OVERLAP
    ingest_module.CHUNK_SIZE = 10
    ingest_module.CHUNK_OVERLAP = 3

    try:
        fake_file = tmp_path / "doc.txt"
        sample_text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mocker.patch.object(Path, "read_text", return_value=sample_text)
        chunks = load_text(fake_file)
        assert len(chunks) >= 2
        # end of first chunk overlaps with start of second
        end_of_first = chunks[0].text[-3:]
        start_of_second = chunks[1].text[:3]
        assert end_of_first == start_of_second
    finally:
        ingest_module.CHUNK_SIZE = original_chunk_size
        ingest_module.CHUNK_OVERLAP = original_overlap


# ---------------------------------------------------------------------------
# 5. test_load_text_single_short
# ---------------------------------------------------------------------------

def test_load_text_single_short(mocker, tmp_path):
    fake_file = tmp_path / "short.txt"
    short_text = "Short text"
    mocker.patch.object(Path, "read_text", return_value=short_text)
    chunks = load_text(fake_file)
    assert len(chunks) == 1
    assert chunks[0].text == short_text


# ---------------------------------------------------------------------------
# 6. test_load_pdf
# ---------------------------------------------------------------------------

def test_load_pdf(mocker, tmp_path):
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.touch()

    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page content here. " * 10

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.close = MagicMock()

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    with patch.dict("sys.modules", {"fitz": mock_fitz}):
        chunks = load_pdf(fake_pdf)

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    # page numbers should be set (1-indexed)
    page_nums = {c.page for c in chunks}
    assert page_nums.issubset({1, 2})


# ---------------------------------------------------------------------------
# 7. test_load_file_dispatches_pdf
# ---------------------------------------------------------------------------

def test_load_file_dispatches_pdf(mocker, tmp_path):
    fake_pdf = tmp_path / "document.pdf"
    fake_pdf.touch()

    mock_load_pdf = mocker.patch("local_rag.ingest.load_pdf", return_value=[])
    load_file(fake_pdf)
    mock_load_pdf.assert_called_once_with(fake_pdf)


# ---------------------------------------------------------------------------
# 8. test_load_file_dispatches_text
# ---------------------------------------------------------------------------

def test_load_file_dispatches_text(mocker, tmp_path):
    fake_txt = tmp_path / "notes.txt"
    fake_txt.touch()

    mock_load_text = mocker.patch("local_rag.ingest.load_text", return_value=[])
    load_file(fake_txt)
    mock_load_text.assert_called_once_with(fake_txt)


# ---------------------------------------------------------------------------
# 9. test_load_directory
# ---------------------------------------------------------------------------

def test_load_directory(mocker, tmp_path):
    (tmp_path / "a.txt").write_text("text a")
    (tmp_path / "b.md").write_text("text b")
    (tmp_path / "c.pdf").write_text("fake pdf")
    (tmp_path / "d.bin").write_text("ignored")

    dummy_chunk = Chunk(text="x", source="s", page=1, chunk_id="id")
    mock_load_file = mocker.patch("local_rag.ingest.load_file", return_value=[dummy_chunk])

    chunks = load_directory(tmp_path)

    # load_file should be called 3 times (txt, md, pdf) but NOT for .bin
    assert mock_load_file.call_count == 3
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# 10. test_chunk_id_uniqueness
# ---------------------------------------------------------------------------

def test_chunk_id_uniqueness():
    id1 = _make_chunk_id("first text")
    id2 = _make_chunk_id("second text")
    id3 = _make_chunk_id("third text")
    assert id1 != id2
    assert id1 != id3
    assert id2 != id3
