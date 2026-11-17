import builtins
import pytest

from argus import local_cli


def test_list_location_shows_directory_entries(tmp_path, capsys):
    (tmp_path / "folder").mkdir()
    (tmp_path / "notes.txt").write_text("local", encoding="utf-8")

    local_cli.list_location(str(tmp_path))

    output = capsys.readouterr().out
    assert "[DIR] folder" in output
    assert "notes.txt" in output


def test_delete_file_requires_exact_confirmation(tmp_path, monkeypatch, capsys):
    target = tmp_path / "remove-me.txt"
    target.write_text("data", encoding="utf-8")
    monkeypatch.setattr(builtins, "input", lambda _prompt: "DELETE")

    assert local_cli.delete_file(str(target)) is True
    assert not target.exists()
    assert "Deleted:" in capsys.readouterr().out


def test_delete_file_cancellation_preserves_file(tmp_path, monkeypatch, capsys):
    target = tmp_path / "keep-me.txt"
    target.write_text("data", encoding="utf-8")
    monkeypatch.setattr(builtins, "input", lambda _prompt: "yes")

    assert local_cli.delete_file(str(target)) is False
    assert target.exists()
    assert "Cancelled" in capsys.readouterr().out


def test_delete_file_rejects_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builtins, "input", lambda _prompt: pytest.fail("must not prompt")
    )

    with pytest.raises(ValueError, match="Only individual files"):
        local_cli.delete_file(str(tmp_path))


def test_open_notepad_uses_fixed_executable(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        local_cli.subprocess,
        "Popen",
        lambda command: calls.append(command) or type("Process", (), {"pid": 42})(),
    )

    local_cli.open_target("notepad")

    assert calls == [["notepad.exe"]]
    assert "Opened notepad (PID: 42)" in capsys.readouterr().out


def test_open_folder_path_uses_explorer(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        local_cli.subprocess,
        "Popen",
        lambda command: calls.append(command) or type("Process", (), {"pid": 7})(),
    )

    local_cli.open_target(str(tmp_path))

    assert calls == [["explorer.exe", str(tmp_path.resolve())]]
    assert "Opened folder:" in capsys.readouterr().out


def test_open_document_path_uses_file_association(tmp_path, monkeypatch, capsys):
    target = tmp_path / "notes.txt"
    target.write_text("notes", encoding="utf-8")
    opened = []
    monkeypatch.setattr(local_cli.os, "startfile", opened.append)

    local_cli.open_target(str(target))

    assert opened == [str(target.resolve())]
    assert "Opened file:" in capsys.readouterr().out


def test_open_rejects_arbitrary_command(monkeypatch):
    monkeypatch.setattr(
        local_cli.subprocess,
        "Popen",
        lambda _command: pytest.fail("unsupported target must not launch"),
    )

    with pytest.raises(FileNotFoundError, match="Target not found"):
        local_cli.open_target("powershell -Command Remove-Item")


def test_open_refuses_executable_path(tmp_path, monkeypatch):
    target = tmp_path / "unexpected.exe"
    target.write_bytes(b"not run")
    monkeypatch.setattr(
        local_cli.os,
        "startfile",
        lambda _path: pytest.fail("executable path must not run"),
    )

    with pytest.raises(ValueError, match="Refusing to launch"):
        local_cli.open_target(str(target))


def test_main_reports_missing_location(tmp_path, capsys):
    result = local_cli.main(["list", str(tmp_path / "missing")])

    assert result == 1
    assert "Path not found" in capsys.readouterr().err


def test_mcp_call_requires_confirmation(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(builtins, "input", lambda _prompt: "no")
    monkeypatch.setattr(
        local_cli,
        "run_mcp",
        lambda *args: calls.append(args)
        or pytest.fail("cancelled call must not connect"),
    )

    result = local_cli.main(["mcp", "call", "get_system_info"])

    assert result == 1
    assert calls == []
    assert "Cancelled" in capsys.readouterr().out


def test_mcp_call_cancels_when_terminal_input_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _prompt: (_ for _ in ()).throw(EOFError()),
    )
    monkeypatch.setattr(
        local_cli,
        "run_mcp",
        lambda *_args: pytest.fail("call must not run without confirmation"),
    )

    result = local_cli.main(["mcp", "call", "get_system_info"])

    assert result == 1
    assert "No interactive input" in capsys.readouterr().out


def test_mcp_call_requires_json_object(monkeypatch, capsys):
    monkeypatch.setattr(
        builtins,
        "input",
        lambda _prompt: pytest.fail("invalid arguments must not prompt"),
    )

    result = local_cli.main(["mcp", "call", "get_system_info", "--args", "[]"])

    assert result == 1
    assert "--args must be a JSON object" in capsys.readouterr().err
