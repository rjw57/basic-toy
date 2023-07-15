import runpy

from rwbasic.tool import main


def test_run_module(mocker):
    """Running the rwbasic module should call rwbasic.main."""
    main_mock = mocker.patch("rwbasic.tool.main")
    runpy.run_module("rwbasic")
    main_mock.assert_called()


def test_main(mocker):
    repl_mock = mocker.patch("rwbasic.tool.ReplSession")
    mocker.patch("sys.argv", ["rwbasic"])
    main()
    repl_mock.return_value.run.assert_called()
