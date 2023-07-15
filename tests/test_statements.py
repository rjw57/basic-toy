from rwbasic.interpreter import Interpreter


def test_empty_statement(interpreter: Interpreter):
    interpreter.execute("")
