from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HealthCheckRequest(_message.Message):
    __slots__ = ("args",)
    ARGS_FIELD_NUMBER: _ClassVar[int]
    args: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, args: _Optional[_Iterable[str]] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("success", "exit_code", "output", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    exit_code: int
    output: str
    error: str
    def __init__(self, success: bool = ..., exit_code: _Optional[int] = ..., output: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
