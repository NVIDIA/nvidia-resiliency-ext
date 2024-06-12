# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import inspect
import typing


def check_type(annotation, cls):
    if annotation == cls:
        return True
    if getattr(annotation, '__origin__', None) == typing.Union and cls in annotation.__args__:
        return True
    if getattr(annotation, '__origin__', None) == typing.Optional:
        return check_type(annotation.__args__[0], cls)
    if getattr(cls, '__origin__', None) == typing.Union:
        return any(check_type(annotation, cls_arg) for cls_arg in cls.__args__)
    return issubclass(cls, annotation)


def count_type_in_params(fn, cls):
    signature = inspect.signature(fn)
    res = sum(check_type(param.annotation, cls) for param in signature.parameters.values())
    return res


def substitute_param_value(fn, args, kwargs, subs):
    signature = inspect.signature(fn)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    for name, param in signature.parameters.items():
        for cls, value in subs.items():
            if check_type(param.annotation, cls):
                bound_args.arguments[name] = value

    return bound_args.args, bound_args.kwargs


def enforce_subclass(argument, class_or_tuple):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    caller_locals = caller_frame.f_locals
    value = caller_locals[argument]
    value_type = type(value)
    if not issubclass(value, class_or_tuple):
        msg = (
            f'{argument=} needs to be a subclass of {class_or_tuple}, '
            f'but got {argument}={value} of type={value_type}'
        )
        raise TypeError(msg)


def enforce_type(argument, class_or_tuple):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    caller_locals = caller_frame.f_locals
    value = caller_locals[argument]
    value_type = type(value)
    if not isinstance(value, class_or_tuple):
        msg = (
            f'{argument=} needs to be an instance of {class_or_tuple}, '
            f'but got {argument}={value} of type={value_type}'
        )
        raise TypeError(msg)


def enforce_value(condition):
    if not condition:
        # Access the previous frame in the call stack
        frame = inspect.currentframe().f_back
        # Get the code context of the calling frame
        code_context = inspect.getframeinfo(frame).code_context
        if code_context:
            # Extract the line of code where `enforce_value` was called
            line = code_context[0].strip()
            # Parse the line to an Abstract Syntax Tree (AST)
            tree = ast.parse(line)
            # Navigate the AST to find the condition expression
            if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                call = tree.body[0].value
                co_name = inspect.currentframe().f_code.co_name
                if hasattr(call.func, 'id') and call.func.id == co_name:
                    # Get the condition argument
                    arg = call.args[0]
                    # Reconstruct the source code of the condition
                    condition_str = ast.unparse(arg)

                    # Collect variable names from the condition
                    class NameCollector(ast.NodeVisitor):
                        def __init__(self):
                            self.names = set()

                        def visit_Name(self, node):
                            self.names.add(node.id)

                        def visit_Attribute(self, node):
                            # Handle attributes like obj.attr
                            self.generic_visit(node)

                    collector = NameCollector()
                    collector.visit(arg)
                    variable_names = collector.names
                    # Get variable values from the frame's locals and globals
                    values = {}
                    for name in variable_names:
                        if name in frame.f_locals:
                            values[name] = frame.f_locals[name]
                        elif name in frame.f_globals:
                            values[name] = frame.f_globals[name]
                        else:
                            values[name] = '<undefined>'
                    # Prepare values string
                    values_str = ', '.join(f'{name}={values[name]!r}' for name in sorted(values))
                else:
                    condition_str = '<unknown condition>'
                    values_str = ''
            else:
                condition_str = '<unknown condition>'
                values_str = ''
        else:
            condition_str = '<unknown condition>'
            values_str = ''
        # Raise an exception with the condition and variable values
        if values_str:
            message = f'Condition "{condition_str}" failed. Variables: {values_str}'
        else:
            message = f'Condition "{condition_str}" failed.'
        raise ValueError(message)
