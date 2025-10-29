#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    HTTP server for fault tolerance launcher.

    Provides REST API endpoints to query launcher operations.
    Currently exposes:
    - / - Basic launcher status (launcher_start_time)
    - /cycles - GET/POST endpoint to query and update cycle information
"""

import json
import logging
from datetime import datetime
from socketserver import ThreadingMixIn
import threading
from typing import Optional
from wsgiref.simple_server import WSGIServer, make_server

from werkzeug.exceptions import HTTPException
from werkzeug.routing import Map, Rule
from werkzeug.wrappers import Request, Response

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig


class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    """WSGI server that handles requests in separate threads."""

    daemon_threads = True
    allow_reuse_address = True


class LauncherHttpHandler:
    """
    HTTP handler for launcher REST API endpoints.

    Provides simplified API to query launcher status and update cycle information.

    Usage:
        # CycleManager is a singleton on the agent
        handler = LauncherHttpHandler(agent)
        server_manager = HttpServerManager(handler, host='0.0.0.0', port=8080)
        server_manager.start()
    """

    def __init__(self, agent):
        """
        Initialize HTTP handler with reference to LocalElasticAgent.

        Args:
            agent: LocalElasticAgent instance (must have cycle_manager attribute)
        """
        self.agent = agent
        self._launcher_start_time = datetime.utcnow()
        self._logger = logging.getLogger(LogConfig.name)

        # Define URL routing
        self._url_map = Map(
            [
                Rule('/', endpoint=self.handle_status),
                Rule('/cycles', endpoint=self.handle_cycles, methods=['GET', 'POST']),
            ]
        )

    def __call__(self, environ, start_response):
        """WSGI application entry point."""
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def dispatch_request(self, request):
        """Route the request to the appropriate handler."""
        adapter = self._url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            return endpoint(request, **values)
        except HTTPException as e:
            return e
        except Exception as e:
            self._logger.exception("Error handling request: %s", e)
            return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

    def handle_status(self, request):
        """
        Root endpoint - basic launcher status.

        Returns:
            JSON with launcher_start_time
        """
        status = {
            "launcher_start_time": self._launcher_start_time.isoformat() + "Z",
        }

        return self._json_response(status)

    def handle_cycles(self, request):
        """
        Cycles endpoint for GET (query cycles) or POST (update cycle).

        GET:
            - /cycles - returns all cached cycles
            - /cycles?cycle_number=3 - returns specific cycle (by number)
            - /cycles?cycle_number=-1 - returns last cycle (negative indexing supported)

        POST:
            - Updates cycle with failure_reason and/or metadata
            - Request body (JSON):
                {
                    "cycle_number": <int>,       # required
                    "failure_reason": <str>,     # optional
                    "metadata": <dict>           # optional
                }

        Returns:
            JSON with cycle information including all profiling events
        """
        # Access cycle manager from agent
        cycle_manager = self.agent.cycle_manager

        if request.method == 'GET':
            # Check for cycle_number query parameter
            cycle_number_param = request.args.get('cycle_number')
            cycle_number = None

            if cycle_number_param is not None:
                try:
                    cycle_number = int(cycle_number_param)
                except (ValueError, TypeError):
                    return self._json_response(
                        {"error": "Invalid 'cycle_number' parameter. Must be an integer."},
                        status=400,
                    )

            # Get cycles (all or specific)
            cycles_data = cycle_manager.get_cycles(cycle_number=cycle_number)

            if cycles_data is None:
                return self._json_response(
                    {"error": f"Cycle {cycle_number} not found in cache"}, status=404
                )

            return self._json_response(cycles_data)

        elif request.method == 'POST':
            try:
                # Get data from JSON body
                if not request.content_type or 'application/json' not in request.content_type:
                    return self._json_response(
                        {"error": "Content-Type must be application/json"}, status=400
                    )

                data = request.get_json(force=True)

                # Validate required fields
                if 'cycle_number' not in data:
                    return self._json_response(
                        {"error": "Missing required field: 'cycle_number'"}, status=400
                    )

                cycle_number = data['cycle_number']

                # Get the cycle (must exist - don't create)
                cycles_data = cycle_manager.get_cycles(cycle_number=cycle_number)
                if not cycles_data:
                    return self._json_response(
                        {"error": f"Cycle {cycle_number} not found"}, status=404
                    )

                # Prepare update kwargs
                update_kwargs = {}
                if 'failure_reason' in data:
                    update_kwargs['failure_reason'] = str(data['failure_reason'])
                if 'metadata' in data:
                    if not isinstance(data['metadata'], dict):
                        return self._json_response(
                            {"error": "'metadata' must be a dictionary"}, status=400
                        )
                    update_kwargs['metadata'] = data['metadata']

                # At least one field must be provided
                if not update_kwargs:
                    return self._json_response(
                        {"error": "Must provide at least 'failure_reason' or 'metadata'"},
                        status=400,
                    )

                # Get the cycle object and update
                cycle = cycle_manager._cycles.get(cycle_number)
                cycle.update(**update_kwargs)

                # Get updated cycle info
                cycle_info = cycle.to_dict()

                return self._json_response(
                    {
                        "status": "updated",
                        "cycle": cycle_info,
                    },
                    status=200,
                )

            except Exception as e:
                self._logger.exception("Error updating cycle")
                return self._json_response(
                    {"error": f"Internal server error: {str(e)}"}, status=500
                )

        else:
            return self._json_response(
                {"error": "Method not allowed. Use GET or POST."}, status=405
            )

    def _json_response(self, data, status=200):
        """Helper to create JSON response."""
        return Response(
            json.dumps(data, indent=2),
            status=status,
            mimetype='application/json',
            headers={'Access-Control-Allow-Origin': '*'},
        )


class HttpServerManager:
    """
    Manages the HTTP server lifecycle for the launcher REST API.

    Handles starting and stopping the HTTP server in a background thread,
    providing clean lifecycle management for the launcher's HTTP API.

    Usage:
        manager = HttpServerManager(handler, host='0.0.0.0', port=8080)
        manager.start()
        try:
            # ... do work ...
        finally:
            manager.stop()
    """

    def __init__(self, handler: LauncherHttpHandler, host: str = '0.0.0.0', port: int = 2025):
        """
        Initialize HTTP server manager.

        Args:
            handler: LauncherHttpHandler instance
            host: Host address to bind to
            port: Port to listen on (default: 2025)
        """
        self.handler = handler
        self.host = host
        self.port = port
        self.server: Optional[WSGIServer] = None
        self.thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger(LogConfig.name)

    def start(self):
        """Start HTTP server in background thread."""
        try:
            self.server = make_server(
                self.host, self.port, self.handler, server_class=ThreadedWSGIServer
            )
            self.thread = threading.Thread(
                target=self.server.serve_forever, daemon=True, name="launcher-http-server"
            )
            self.thread.start()
            self._logger.info(f"Launcher HTTP server started on http://{self.host}:{self.port}")
            self._logger.info("  Endpoints: /status (GET), /cycles (GET/POST)")
        except Exception as e:
            self._logger.error(f"Failed to start launcher HTTP server: {e}")
            raise

    def stop(self):
        """Gracefully shutdown the HTTP server."""
        if self.server:
            self._logger.info("Shutting down launcher HTTP server...")
            self.server.shutdown()
            self.server.server_close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self._logger.info("Launcher HTTP server stopped")
