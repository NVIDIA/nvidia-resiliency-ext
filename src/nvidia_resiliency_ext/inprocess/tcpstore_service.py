#!/usr/bin/env python3
"""
External TCPStore Service

This service runs TCPStore independently of training processes to solve
the barrier problem across process restarts. The service persists across
rank restarts and provides a stable store for distributed training.

IMPORTANT: This service can be started on all ranks, but only Rank 0 will
actually host the TCPStore. Other ranks will exit gracefully without hosting
the service.
"""

import argparse
import logging
import os
import signal
import socket
import sys
import time

import torch.distributed as dist


class TCPStoreService:
    """
    External TCPStore service that runs independently of training processes.

    This service provides a persistent TCPStore that survives rank restarts,
    solving the barrier problem where ranks get stuck waiting on old stores.

    NOTE: Only Rank 0 will host the TCPStore. Other ranks can start this
    service but will exit gracefully without hosting the service.
    """

    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        timeout: int = 300,
        use_libuv: bool = True,
        log_level: str = "INFO",
    ):
        self.host = host
        self.port = port
        self.world_size = world_size
        self.timeout = timeout
        self.use_libuv = use_libuv
        self.store = None
        self.running = False

        self.logger = logging.getLogger(__name__)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the TCPStore service."""
        self.logger.info(f"Starting TCPStore service on {self.host}:{self.port}")

        try:
            # Create TCPStore server
            self.store = dist.TCPStore(
                host_name=self.host,
                port=self.port,
                world_size=self.world_size,
                is_master=True,
                timeout=time.time() + self.timeout,
                multi_tenant=True,
                use_libuv=self.use_libuv,
                wait_for_workers=False,  # Don't wait for workers to start
            )

            self.running = True

            # Keep the service running
            self._run()

        except Exception as e:
            self.logger.error(f"Failed to start TCPStore service: {e}")
            raise

    def _run(self):
        """Main service loop."""
        self.logger.info("TCPStore service is running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)

                # Optional: Add health checks or monitoring here
                # For example, check if store is still responsive

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.stop()

    def stop(self):
        """Stop the TCPStore service."""
        if self.running:
            self.logger.info("Stopping TCPStore service...")
            self.running = False

            if self.store is not None:
                try:
                    # Note: TCPStore doesn't have a clean shutdown method
                    # The service will be cleaned up when the process exits
                    del self.store
                except Exception:
                    pass
                self.store = None

            self.logger.info("TCPStore service stopped")

    def get_connection_info(self) -> dict:
        """Get connection information for clients."""
        return {
            'host': self.host,
            'port': self.port,
            'world_size': self.world_size,
            'timeout': self.timeout,
            'use_libuv': self.use_libuv,
        }


def main():
    """Main entry point for the external TCPStore service."""
    parser = argparse.ArgumentParser(
        description="External TCPStore Service - Can be started on all ranks, but only Rank 0 hosts the service"
    )
    parser.add_argument(
        '--host', default='0.0.0.0', help='Host to bind the TCPStore server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,  # Will be set to MASTER_PORT + port_offset if not specified
        help='Port to bind the TCPStore server to (default: MASTER_PORT + port_offset)',
    )
    parser.add_argument(
        '--port-offset',
        type=int,
        default=2,
        help='Port offset from MASTER_PORT (default: 2)',
    )
    parser.add_argument(
        '--world-size',
        type=int,
        required=True,
        help='Number of workers that will connect to this store',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for store operations (default: 300)',
    )
    parser.add_argument(
        '--use-libuv',
        action='store_true',
        default=True,
        help='Use libuv backend for better performance (default: True)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )

    args = parser.parse_args()

    # Setup logging first
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=getattr(logging, args.log_level.upper()),
    )
    logger = logging.getLogger(__name__)

    # Get rank from environment (default to 0 if not set)
    rank = int(os.environ.get('RANK', '0'))
    if rank != 0:
        sys.exit(0)

    # Only Rank 0 creates the actual TCPStore server
    logger.info(f"Rank {rank}: Hosting TCPStore service (Rank 0)")

    # Get environment variables for debugging
    hostname = socket.gethostname()
    master_addr = os.environ.get('MASTER_ADDR')
    master_port = os.environ.get('MASTER_PORT')

    # Require MASTER_ADDR and MASTER_PORT to be set
    if master_addr is None:
        logger.error("MASTER_ADDR environment variable is not set")
        sys.exit(1)

    if master_port is None:
        logger.error("MASTER_PORT environment variable is not set")
        sys.exit(1)

    # Set default port to MASTER_PORT + port_offset if not specified
    if args.port is None:
        args.port = int(master_port) + args.port_offset

    # Check if hostname matches MASTER_ADDR and log warning if not
    if hostname != master_addr:
        logger.warning("Hostname mismatch on Rank 0!")
        logger.warning(f"  Current hostname: {hostname}")
        logger.warning(f"  MASTER_ADDR: {master_addr}")
        logger.warning("This may cause connection issues if other ranks cannot reach this host")
        logger.warning("Ensure other ranks can connect to the host specified by MASTER_ADDR")
    else:
        logger.info(f"✓ Hostname validation passed: {hostname} matches MASTER_ADDR")

    logger.info("==== Creating TCPStore server ====")
    logger.info(
        f"Hostname: {hostname}, MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}, Port Offset: {args.port_offset}"
    )
    logger.info(
        f"RANK: {rank}, Service Host: {args.host}, Service Port: {args.port}, World Size: {args.world_size}"
    )

    # Create and start the service
    service = TCPStoreService(
        host=args.host,
        port=args.port,
        world_size=args.world_size,
        timeout=args.timeout,
        use_libuv=args.use_libuv,
        log_level=args.log_level,
    )

    try:
        service.start()
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
