import os
from multiprocessing import Pool

import pytest

from nvidia_resiliency_ext.fault_tolerance.ipc_connector import IpcConnector


def _sender_process(rank):
    socket_path = '/tmp/test_ipc_socket'
    sender = IpcConnector(socket_path)
    sender.send(
        (
            rank,
            "Test Message 1",
        )
    )
    sender.send(
        (
            rank,
            "Test Message 2",
        )
    )
    sender.send(
        (
            rank,
            "STOP",
        )
    )


def test_ipc_connector_send_receive():
    socket_path = '/tmp/test_ipc_socket'

    receiver = IpcConnector(socket_path)
    receiver.start_receiving()

    # 2nd start receiving should fail
    with pytest.raises(Exception):
        receiver.start_receiving()

    # receiver should be empty
    assert not receiver.peek_received()
    assert not receiver.fetch_received()

    # clear on empty does nothing
    receiver.clear()
    assert not receiver.peek_received()
    assert not receiver.fetch_received()

    # try to send and receive a few times
    attempts = 4
    num_processes = 4
    ranks = range(num_processes)
    for _ in range(attempts):
        # send messages from sub-processes
        with Pool(processes=num_processes) as pool:
            _ = pool.map(_sender_process, ranks)
        # peek_received should not clear internal message queue
        assert len(receiver.peek_received()) == num_processes * 3
        assert len(receiver.peek_received()) == num_processes * 3
        for t in range(num_processes):
            assert (t, "Test Message 1") in receiver.peek_received()
            assert (t, "Test Message 2") in receiver.peek_received()
            assert (t, "STOP") in receiver.peek_received()
        # should be empty again after .clear
        receiver.clear()
        assert len(receiver.peek_received()) == 0
        # send messages from sub-processes again
        with Pool(processes=num_processes) as pool:
            _ = pool.map(_sender_process, ranks)
        # fetch_received clears the internal message queue
        received_messages = receiver.fetch_received()
        assert len(receiver.fetch_received()) == 0
        assert len(received_messages) == num_processes * 3
        for t in range(num_processes):
            assert (t, "Test Message 1") in received_messages
            assert (t, "Test Message 2") in received_messages
            assert (t, "STOP") in received_messages

    receiver.stop_receiving()
    assert not os.path.exists(socket_path)
    receiver.stop_receiving()  # no op
