This is test of fault handing in simulated (dockerized) multi node setup.
You need docker and docker-compose installed + NVIDIA extensions that allow to use GPU.
There should be GPU available, one is enough.

Run command (current dir should be repo root dir)
<repo>/fault_tolerance$ docker-compose  -f tests/sim-multinode/compose-c10d.yaml up --force-recreate --build
OR
<repo>/fault_tolerance$ docker-compose  -f tests/sim-multinode/compose-etcd.yaml up --force-recreate --build

Expected result is "SUCCEEDED (...)" line printed at the end.