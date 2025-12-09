PROTO_DIR := src/nvidia_resiliency_ext/shared_utils/proto
PROTO_FILE := $(PROTO_DIR)/nvhcd.proto

.PHONY: gen
gen:
	python -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=$(PROTO_DIR) --pyi_out=$(PROTO_DIR) --grpc_python_out=$(PROTO_DIR) $(PROTO_FILE)

.PHONY: clean-protos
clean:
	rm -f $(PROTO_DIR)/nvhcd_pb2.py $(PROTO_DIR)/nvhcd_pb2.pyi $(PROTO_DIR)/nvhcd_pb2_grpc.py

.PHONY: deps
deps:
	poetry install

.PHONY: update-deps
update-deps:
	poetry update
