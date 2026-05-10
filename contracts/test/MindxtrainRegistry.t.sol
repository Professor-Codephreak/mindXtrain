// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {MindXTrainRegistry} from "../src/mindxtrain_registry.sol";

contract MindXTrainRegistryTest is Test {
    MindXTrainRegistry registry;

    function setUp() public {
        registry = new MindXTrainRegistry();
    }

    function test_anchor_emits_event() public {
        bytes32 runId = keccak256("run-1");
        bytes32 yamlHash = keccak256("yaml");
        bytes32 datasetHash = keccak256("dataset");
        bytes32 checkpointHash = keccak256("checkpoint");
        bytes32 evalHash = keccak256("eval");

        vm.expectEmit(true, true, false, true);
        emit MindXTrainRegistry.ReceiptAnchored(
            runId,
            address(this),
            yamlHash,
            datasetHash,
            checkpointHash,
            evalHash,
            uint64(block.timestamp)
        );
        registry.anchor(runId, yamlHash, datasetHash, checkpointHash, evalHash);
    }

    function test_anchor_persists_receipt() public {
        bytes32 runId = keccak256("run-2");
        registry.anchor(
            runId, keccak256("y"), keccak256("d"), keccak256("c"), keccak256("e")
        );
        MindXTrainRegistry.Receipt memory r = registry.get(runId);
        assertEq(r.yamlHash, keccak256("y"));
        assertEq(r.publisher, address(this));
        assertGt(r.timestamp, 0);
        assertTrue(registry.exists(runId));
    }

    function test_anchor_rejects_overwrite() public {
        bytes32 runId = keccak256("run-3");
        registry.anchor(
            runId, keccak256("y"), keccak256("d"), keccak256("c"), keccak256("e")
        );
        vm.expectRevert(
            abi.encodeWithSelector(MindXTrainRegistry.ReceiptAlreadyExists.selector, runId)
        );
        registry.anchor(
            runId, keccak256("y2"), keccak256("d2"), keccak256("c2"), keccak256("e2")
        );
    }

    function test_exists_false_for_unknown() public view {
        assertFalse(registry.exists(keccak256("never-anchored")));
    }

    function testFuzz_anchor_distinct_run_ids(bytes32 a, bytes32 b) public {
        vm.assume(a != b);
        registry.anchor(a, bytes32(0), bytes32(0), bytes32(0), bytes32(0));
        registry.anchor(b, bytes32(0), bytes32(0), bytes32(0), bytes32(0));
        assertTrue(registry.exists(a));
        assertTrue(registry.exists(b));
    }
}
