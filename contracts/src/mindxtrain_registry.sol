// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.26;

/// @title MindXTrainRegistry
/// @notice Write-once anchoring contract for mindXtrain run receipts.
/// @dev Cypherpunk2048 standard: immutable, no proxy, no Ownable, no admin keys,
///      no pause, no setter. Receipts cannot be overwritten or revoked.
contract MindXTrainRegistry {
    struct Receipt {
        bytes32 yamlHash;
        bytes32 datasetCidHash;
        bytes32 checkpointCidHash;
        bytes32 evalReportHash;
        address publisher;
        uint64 timestamp;
    }

    mapping(bytes32 => Receipt) private _receipts;

    event ReceiptAnchored(
        bytes32 indexed runId,
        address indexed publisher,
        bytes32 yamlHash,
        bytes32 datasetCidHash,
        bytes32 checkpointCidHash,
        bytes32 evalReportHash,
        uint64 timestamp
    );

    error ReceiptAlreadyExists(bytes32 runId);

    function anchor(
        bytes32 runId,
        bytes32 yamlHash,
        bytes32 datasetCidHash,
        bytes32 checkpointCidHash,
        bytes32 evalReportHash
    ) external {
        if (_receipts[runId].timestamp != 0) revert ReceiptAlreadyExists(runId);
        Receipt memory r = Receipt({
            yamlHash: yamlHash,
            datasetCidHash: datasetCidHash,
            checkpointCidHash: checkpointCidHash,
            evalReportHash: evalReportHash,
            publisher: msg.sender,
            timestamp: uint64(block.timestamp)
        });
        _receipts[runId] = r;
        emit ReceiptAnchored(
            runId,
            msg.sender,
            yamlHash,
            datasetCidHash,
            checkpointCidHash,
            evalReportHash,
            r.timestamp
        );
    }

    function get(bytes32 runId) external view returns (Receipt memory) {
        return _receipts[runId];
    }

    function exists(bytes32 runId) external view returns (bool) {
        return _receipts[runId].timestamp != 0;
    }
}
