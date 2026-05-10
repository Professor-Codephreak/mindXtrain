// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.26;

/// @title X402Receiver
/// @notice Receives x402 HTTP 402 payment proofs and validates Algorand settlement
///         hashes against an off-chain facilitator (parsec-wallet or Coinbase).
/// @dev Cypherpunk2048 standard: immutable, no proxy, no admin. The facilitator
///      address is fixed at deploy time; rotating it requires deploying a new
///      contract.
contract X402Receiver {
    address public immutable facilitator;
    bytes32 public immutable assetIdHash; // hash of (chain, asset_id) tuple, e.g. ("algorand", 203977300)

    mapping(bytes32 => bool) public seen;

    event PaymentValidated(
        bytes32 indexed invoiceId,
        bytes32 indexed settlementProof,
        address indexed payer,
        uint256 amount
    );

    error AlreadySeen(bytes32 invoiceId);
    error NotFacilitator(address sender);

    constructor(address facilitator_, bytes32 assetIdHash_) {
        facilitator = facilitator_;
        assetIdHash = assetIdHash_;
    }

    /// @notice The facilitator submits a settlement proof; this contract records it.
    /// @dev Off-chain x402 flow: caller pays the Algorand asset, parsec-wallet
    ///      builds an EIP-712 attestation, the facilitator submits it here.
    function recordSettlement(
        bytes32 invoiceId,
        bytes32 settlementProof,
        address payer,
        uint256 amount
    ) external {
        if (msg.sender != facilitator) revert NotFacilitator(msg.sender);
        if (seen[invoiceId]) revert AlreadySeen(invoiceId);
        seen[invoiceId] = true;
        emit PaymentValidated(invoiceId, settlementProof, payer, amount);
    }
}
