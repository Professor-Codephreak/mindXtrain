// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.26;

import {Script, console} from "forge-std/Script.sol";
import {MindXTrainRegistry} from "../src/mindxtrain_registry.sol";
import {X402Receiver} from "../src/x402_receiver.sol";

contract Deploy is Script {
    function run() external {
        uint256 pk = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address facilitator = vm.envAddress("X402_FACILITATOR");
        // ("algorand", 203977300) — Algorand mainnet USDC ASA
        bytes32 assetIdHash = keccak256(abi.encode("algorand", uint256(203977300)));

        vm.startBroadcast(pk);
        MindXTrainRegistry registry = new MindXTrainRegistry();
        X402Receiver receiver = new X402Receiver(facilitator, assetIdHash);
        vm.stopBroadcast();

        console.log("MindXTrainRegistry deployed at:", address(registry));
        console.log("X402Receiver deployed at:", address(receiver));
    }
}
