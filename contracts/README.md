# mindXtrain on-chain contracts

Foundry workspace for the immutable, no-admin, no-proxy contracts that anchor
mindXtrain run receipts and receive x402 settlement proofs.

## Contracts

- `src/mindxtrain_registry.sol` — write-once anchoring of `(yamlHash, datasetCidHash, checkpointCidHash, evalReportHash)` per `runId`.
- `src/x402_receiver.sol` — records x402 settlement proofs from a fixed facilitator address.

Both follow the cypherpunk2048 standard: no upgradeable proxies, no `Ownable`, no admin keys, no pause function, no setters. Rotating any parameter requires a fresh deployment.

## Day-2 setup (on the MI300X droplet, where Foundry installs alongside the training stack)

```bash
curl -L https://foundry.paradigm.xyz | bash && foundryup
cd contracts
forge install foundry-rs/forge-std --no-git
forge build
forge test --gas-report
```

## Deploy

```bash
export DEPLOYER_PRIVATE_KEY=0x...
export X402_FACILITATOR=0x...     # parsec-wallet or Coinbase facilitator
export BASE_SEPOLIA_RPC_URL=https://sepolia.base.org
forge script script/Deploy.s.sol --rpc-url base_sepolia --broadcast --verify
```

Day-3 deploy moves to `--rpc-url base` (mainnet).
