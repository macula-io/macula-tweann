# Changelog

All notable changes to the macula-tweann project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- See individual version documents for detailed planning

---

## [0.11.0] - 2025-12-06

### Summary
**ONNX Export & Documentation Release** - Export trained networks to ONNX format for inference in Python, JavaScript, and other frameworks.

### Added

#### ONNX Export
- **network_onnx.erl** (~200 lines): Export evolved networks to ONNX format
  - `export/2` - Export network to ONNX binary file
  - `to_onnx/1` - Convert network to ONNX protobuf structure
  - Supports feedforward networks with standard activation functions
  - Compatible with ONNX Runtime, PyTorch, TensorFlow

#### Documentation
- **Academic references** added to README.md and guides/overview.md
  - Hasani et al. (2021) - Liquid Time-constant Networks
  - Hasani et al. (2022) - Closed-form Continuous-time Neural Networks
  - Stanley & Miikkulainen (2002) - NEAT
  - Sher (2012) - Handbook of Neuroevolution Through Erlang (DXNN2)

- **scripts/check-links.sh**: Documentation link quality checker

### Fixed
- Broken DXNN2 reference link in v0.3.1-architectural-alignment.md

### Test Results
- 270+ tests passing
- Dialyzer clean

---

## [0.10.0] - 2025-12-03

### Summary
**Liquid Time-Constant (LTC) Neurons Release** - First TWEANN library with LTC/CfC neuron support in Erlang/OTP.

LTC neurons enable adaptive temporal processing with input-dependent time constants. This is a major feature release that extends macula-tweann with continuous-time neural dynamics based on peer-reviewed research.

### Added

#### Core LTC Modules
- **ltc_dynamics.erl** (~380 lines): Core LTC/CfC computation engine
  - `evaluate_cfc/4,5` - CfC closed-form evaluation (~100x faster than ODE)
  - `evaluate_ode/5,6` - ODE-based evaluation (Euler integration)
  - `compute_backbone/3` - Time constant modulation network
  - `compute_head/2` - Target state computation
  - `compute_liquid_tau/4` - Adaptive time constant calculation
  - `clamp_state/2`, `reset_state/0` - State management utilities
  - Full EDoc with academic references

- **neuron_ltc.erl** (~280 lines): LTC-specific neuron process
  - Full process lifecycle with internal state persistence
  - CfC and ODE modes supported
  - Reset/get state operations
  - LTC parameter update support

#### LTC Evolution Support (genome_mutator.erl)
- `mutate_neuron_type/1` - Switch neurons between standard/ltc/cfc modes
- `mutate_time_constant/1` - Perturb tau (base time constant)
- `mutate_state_bound/1` - Perturb state bound A
- `mutate_ltc_weights/1` - Perturb backbone/head network weights
- `select_ltc_neuron/1` - Helper to select LTC/CfC neurons

#### Rust NIF LTC Support (native/src/lib.rs)
- `evaluate_cfc/4` - Fast CfC evaluation in Rust
- `evaluate_cfc_with_weights/6` - CfC with custom backbone/head weights
- `evaluate_ode/5` - ODE-based evaluation in Rust
- `evaluate_ode_with_weights/7` - ODE with custom weights
- `evaluate_cfc_batch/4` - Batch CfC evaluation for time series

#### Extended Records
- **records.hrl**: Extended `#neuron` record with LTC fields
  - `neuron_type` (standard | ltc | cfc)
  - `time_constant` (τ - base time constant)
  - `state_bound` (A - prevents state explosion)
  - `ltc_backbone_weights` - f() backbone network
  - `ltc_head_weights` - h() head network
  - `internal_state` - x(t) persistent state

- **types.hrl**: New LTC type specifications
  - `neuron_type()`, `time_constant()`, `state_bound()`
  - `internal_state()`, `time_step()`
  - `ltc_backbone_weights()`, `ltc_head_weights()`
  - `ltc_params()` map type

#### Documentation
- **guides/ltc-neurons.md**: Comprehensive LTC concepts guide
  - Mathematical foundations (LTC ODE, CfC closed-form)
  - Neuron types comparison table
  - Implementation details and key properties
  - Use cases and academic references

- **guides/ltc-usage-guide.md**: Practical usage guide
  - API reference with examples
  - Parameter tuning guide
  - Time series processing examples
  - Troubleshooting section

- **design_docs/diagrams/ltc-neuron-architecture.svg**: Architecture diagram
- **design_docs/diagrams/ltc-vs-standard-neurons.svg**: Comparison diagram

### Changed
- **constructor.erl**: Extended to spawn LTC neurons
  - `spawn_neuron_by_type/2` dispatches based on `neuron_type`
  - `spawn_standard_neuron/2` for standard neurons
  - `spawn_ltc_neuron/3` for ltc/cfc neurons

- **README.md**: Updated with LTC as primary feature
- **rebar.config**: Added LTC guides to ex_doc configuration

### Performance
- CfC evaluation: ~100x faster than ODE-based LTC
- State bounded dynamics prevent numerical overflow
- Configurable time constants for different response speeds

### Academic References
- Hasani, R., Lechner, M., et al. (2021). "Liquid Time-constant Networks." AAAI 2021.
- Hasani, R., Lechner, M., et al. (2022). "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence.

### Test Results
- 468 tests passing (including 68 new LTC tests: 45 core + 11 mutation + 12 NIF)
- Dialyzer clean (1 pre-existing warning)

---

## Migration from DXNN2

### Key Differences

1. **Naming**: All cryptic abbreviations replaced
   - `idps` -> `weighted_inputs`
   - `af` -> `activation_function`
   - `pf` -> `plasticity_function`
   - `vl` -> `vector_length`
   - See RELEASE_STRATEGY.md Section 6.1 for complete mapping

2. **State Management**: Records instead of parameter lists
   - cortex: 10 parameters -> cortex_state record
   - neuron: 14 parameters -> neuron_state record
   - exoself: 24 parameters -> exoself_state record

3. **Error Handling**: Structured errors instead of exit()
   - `exit("ERROR...")` -> `{error, {type, reason}}`

4. **APIs**: Modern OTP
   - `now()` -> `erlang:monotonic_time()`
   - `random` -> `rand`

### Migration Steps

1. Update type imports from types.hrl
2. Update record field names
3. Update function return types for error cases
4. Update time-related code
5. Run test suite to verify

---

## References

- `design_docs/DXNN2_CODEBASE_ANALYSIS.md` - DXNN2 original codebase analysis (internal)
- `design_docs/README.md` - Refactoring principles (internal)
- [Release Strategy](RELEASE_STRATEGY.md)

---

[Unreleased]: https://github.com/macula-io/macula-tweann/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/macula-io/macula-tweann/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/macula-io/macula-tweann/releases/tag/v0.10.0
