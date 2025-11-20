# Changelog

All notable changes to the macula-tweann project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- See individual version documents for detailed planning

---

## [1.0.0] - TBD

### Summary
First production release of macula-tweann. Complete refactoring of DXNN2 into production-quality Erlang code.

### Added
- Complete API documentation (EDoc)
- Architecture documentation (ARCHITECTURE.md)
- Configuration guide (CONFIGURATION.md)
- Deployment guide (DEPLOYMENT.md)
- Health check module (tweann_health)
- Release configuration for production builds

### Changed
- Finalized all documentation
- Archived benchmark baselines

### Quality Metrics
- Test coverage: 90%+ core modules
- Dialyzer warnings: 0
- Documentation: 100% public functions

---

## [0.9.0] - TBD

### Summary
Performance optimization release. Profiled hot paths, optimized critical operations, removed dead code.

### Added
- Performance benchmark suite (tweann_benchmark)
- Cached tanh for activation optimization
- Batch random number generation for perturbation

### Changed
- Optimized signal aggregation with list comprehensions
- Optimized weight perturbation with batch operations
- Converted Mnesia reads to dirty reads where safe

### Removed
- Dead code from functions.erl
- Unused record fields from neuron state
- Commented-out code throughout codebase

### Performance
- Forward propagation: <1ms for 100 neurons
- Weight perturbation: <1ms for 1000 weights
- Mnesia read: <100us per record

---

## [0.8.0] - TBD

### Summary
Process safety release. Added crash propagation, timeouts, and recovery strategies.

### Added
- Process monitors for cortex and critical processes
- Timeouts in all receive loops
- Crash recovery strategies (transient/permanent classification)
- Safe termination propagation

### Changed
- All spawns now use spawn_link
- EXIT messages properly handled
- Processes terminate cleanly on parent crash

### Configuration
- Configurable timeout values
- Recovery retry limits

---

## [0.7.0] - TBD

### Summary
Bug fix and API modernization release. Fixed all identified bugs, updated deprecated APIs.

### Fixed
- **cortex.erl line 65**: Typo "termiante" corrected to "terminate"
- **neuron.erl line 299**: Plasticity perturbation now returns updated values
- **genome_mutator.erl line 26**: Typo "PARAMTERS" corrected to "PARAMETERS"

### Changed
- Replaced `now()` with `erlang:monotonic_time()` throughout
- Replaced `random` module with `rand` module throughout
- Added structured error returns (no more `exit()` with strings)

### Added
- time_utils module for modern time operations
- tweann_logger module for structured logging
- Error type definitions for all operations

---

## [0.6.0] - TBD

### Summary
Population management release. Refactored population_monitor with clear state management.

### Changed
- **population_monitor.erl**:
  - Renamed all state fields to clear names
  - "activeAgent_IdPs" renamed to "active_agent_processes"
  - Decomposed gen_server handlers
  - Documented evolutionary generation loop

### Added
- Species management functions
- Selection phase documentation
- Multi-objective fitness handling

### Integration
- Verified integration with genome_mutator
- End-to-end evolution cycle tested

---

## [0.5.0] - TBD

### Summary
Evolution engine release. Consolidated mutation operators, extracted utilities.

### Fixed
- **genome_mutator.erl line 26**: Typo in constant name

### Changed
- **genome_mutator.erl**:
  - Consolidated 4 duplicate mutation functions into 1 parametric function
  - Decomposed add_neuron into modular steps
  - Documented all magic numbers

### Added
- **selection_utils.erl**: Roulette wheel selection, random selection
- **perturbation_utils.erl**: Weight perturbation, plasticity perturbation

### Documented
- Mutation operator categories
- Selection algorithms
- Momentum-based perturbation formula

---

## [0.4.0] - TBD

### Summary
Network lifecycle release. Refactored cortex and exoself with state records.

### Fixed
- **cortex.erl line 65**: Typo "termiante" corrected to "terminate"

### Changed
- **cortex.erl**:
  - Converted 10-parameter loop to state record
  - Consolidated 3 identical fitness update clauses
  - Replaced custom vector_add with lists:zipwith
  - Removed process dictionary usage

- **exoself.erl**:
  - Renamed all state fields (24 fields)
  - "idsNpids" renamed to "id_to_process_map"
  - Decomposed 50-line prep function into modules

### Added
- State transition documentation for cortex
- Tuning algorithm documentation for exoself

---

## [0.3.0] - TBD

### Summary
Neural core release. Refactored plasticity and neuron modules.

### Changed
- **plasticity.erl**:
  - Consolidated Hebbian variants into parametric function
  - Documented all learning rules with formulas
  - Reduced nesting in weight update functions

- **neuron.erl**:
  - Renamed all cryptic state fields
  - "si_pidps_bl" renamed to "weighted_inputs_baseline"
  - Removed ?RO_SIGNAL macro
  - Extracted weight perturbation to function
  - Documented saturation limits

### Added
- Mathematical documentation for all plasticity rules
- Weight management lifecycle documentation
- Heredity type (Darwinian/Lamarckian) documentation

---

## [0.2.0] - TBD

### Summary
Core primitives release. Refactored pure mathematical functions.

### Changed
- **signal_aggregator.erl**:
  - Added comprehensive module documentation
  - Documented weight tuple format {W, DW, LP, LPs}
  - Added type specifications for all functions

- **functions.erl**:
  - Removed all dead code (commented functions)
  - Documented all activation functions with mathematical definitions
  - Added type specifications

### Added
- Test suite for signal_aggregator (100% coverage)
- Test suite for functions (100% coverage)

---

## [0.1.0] - TBD

### Summary
Project bootstrap release. Established test infrastructure and type specifications.

### Added
- **Test infrastructure**:
  - test/unit/ directory structure
  - test/integration/ directory structure
  - test_helpers.erl for common test utilities
  - sample_genotypes.erl for test fixtures

- **Type specifications**:
  - types.hrl with core type definitions
  - Documented weight_spec type
  - Documented neuron_id, sensor_id, actuator_id types

- **Static analysis**:
  - Dialyzer configuration in rebar.config
  - Initial PLT creation

- **Documentation templates**:
  - Module documentation template
  - Function documentation template

### Changed
- **records.hrl**:
  - Added inline documentation for all fields
  - Documented weight tuple format

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

## Release Schedule

| Version | Phase | Target Week | Status |
|---------|-------|-------------|--------|
| 0.1.0 | Foundation | 1-2 | Planned |
| 0.2.0 | Foundation | 2-3 | Planned |
| 0.3.0 | Foundation | 3-5 | Planned |
| 0.4.0 | Structural | 5-7 | Planned |
| 0.5.0 | Structural | 7-9 | Planned |
| 0.6.0 | Structural | 9-11 | Planned |
| 0.7.0 | Robustness | 11-13 | Planned |
| 0.8.0 | Robustness | 13-15 | Planned |
| 0.9.0 | Performance | 15-17 | Planned |
| 1.0.0 | Performance | 17-18 | Planned |

---

## References

- [DXNN2 Original Codebase](../DXNN2_CODEBASE_ANALYSIS.md)
- [Refactoring Principles](../README.md)
- [Release Strategy](RELEASE_STRATEGY.md)

---

[Unreleased]: https://github.com/macula-io/macula-tweann/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/macula-io/macula-tweann/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/macula-io/macula-tweann/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/macula-io/macula-tweann/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/macula-io/macula-tweann/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/macula-io/macula-tweann/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/macula-io/macula-tweann/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/macula-io/macula-tweann/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/macula-io/macula-tweann/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/macula-io/macula-tweann/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/macula-io/macula-tweann/releases/tag/v0.1.0
