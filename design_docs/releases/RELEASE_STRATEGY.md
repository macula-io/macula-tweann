# Macula-TWEANN Release Strategy

## Overview

This document outlines the versioning strategy and release plan for migrating DXNN2 to macula-tweann. The strategy follows semantic versioning (SemVer) with a focus on incremental, testable releases that transform the codebase from research-quality to production-ready code.

## Versioning Philosophy

### Semantic Versioning

- **MAJOR (1.x.x)**: Production-ready release with stable API
- **MINOR (0.x.0)**: Feature-complete milestones during development
- **PATCH (0.x.x)**: Bug fixes and minor improvements within a milestone

### Pre-1.0 Development

All versions before 1.0.0 are considered development releases:
- API may change between minor versions
- Each minor version represents a significant refactoring milestone
- Quality gates must pass before advancing to next version

## Release Timeline

**Total Duration**: 15-18 weeks to v1.0.0

```
Week 1-2:   v0.1.0 - Project Bootstrap
Week 2-3:   v0.2.0 - Core Primitives
Week 3-5:   v0.3.0 - Neural Core       } Foundation Phase
------------------------------------------------
Week 5-7:   v0.4.0 - Network Lifecycle
Week 7-9:   v0.5.0 - Evolution Engine  } Structural Phase
Week 9-11:  v0.6.0 - Population Management
------------------------------------------------
Week 11-13: v0.7.0 - Hardened Core
Week 13-15: v0.8.0 - Process Safety    } Robustness Phase
------------------------------------------------
Week 15-17: v0.9.0 - Optimized
Week 17-18: v1.0.0 - Production Ready  } Performance Phase
```

## Phase Overview

### Phase 1: Foundation (v0.1.0 - v0.3.0)

**Duration**: 5 weeks
**Focus**: Establish test infrastructure, refactor pure functions, document core modules

**Objectives**:
- Create comprehensive test infrastructure
- Refactor pure mathematical functions with TDD
- Establish type specifications and documentation standards
- Enable dialyzer static analysis

**Key Deliverables**:
- Test directory structure with unit/integration layout
- Refactored `signal_aggregator.erl`, `functions.erl`, `plasticity.erl`
- Type specifications for all core data types
- Module-level documentation for neural components

### Phase 2: Structural (v0.4.0 - v0.6.0)

**Duration**: 6 weeks
**Focus**: Refactor complex modules, break down large functions, improve architecture

**Objectives**:
- Decompose `cortex.erl` state machine
- Modularize `exoself.erl` initialization
- Consolidate duplicate code in `genome_mutator.erl`
- Extract utility modules

**Key Deliverables**:
- State record-based cortex with clear transitions
- Modular network lifecycle management
- Unified mutation operator framework
- Population management with clear species handling

### Phase 3: Robustness (v0.7.0 - v0.8.0)

**Duration**: 4 weeks
**Focus**: Fix bugs, improve error handling, add process safety

**Objectives**:
- Fix all identified bugs (typos, logic errors)
- Update deprecated APIs (now(), random module)
- Add timeouts to receive loops
- Implement process linking and monitoring

**Key Deliverables**:
- Zero known bugs
- OTP 25+ compatibility
- Comprehensive error handling
- Crash recovery mechanisms

### Phase 4: Performance (v0.9.0 - v1.0.0)

**Duration**: 3 weeks
**Focus**: Optimize hot paths, clean up dead code, prepare for production

**Objectives**:
- Profile and optimize critical paths
- Remove dead code and unused fields
- Complete documentation
- Final integration testing

**Key Deliverables**:
- Performance benchmarks
- Clean codebase with no dead code
- Complete API documentation
- Production deployment guide

## Quality Gates

Every release must pass these quality gates before advancement:

### Mandatory Gates

1. **Test Coverage**
   - Unit tests: 100% coverage for refactored modules
   - Integration tests: All process interactions covered
   - All tests pass (zero failures)

2. **Static Analysis**
   - Dialyzer: Zero warnings
   - Linting: Pass style checks

3. **Documentation**
   - Module documentation: All modules documented
   - Function documentation: All public functions
   - Type specifications: All exported functions

4. **Code Quality**
   - No cryptic abbreviations in refactored code
   - Maximum 1 level of nesting
   - Pattern matching over case/if

### Phase-Specific Gates

**Foundation Phase**:
- Pure functions are isolated and testable
- Type system enables dialyzer

**Structural Phase**:
- No function exceeds 50 lines
- State records replace parameter lists
- No duplicate code blocks

**Robustness Phase**:
- All receive loops have timeouts
- Process crashes don't cascade
- Error returns are structured

**Performance Phase**:
- Hot paths profiled and optimized
- Benchmark results documented
- Memory usage acceptable

## Testing Strategy

### Test-Driven Development (TDD)

All refactoring follows the Red-Green-Refactor cycle:

1. **Red**: Write failing test specifying desired behavior
2. **Green**: Implement minimal code to pass
3. **Refactor**: Improve code quality while maintaining tests

### Test Organization

```
test/
  unit/
    signal_aggregator_test.erl
    functions_test.erl
    plasticity_test.erl
    neuron_test.erl
    cortex_test.erl
    genome_mutator_test.erl
  integration/
    network_evaluation_test.erl
    evolution_cycle_test.erl
    population_test.erl
  fixtures/
    sample_genotypes.erl
    test_helpers.erl
```

### Coverage Targets

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| signal_aggregator | 100% | N/A |
| functions | 100% | N/A |
| plasticity | 100% | N/A |
| neuron | 90% | Process interaction |
| cortex | 90% | Sync cycle |
| exoself | 80% | Lifecycle |
| genome_mutator | 90% | Mutation operators |
| population_monitor | 80% | Evolution cycle |

## Dependencies and Prerequisites

### External Dependencies

- **Erlang/OTP**: 25.0+ (for modern APIs)
- **Mnesia**: Distributed database (existing)
- **EUnit**: Test framework
- **Dialyzer**: Static analysis
- **Cover**: Code coverage

### Internal Dependencies

Modules must be refactored in dependency order:

```
Layer 1 (No dependencies):
  - functions.erl
  - signal_aggregator.erl

Layer 2 (Layer 1 only):
  - plasticity.erl

Layer 3 (Layers 1-2):
  - neuron.erl
  - sensor.erl
  - actuator.erl

Layer 4 (Layers 1-3):
  - cortex.erl

Layer 5 (Layers 1-4):
  - exoself.erl

Layer 6 (Database):
  - genotype.erl

Layer 7 (All layers):
  - genome_mutator.erl
  - population_monitor.erl
  - polis.erl
```

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing functionality | Medium | High | Comprehensive test suite before refactoring |
| Performance regression | Low | Medium | Profile before/after each phase |
| Incompatible API changes | Medium | Medium | Maintain backward compatibility layer |
| Mnesia migration issues | Low | High | Test database operations extensively |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Underestimated complexity | Medium | Medium | Buffer time in each phase |
| Hidden dependencies | Medium | Low | Thorough dependency analysis |
| Test writing overhead | High | Low | Parallelize test development |

## Release Artifacts

Each release includes:

1. **Source Code**: Tagged in git with version
2. **Release Notes**: Summary of changes, breaking changes, migration guide
3. **Test Results**: Coverage report, test output
4. **Documentation**: Updated API docs, examples
5. **Changelog Entry**: Added to CHANGELOG.md

## Success Criteria for v1.0.0

The project is ready for v1.0.0 when:

1. **Functionality**: All DXNN2 features preserved
2. **Quality**: Zero dialyzer warnings, zero known bugs
3. **Testing**: 90%+ overall coverage
4. **Documentation**: Complete API and architecture docs
5. **Performance**: Meets or exceeds DXNN2 benchmarks
6. **Readability**: All code self-documenting with clear naming
7. **Maintainability**: Easy to understand and extend

## Version Document Index

Each version has detailed planning documentation:

| Version | Document | Phase | Duration |
|---------|----------|-------|----------|
| 0.1.0 | [v0.1.0-project-bootstrap.md](v0.1.0-project-bootstrap.md) | Foundation | 1-2 weeks |
| 0.2.0 | [v0.2.0-core-primitives.md](v0.2.0-core-primitives.md) | Foundation | 1 week |
| 0.3.0 | [v0.3.0-neural-core.md](v0.3.0-neural-core.md) | Foundation | 2 weeks |
| 0.4.0 | [v0.4.0-network-lifecycle.md](v0.4.0-network-lifecycle.md) | Structural | 2 weeks |
| 0.5.0 | [v0.5.0-evolution-engine.md](v0.5.0-evolution-engine.md) | Structural | 2 weeks |
| 0.6.0 | [v0.6.0-population-management.md](v0.6.0-population-management.md) | Structural | 2 weeks |
| 0.7.0 | [v0.7.0-hardened-core.md](v0.7.0-hardened-core.md) | Robustness | 2 weeks |
| 0.8.0 | [v0.8.0-process-safety.md](v0.8.0-process-safety.md) | Robustness | 2 weeks |
| 0.9.0 | [v0.9.0-optimized.md](v0.9.0-optimized.md) | Performance | 2 weeks |
| 1.0.0 | [v1.0.0-production-ready.md](v1.0.0-production-ready.md) | Performance | 1 week |

## References

- [DXNN2_CODEBASE_ANALYSIS.md](../DXNN2_CODEBASE_ANALYSIS.md) - Detailed codebase analysis
- [README.md](../README.md) - Refactoring principles and standards
- [ANALYSIS_SUMMARY.txt](../ANALYSIS_SUMMARY.txt) - Executive summary
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Author**: Macula-TWEANN Development Team
