# Macula TWEANN

**Topology and Weight Evolving Artificial Neural Networks for Erlang**

## Introduction

Macula TWEANN is an evolutionary neural network library that implements the TWEANN paradigm - allowing neural networks to evolve both their topology (structure) and weights. Networks can add neurons, connections, sensors, and actuators while optimizing their synaptic weights through natural selection.

Based on DXNN2 by Gene Sher (from "Handbook of Neuroevolution Through Erlang"), this library provides a production-ready implementation with modern Erlang practices, process safety, and comprehensive logging.

## Architecture

![Module Dependencies](assets/module-dependencies.svg)

The library follows a layered architecture:
- **Core**: Genotype storage and morphology definitions
- **Network**: Process-based neural network components (cortex, sensors, neurons, actuators)
- **Evolution**: Mutation operators, crossover, and selection algorithms
- **Population**: Multi-agent evolution with speciation

See the [Architecture Guide](architecture.html) for details.

## Documentation

### Getting Started
- [TWEANN Basics](tweann-basics.html) - What is TWEANN and how does it work?
- [Installation](installation.html) - Add to your project
- [Quick Start](quickstart.html) - Basic usage examples

### Architecture
- [C4 Architecture Model](c4-architecture.html) - Multi-level architectural views (Context, Container, Component, Code)
- [Architecture Details](architecture.html) - Layer-by-layer system design
- [Vision: Distributed Mega-Brain](vision-distributed-mega-brain.html) - Planet-scale evolutionary intelligence on Macula mesh
- [Addendum: Military & Civil Resilience](addendum-military-civil-resilience.html) - Defense, security, and infrastructure implications
- [Addendum: Anti-Drone Defense](addendum-anti-drone-defense.html) - Counter-swarm systems and autonomous threat mitigation

### Core Concepts
See the module documentation for detailed API reference on:
- **Genotypes** - Neural network blueprints (genotype module)
- **Phenotypes** - Running network processes (constructor, exoself modules)
- **Evolution** - Mutation and selection (genome_mutator, selection_algorithm modules)
- **Morphologies** - Problem domains (morphology module)
- **Speciation** - Diversity preservation (species_identifier module)
- **Multi-objective** - Pareto optimization (fitness_postprocessor module)
- **Process Safety** - Timeouts and crash handling (cortex, neuron modules)

### API Reference
- [Module Index](api-reference.html) - Complete API documentation

## Acknowledgements

Based on DXNN2 by Gene Sher. Adapted and extended by [Macula.io](https://macula.io).

## Academic References

### Core TWEANN/NEAT Papers

- **Sher, G.I.** (2013). *Handbook of Neuroevolution Through Erlang*. Springer.
  Primary reference for DXNN2 architecture and Erlang-specific patterns.

- **Stanley, K.O. & Miikkulainen, R.** (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.
  Foundational NEAT paper introducing speciation and structural innovation protection.

### LTC/CfC Neurons

- **Hasani, R., Lechner, M., et al.** (2021). Liquid Time-constant Networks. *AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666.
  Introduces adaptive time-constant neurons with continuous-time dynamics.

- **Hasani, R., Lechner, M., et al.** (2022). Closed-form Continuous-time Neural Networks. *Nature Machine Intelligence*, 4, 992-1003.
  CfC approximation enabling ~100x speedup over ODE-based evaluation.

### Foundational Work

- **Holland, J.H.** (1975). *Adaptation in Natural and Artificial Systems*. MIT Press.
  Foundational text on genetic algorithms.

- **Yao, X.** (1999). Evolving Artificial Neural Networks. *Proceedings of the IEEE*, 87(9), 1423-1447.
  Comprehensive neuroevolution survey.

## Related Projects

### Macula Ecosystem

- **macula** - HTTP/3 mesh networking for distributed neuroevolution
- **macula_neuroevolution** - Population-based evolutionary training engine

### External

- **DXNN2** - Gene Sher's original Erlang implementation
- **NEAT-Python** - Python NEAT implementation
- **LTC Reference** - MIT/ISTA reference LTC implementation

## License

Apache License 2.0
