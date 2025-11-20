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

### References

- Sher, Gene I. *"Handbook of Neuroevolution Through Erlang"* (2013)
- Stanley, Kenneth O., and Risto Miikkulainen. *"Evolving neural networks through augmenting topologies."* Evolutionary Computation 10.2 (2002): 99-127.

## License

Apache License 2.0
