# macula-tweann

**Topology and Weight Evolving Artificial Neural Networks for Erlang**

[![Hex.pm](https://img.shields.io/hexpm/v/macula_tweann.svg)](https://hex.pm/packages/macula_tweann)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/macula_tweann/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Macula TWEANN is an evolutionary neural network library for Erlang, based on DXNN2 by Gene Sher (from his book "Handbook of Neuroevolution Through Erlang"). It implements the TWEANN (Topology and Weight Evolving Artificial Neural Networks) paradigm, allowing neural networks to evolve both their connection weights and their structural topology.

## Features

- **Topology Evolution**: Networks can add/remove neurons, connections, sensors, and actuators
- **Weight Evolution**: Synaptic weights evolve through perturbation and selection
- **Multiple Morphologies**: XOR, pole balancing, T-maze navigation, and more
- **Speciation**: Behavioral fingerprinting for diversity preservation
- **Multi-objective Optimization**: Pareto dominance and fitness postprocessing
- **Process Safety**: Configurable timeouts and crash handling
- **Structured Logging**: Comprehensive debugging and monitoring support

## Architecture

### Module Dependencies

![Module Dependencies](doc/diagrams/module-dependencies.svg)

The library is organized into logical layers:
- **Core Data**: Genotype storage and morphology definitions
- **Network Components**: Cortex, sensors, neurons, actuators
- **Construction**: Phenotype construction from genotypes
- **Evolution**: Mutation operators, crossover, selection
- **Population**: Population management and species identification
- **Utilities**: Logging, signal processing, perturbation

### Process Supervision

![Supervision Tree](doc/diagrams/supervision-tree.svg)

The process hierarchy uses `spawn_link` for crash propagation:
- Population monitor (gen_server) manages multiple agent evaluations
- Each agent has an exoself process coordinating its network
- Network components (cortex, sensors, neurons, actuators) are linked
- Timeouts prevent infinite hangs (30s for cortex, 10s for neurons)

### Evaluation Cycle

![Evaluation Cycle Sequence](doc/diagrams/evaluation-cycle-sequence.svg)

The sense-think-act cycle:
1. Cortex triggers all sensors synchronously
2. Sensors read from environment (scape)
3. Sensors forward signals to neurons
4. Neurons aggregate inputs, apply activation functions
5. Neurons forward outputs through the network
6. Actuators collect neuron outputs
7. Actuators interact with environment and compute fitness
8. Cortex aggregates fitness and reports to exoself

### Mutation Process

![Mutation Sequence](doc/diagrams/mutation-sequence.svg)

Evolution through genetic operators:
1. Selection algorithm chooses survivors
2. Offspring cloned from parents
3. Mutation operators selected via roulette wheel
4. Topological (add_neuron, add_link) or parametric (mutate_weights) mutations applied
5. Genotype updated in database
6. Process logged for debugging

## Quick Start

Add to your `rebar.config`:

```erlang
{deps, [
    {macula_tweann, "~> 0.8.0"}
]}.
```

Or for Mix projects, add to `mix.exs`:

```elixir
def deps do
  [
    {:macula_tweann, "~> 0.8.0"}
  ]
end
```

### Basic Usage

```erlang
%% Initialize database
genotype:init_db().

%% Create a simple XOR agent
Constraint = #constraint{morphology = xor_mimic},
{ok, AgentId} = genotype:construct_agent(Constraint).

%% Construct phenotype (network processes)
Phenotype = constructor:construct_phenotype(AgentId).

%% Evaluate
cortex:sync(Phenotype#phenotype.cortex_pid).

%% Evolve
genome_mutator:mutate(AgentId).
```

## Documentation

- **[API Reference](https://hexdocs.pm/macula_tweann/)** - Complete module documentation
- **[Design Docs](design_docs/)** - Architecture and implementation details
- **[Release Notes](design_docs/releases/CHANGELOG.md)** - Version history

## Testing

```bash
rebar3 eunit          # Unit tests
rebar3 dialyzer       # Static analysis
```

## Roadmap

- ✅ **v0.1.0-0.6.1**: Core implementation, evolution engine, population management
- ✅ **v0.7.0**: Logging and error handling (Hardened Core)
- ✅ **v0.8.0**: Timeouts and process safety
- 🚧 **v0.8.1**: Architecture diagrams and documentation (current)
- 📋 **v0.9.0**: Performance profiling and optimization
- 📋 **v1.0.0**: Production-ready release

See [design_docs/releases/](design_docs/releases/) for detailed roadmaps.

## Contributing

Issues and pull requests welcome! Please follow existing code style and include tests.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Credits

Based on DXNN2 by Gene Sher. Adapted and extended by Macula.io.

## References

- Sher, Gene I. "Handbook of Neuroevolution Through Erlang" (2013)
- Stanley, Kenneth O., and Risto Miikkulainen. "Evolving neural networks through augmenting topologies." Evolutionary Computation (2002)
