# macula-tweann

**Topology and Weight Evolving Artificial Neural Networks for Erlang**

[![Hex.pm](https://img.shields.io/hexpm/v/macula_tweann.svg)](https://hex.pm/packages/macula_tweann)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/macula_tweann/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/macula-io/macula-tweann/blob/main/LICENSE)

Evolutionary neural networks that evolve both topology and weights, now with **Liquid Time-Constant (LTC) neurons** for adaptive temporal processing. Based on DXNN2 by Gene Sher.

## Highlights

- **First TWEANN library with LTC neurons** in Erlang/OTP
- **CfC closed-form approximation** - ~100x faster than ODE-based LTC
- **Hybrid networks** - Mix standard and LTC neurons in the same network
- **Production ready** - Comprehensive logging, error handling, and process safety

## Quick Start

```erlang
%% Add to rebar.config
{deps, [{macula_tweann, "~> 0.10.0"}]}.

%% Create and evolve a standard network
genotype:init_db(),
Constraint = #constraint{morphology = xor_mimic},
{ok, AgentId} = genotype:construct_agent(Constraint),
genome_mutator:mutate(AgentId).

%% Use LTC dynamics directly
{NewState, Output} = ltc_dynamics:evaluate_cfc(Input, State, Tau, Bound).
```

## LTC Neurons

Liquid Time-Constant neurons enable **adaptive temporal processing** with input-dependent time constants:

![LTC Architecture](doc/diagrams/ltc-neuron-architecture.svg)

```erlang
%% CfC evaluation (fast, closed-form)
{State1, _} = ltc_dynamics:evaluate_cfc(1.0, 0.0, 1.0, 1.0),
{State2, _} = ltc_dynamics:evaluate_cfc(1.0, State1, 1.0, 1.0).
%% State persists between evaluations - temporal memory!
```

Key equations:
- **LTC ODE**: `dx/dt = -[1/τ + f(x,I,θ)]·x + f(x,I,θ)·A`
- **CfC**: `x(t+Δt) = σ(-f)·x(t) + (1-σ(-f))·h` (100x faster)

See the [LTC Neurons Guide](https://hexdocs.pm/macula_tweann/ltc-neurons.html) for details.

## Documentation

- **[Installation](https://hexdocs.pm/macula_tweann/installation.html)** - Add to your project
- **[Quick Start](https://hexdocs.pm/macula_tweann/quickstart.html)** - Basic usage
- **[LTC Neurons](https://hexdocs.pm/macula_tweann/ltc-neurons.html)** - Temporal dynamics
- **[LTC Usage Guide](https://hexdocs.pm/macula_tweann/ltc-usage-guide.html)** - Practical examples
- **[Architecture](https://hexdocs.pm/macula_tweann/architecture.html)** - System design
- **[API Reference](https://hexdocs.pm/macula_tweann/api-reference.html)** - Module documentation

## Features

### Neural Network Evolution
- **Topology Evolution**: Networks add/remove neurons and connections
- **Weight Evolution**: Synaptic weights optimized through selection
- **Speciation**: Behavioral diversity preservation (NEAT-style)
- **Multi-objective**: Pareto dominance optimization

### LTC/CfC Neurons (NEW in 0.10.0)
- **Temporal Memory**: Neurons maintain persistent internal state
- **Adaptive Dynamics**: Input-dependent time constants
- **CfC Mode**: ~100x faster than ODE-based evaluation
- **Hybrid Networks**: Mix standard and LTC neurons

### Production Quality
- **Process Safety**: Timeouts and crash handling
- **Comprehensive Logging**: Structured logging throughout
- **Rust NIF (optional)**: High-performance network evaluation
- **Mnesia Storage**: Persistent genotype storage

## Architecture

![Module Dependencies](doc/diagrams/module-dependencies.svg)

Process-based neural networks with evolutionary operators. See [Architecture Guide](https://hexdocs.pm/macula_tweann/architecture.html) for details.

## Testing

```bash
rebar3 eunit          # Unit tests (513 tests)
rebar3 dialyzer       # Static analysis
rebar3 ex_doc         # Generate documentation
```

## Academic References

### TWEANN/NEAT
- Sher, G.I. *Handbook of Neuroevolution Through Erlang* (2013)
- Stanley, K.O., Miikkulainen, R. *Evolving Neural Networks Through Augmenting Topologies* (2002)

### LTC/CfC
- Hasani, R., Lechner, M., et al. *Liquid Time-constant Networks* (AAAI 2021)
- Hasani, R., Lechner, M., et al. *Closed-form Continuous-time Neural Networks* (Nature Machine Intelligence, 2022)

## License

Apache License 2.0 - See [LICENSE](https://github.com/macula-io/macula-tweann/blob/main/LICENSE)

## Credits

Based on DXNN2 by Gene Sher. Adapted with LTC extensions by [Macula.io](https://macula.io).
