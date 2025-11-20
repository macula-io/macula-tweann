# macula-tweann

**Topology and Weight Evolving Artificial Neural Networks for Erlang**

[![Hex.pm](https://img.shields.io/hexpm/v/macula_tweann.svg)](https://hex.pm/packages/macula_tweann)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/macula_tweann/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Evolutionary neural networks that evolve both topology and weights. Based on DXNN2 by Gene Sher.

## Quick Start

```erlang
%% Add to rebar.config
{deps, [{macula_tweann, "~> 0.8.3"}]}.

%% Create and evolve
genotype:init_db(),
Constraint = #constraint{morphology = xor_mimic},
{ok, AgentId} = genotype:construct_agent(Constraint),
genome_mutator:mutate(AgentId).
```

## Documentation

- **[Installation](guides/installation.html)** - Add to your project
- **[Quick Start](guides/quickstart.html)** - Basic usage
- **[Architecture](guides/architecture.html)** - System design
- **[API Reference](api-reference.html)** - Module documentation

## Features

- **Topology Evolution**: Networks add/remove neurons and connections
- **Weight Evolution**: Synaptic weights optimized through selection
- **Speciation**: Behavioral diversity preservation
- **Multi-objective**: Pareto dominance optimization
- **Process Safety**: Timeouts and crash handling
- **Production Ready**: Comprehensive logging and error handling

## Architecture

![Module Dependencies](doc/diagrams/module-dependencies.svg)

Process-based neural networks with evolutionary operators. See [Architecture Guide](guides/architecture.html) for details.

## Testing

```bash
rebar3 eunit          # Unit tests
rebar3 dialyzer       # Static analysis
```

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Credits

Based on DXNN2 by Gene Sher. Adapted by [Macula.io](https://macula.io).

### References

- Sher, G.I. *Handbook of Neuroevolution Through Erlang* (2013)
- Stanley, K.O., Miikkulainen, R. *Evolving Neural Networks Through Augmenting Topologies* (2002)
