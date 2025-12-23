# Installation

## Community Edition (Hex.pm)

The Community Edition is available on hex.pm and uses pure Erlang fallbacks for all operations. No Rust toolchain required.

### Rebar3 (Erlang Projects)

Add `macula_tweann` to your `rebar.config`:

```erlang
{deps, [
    {macula_tweann, "~> 0.16.0"}
]}.
```

Then fetch dependencies:

```bash
rebar3 compile
```

### Mix (Elixir Projects)

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:macula_tweann, "~> 0.16.0"}
  ]
end
```

Then fetch dependencies:

```bash
mix deps.get
```

## Enterprise Edition (10-15x Faster)

The Enterprise Edition includes high-performance Rust NIFs for compute-intensive operations. Add the private `macula_nn_nifs` package alongside macula-tweann:

```erlang
{deps, [
    {macula_tweann, "~> 0.16.0"},
    {macula_nn_nifs, {git, "git@github.com:macula-io/macula-nn-nifs.git", {tag, "v0.1.0"}}}
]}.
```

### Enterprise Requirements

- Rust 1.70+ and Cargo installed
- SSH access to the private macula-nn-nifs repository

The NIFs are automatically detected and used. No code changes required.

See the [Enterprise NIF Acceleration](enterprise-nifs.md) guide for details.

## From Source

Clone the repository:

```bash
git clone https://github.com/macula-io/macula-tweann.git
cd macula-tweann
rebar3 compile
```

## Requirements

- Erlang/OTP 24 or later
- Mnesia (included with Erlang)
- Rust 1.70+ (Enterprise Edition only)

## Verify Installation

Start an Erlang shell and initialize the database:

```erlang
rebar3 shell

% In the shell:
genotype:init_db().
% => ok
```

If you see `ok`, the installation is successful!

### Verify NIF Acceleration (Enterprise)

```erlang
% Check if enterprise NIFs are loaded
macula_nn_nifs:is_loaded().
% => true (Enterprise) or error (Community)

% Check tweann_nif detection
tweann_nif:is_loaded().
% => true (NIFs available)
```

## Next Steps

See the [Quick Start](quickstart.md) guide to create your first evolving neural network.
