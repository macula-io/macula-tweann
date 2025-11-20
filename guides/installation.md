# Installation

## Hex.pm (Recommended)

Add `macula_tweann` to your `rebar.config`:

```erlang
{deps, [
    {macula_tweann, "~> 0.8.6"}
]}.
```

Then fetch dependencies:

```bash
rebar3 compile
```

## Mix (Elixir Projects)

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:macula_tweann, "~> 0.8.6"}
  ]
end
```

Then fetch dependencies:

```bash
mix deps.get
```

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

## Verify Installation

Start an Erlang shell and initialize the database:

```erlang
rebar3 shell

% In the shell:
genotype:init_db().
% => ok
```

If you see `ok`, the installation is successful!

## Next Steps

See the [Quick Start](quickstart.html) guide to create your first evolving neural network.
