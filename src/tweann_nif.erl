%% @doc Native Implemented Functions for high-performance network evaluation.
%%
%% This module provides Rust-accelerated network evaluation for TWEANN.
%% The NIF is loaded on application start and provides ~50-100x speedup
%% for forward propagation compared to the process-based approach.
%%
%% == Usage ==
%%
%% 1. Compile a genotype to a network reference:
%%    Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices)
%%
%% 2. Evaluate the network:
%%    Outputs = tweann_nif:evaluate(Network, Inputs)
%%
%% 3. For batch evaluation (many inputs, same network):
%%    OutputsList = tweann_nif:evaluate_batch(Network, InputsList)
%%
%% == Network Compilation Format ==
%%
%% Nodes are provided as a list of tuples:
%%   [{Index, Type, Activation, Bias, [{FromIndex, Weight}, ...]}, ...]
%%
%% Where:
%% - Index: integer node index (0-based)
%% - Type: atom (input | hidden | output | bias)
%% - Activation: atom (tanh | sigmoid | relu | etc.)
%% - Bias: float
%% - Connections: list of {FromIndex, Weight} tuples
%%
%% Nodes MUST be in topological order (inputs first, then hidden, then outputs).
%%
%% @copyright 2025 Macula.io
%% @license Apache-2.0
-module(tweann_nif).

-export([
    compile_network/3,
    evaluate/2,
    evaluate_batch/2,
    compatibility_distance/5,
    benchmark_evaluate/3,
    is_loaded/0,
    %% LTC/CfC functions
    evaluate_cfc/4,
    evaluate_cfc_with_weights/6,
    evaluate_ode/5,
    evaluate_ode_with_weights/7,
    evaluate_cfc_batch/4
]).

-on_load(init/0).

-define(APPNAME, macula_tweann).
-define(LIBNAME, tweann_nif).

%% @private
%% @doc Initialize and load the NIF library.
init() ->
    SoName = case code:priv_dir(?APPNAME) of
        {error, bad_name} ->
            case filelib:is_dir(filename:join(["..", priv])) of
                true ->
                    filename:join(["..", priv, ?LIBNAME]);
                _ ->
                    filename:join([priv, ?LIBNAME])
            end;
        Dir ->
            filename:join(Dir, ?LIBNAME)
    end,
    erlang:load_nif(SoName, 0).

%% @doc Check if the NIF is loaded.
%%
%% Returns true if the Rust NIF is available, false otherwise.
%% When false, the pure Erlang fallback should be used.
-spec is_loaded() -> boolean().
is_loaded() ->
    try
        %% Try to call a NIF function - if it returns nif_error, NIF not loaded
        _ = compile_network([], 0, []),
        true
    catch
        error:nif_not_loaded -> false;
        _:_ -> true  %% Other errors mean NIF is loaded but inputs were bad
    end.

%% @doc Compile a network for fast evaluation.
%%
%% Takes a list of nodes in topological order and returns an opaque
%% network reference that can be used with evaluate/2.
%%
%% Node format: {Index, Type, Activation, Bias, Connections}
%% - Index: 0-based node index
%% - Type: input | hidden | output | bias
%% - Activation: tanh | sigmoid | relu | linear | etc.
%% - Bias: float bias value
%% - Connections: [{FromIndex, Weight}, ...]
%%
%% @param Nodes List of node tuples in topological order
%% @param InputCount Number of input nodes
%% @param OutputIndices List of output node indices
%% @returns Opaque network reference
-spec compile_network(
    Nodes :: [{non_neg_integer(), atom(), atom(), float(), [{non_neg_integer(), float()}]}],
    InputCount :: non_neg_integer(),
    OutputIndices :: [non_neg_integer()]
) -> reference().
compile_network(_Nodes, _InputCount, _OutputIndices) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Evaluate a compiled network with given inputs.
%%
%% Performs forward propagation through the network and returns
%% the output values.
%%
%% @param Network Compiled network reference from compile_network/3
%% @param Inputs List of input values (must match InputCount)
%% @returns List of output values
-spec evaluate(Network :: reference(), Inputs :: [float()]) -> [float()].
evaluate(_Network, _Inputs) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Evaluate a network with multiple input sets.
%%
%% More efficient than calling evaluate/2 multiple times when
%% evaluating the same network with different inputs.
%%
%% @param Network Compiled network reference
%% @param InputsList List of input lists
%% @returns List of output lists (one per input set)
-spec evaluate_batch(Network :: reference(), InputsList :: [[float()]]) -> [[float()]].
evaluate_batch(_Network, _InputsList) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Calculate compatibility distance between two genomes.
%%
%% Used for NEAT speciation. Measures how different two genomes are
%% based on their connection genes.
%%
%% Formula: (c1 * E / N) + (c2 * D / N) + (c3 * W)
%% where:
%% - E = excess genes (beyond the other genome's max innovation)
%% - D = disjoint genes (within range but not matching)
%% - W = average weight difference of matching genes
%% - N = max genome size (for normalization)
%%
%% @param ConnectionsA Connections from genome A: [{Innovation, Weight}, ...]
%% @param ConnectionsB Connections from genome B: [{Innovation, Weight}, ...]
%% @param C1 Coefficient for excess genes
%% @param C2 Coefficient for disjoint genes
%% @param C3 Coefficient for weight differences
%% @returns Compatibility distance (lower = more similar)
-spec compatibility_distance(
    ConnectionsA :: [{non_neg_integer(), float()}],
    ConnectionsB :: [{non_neg_integer(), float()}],
    C1 :: float(),
    C2 :: float(),
    C3 :: float()
) -> float().
compatibility_distance(_ConnectionsA, _ConnectionsB, _C1, _C2, _C3) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Benchmark network evaluation.
%%
%% Evaluates the network N times and returns the average time
%% in microseconds per evaluation. Useful for performance testing.
%%
%% @param Network Compiled network reference
%% @param Inputs Input values to evaluate
%% @param Iterations Number of evaluations to perform
%% @returns Average time per evaluation in microseconds
-spec benchmark_evaluate(
    Network :: reference(),
    Inputs :: [float()],
    Iterations :: pos_integer()
) -> float().
benchmark_evaluate(_Network, _Inputs, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% LTC/CfC (Liquid Time-Constant) NIF Functions
%% ============================================================================

%% @doc CfC (Closed-form Continuous-time) evaluation.
%%
%% Fast closed-form approximation of LTC dynamics (~100x faster than ODE).
%% Implements: x' = sigmoid(-f) * x + (1 - sigmoid(-f)) * h
%%
%% @param Input Current input value
%% @param State Current internal state
%% @param Tau Base time constant
%% @param Bound State bound (clamping range)
%% @returns {NewState, Output}
-spec evaluate_cfc(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float()
) -> {float(), float()}.
evaluate_cfc(_Input, _State, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).

%% @doc CfC evaluation with custom backbone and head weights.
%%
%% @param Input Current input value
%% @param State Current internal state
%% @param Tau Base time constant
%% @param Bound State bound
%% @param BackboneWeights Weights for f() backbone network
%% @param HeadWeights Weights for h() head network
%% @returns {NewState, Output}
-spec evaluate_cfc_with_weights(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    BackboneWeights :: [float()],
    HeadWeights :: [float()]
) -> {float(), float()}.
evaluate_cfc_with_weights(_Input, _State, _Tau, _Bound, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

%% @doc ODE-based LTC evaluation.
%%
%% Accurate but slower evaluation using Euler integration.
%% Implements: dx/dt = -[1/tau + f] * x + f * A
%%
%% @param Input Current input value
%% @param State Current internal state
%% @param Tau Base time constant
%% @param Bound State bound
%% @param Dt Integration time step
%% @returns {NewState, Output}
-spec evaluate_ode(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    Dt :: float()
) -> {float(), float()}.
evaluate_ode(_Input, _State, _Tau, _Bound, _Dt) ->
    erlang:nif_error(nif_not_loaded).

%% @doc ODE evaluation with custom weights.
%%
%% @param Input Current input value
%% @param State Current internal state
%% @param Tau Base time constant
%% @param Bound State bound
%% @param Dt Integration time step
%% @param BackboneWeights Weights for f() backbone
%% @param HeadWeights Weights for h() head
%% @returns {NewState, Output}
-spec evaluate_ode_with_weights(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    Dt :: float(),
    BackboneWeights :: [float()],
    HeadWeights :: [float()]
) -> {float(), float()}.
evaluate_ode_with_weights(_Input, _State, _Tau, _Bound, _Dt, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch CfC evaluation for time series.
%%
%% Evaluates a sequence of inputs, maintaining state between steps.
%% Efficient for processing time series data.
%%
%% @param Inputs List of input values (time series)
%% @param InitialState Starting internal state
%% @param Tau Base time constant
%% @param Bound State bound
%% @returns List of {State, Output} tuples
-spec evaluate_cfc_batch(
    Inputs :: [float()],
    InitialState :: float(),
    Tau :: float(),
    Bound :: float()
) -> [{float(), float()}].
evaluate_cfc_batch(_Inputs, _InitialState, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).
