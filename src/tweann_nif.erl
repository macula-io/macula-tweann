%% @doc Native Implemented Functions for high-performance network evaluation.
%%
%% This module provides Rust-accelerated network evaluation for TWEANN.
%% The NIF is loaded on application start and provides ~50-100x speedup
%% for forward propagation compared to the process-based approach.
%%
%% All functions have pure Erlang fallbacks in tweann_nif_fallback.erl
%% that are used automatically when the NIF is not available.
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
-module(tweann_nif).

-export([
    compile_network/3,
    evaluate/2,
    evaluate_batch/2,
    compatibility_distance/5,
    benchmark_evaluate/3,
    is_loaded/0,
    %% Signal aggregation NIFs
    dot_product_flat/3,
    dot_product_batch/1,
    dot_product_preflattened/3,
    flatten_weights/1,
    %% LTC/CfC functions
    evaluate_cfc/4,
    evaluate_cfc_with_weights/6,
    evaluate_ode/5,
    evaluate_ode_with_weights/7,
    evaluate_cfc_batch/4,
    %% Distance and KNN (Novelty Search)
    euclidean_distance/2,
    euclidean_distance_batch/2,
    knn_novelty/4,
    knn_novelty_batch/3,
    %% Statistics
    fitness_stats/1,
    weighted_moving_average/2,
    shannon_entropy/1,
    histogram/4,
    %% Selection
    build_cumulative_fitness/1,
    roulette_select/3,
    roulette_select_batch/3,
    tournament_select/2,
    %% Reward and Meta-Controller
    z_score/3,
    compute_reward_component/2,
    compute_weighted_reward/1
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
%% When false, the pure Erlang fallback is used automatically.
-spec is_loaded() -> boolean().
is_loaded() ->
    try
        _ = nif_compile_network([], 0, []),
        true
    catch
        error:nif_not_loaded -> false;
        _:_ -> true
    end.

%%==============================================================================
%% Network Evaluation Functions
%%==============================================================================

%% @doc Compile a network for fast evaluation.
%%
%% Takes a list of nodes in topological order and returns an opaque
%% network reference that can be used with evaluate/2.
%%
%% Falls back to pure Erlang if NIF not available.
-spec compile_network(
    Nodes :: [{non_neg_integer(), atom(), atom(), float(), [{non_neg_integer(), float()}]}],
    InputCount :: non_neg_integer(),
    OutputIndices :: [non_neg_integer()]
) -> reference() | map().
compile_network(Nodes, InputCount, OutputIndices) ->
    try nif_compile_network(Nodes, InputCount, OutputIndices)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:compile_network(Nodes, InputCount, OutputIndices)
    end.

nif_compile_network(_Nodes, _InputCount, _OutputIndices) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Evaluate a compiled network with given inputs.
%%
%% Performs forward propagation through the network and returns
%% the output values. Falls back to pure Erlang if NIF not available.
-spec evaluate(Network :: reference() | map(), Inputs :: [float()]) -> [float()].
evaluate(Network, Inputs) ->
    try nif_evaluate(Network, Inputs)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate(Network, Inputs)
    end.

nif_evaluate(_Network, _Inputs) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Evaluate a network with multiple input sets.
%%
%% More efficient than calling evaluate/2 multiple times when
%% evaluating the same network with different inputs.
-spec evaluate_batch(Network :: reference() | map(), InputsList :: [[float()]]) -> [[float()]].
evaluate_batch(Network, InputsList) ->
    try nif_evaluate_batch(Network, InputsList)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_batch(Network, InputsList)
    end.

nif_evaluate_batch(_Network, _InputsList) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Calculate compatibility distance between two genomes.
%%
%% Used for NEAT speciation. Measures how different two genomes are
%% based on their connection genes.
-spec compatibility_distance(
    ConnectionsA :: [{non_neg_integer(), float()}],
    ConnectionsB :: [{non_neg_integer(), float()}],
    C1 :: float(),
    C2 :: float(),
    C3 :: float()
) -> float().
compatibility_distance(ConnectionsA, ConnectionsB, C1, C2, C3) ->
    try nif_compatibility_distance(ConnectionsA, ConnectionsB, C1, C2, C3)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:compatibility_distance(ConnectionsA, ConnectionsB, C1, C2, C3)
    end.

nif_compatibility_distance(_ConnectionsA, _ConnectionsB, _C1, _C2, _C3) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Benchmark network evaluation.
%%
%% Evaluates the network N times and returns the average time
%% in microseconds per evaluation.
-spec benchmark_evaluate(
    Network :: reference() | map(),
    Inputs :: [float()],
    Iterations :: pos_integer()
) -> float().
benchmark_evaluate(Network, Inputs, Iterations) ->
    try nif_benchmark_evaluate(Network, Inputs, Iterations)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:benchmark_evaluate(Network, Inputs, Iterations)
    end.

nif_benchmark_evaluate(_Network, _Inputs, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% Signal Aggregation Functions
%%==============================================================================

%% @doc Fast dot product for signal aggregation.
%%
%% Computes: sum(signals[i] * weights[i]) + bias
-spec dot_product_flat(
    Signals :: [float()],
    Weights :: [float()],
    Bias :: float()
) -> float().
dot_product_flat(Signals, Weights, Bias) ->
    try nif_dot_product_flat(Signals, Weights, Bias)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:dot_product_flat(Signals, Weights, Bias)
    end.

nif_dot_product_flat(_Signals, _Weights, _Bias) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch dot product for multiple neurons.
-spec dot_product_batch(Batch :: [{[float()], [float()], float()}]) -> [float()].
dot_product_batch(Batch) ->
    try nif_dot_product_batch(Batch)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:dot_product_batch(Batch)
    end.

nif_dot_product_batch(_Batch) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Dot product with pre-flattened data.
-spec dot_product_preflattened(Signals :: [float()], Weights :: [float()], Bias :: float()) -> float().
dot_product_preflattened(Signals, Weights, Bias) ->
    try nif_dot_product_preflattened(Signals, Weights, Bias)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:dot_product_preflattened(Signals, Weights, Bias)
    end.

nif_dot_product_preflattened(_Signals, _Weights, _Bias) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Flatten weights for efficient dot product.
-spec flatten_weights(WeightedInputs :: [{non_neg_integer(), [{float(), float(), float(), [float()]}]}]) ->
    {[float()], [non_neg_integer()]}.
flatten_weights(WeightedInputs) ->
    try nif_flatten_weights(WeightedInputs)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:flatten_weights(WeightedInputs)
    end.

nif_flatten_weights(_WeightedInputs) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% LTC/CfC (Liquid Time-Constant) Functions
%%==============================================================================

%% @doc CfC (Closed-form Continuous-time) evaluation.
%%
%% Fast closed-form approximation of LTC dynamics (~100x faster than ODE).
-spec evaluate_cfc(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float()
) -> {float(), float()}.
evaluate_cfc(Input, State, Tau, Bound) ->
    try nif_evaluate_cfc(Input, State, Tau, Bound)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_cfc(Input, State, Tau, Bound)
    end.

nif_evaluate_cfc(_Input, _State, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).

%% @doc CfC evaluation with custom backbone and head weights.
-spec evaluate_cfc_with_weights(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    BackboneWeights :: [float()],
    HeadWeights :: [float()]
) -> {float(), float()}.
evaluate_cfc_with_weights(Input, State, Tau, Bound, BackboneWeights, HeadWeights) ->
    try nif_evaluate_cfc_with_weights(Input, State, Tau, Bound, BackboneWeights, HeadWeights)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_cfc_with_weights(Input, State, Tau, Bound, BackboneWeights, HeadWeights)
    end.

nif_evaluate_cfc_with_weights(_Input, _State, _Tau, _Bound, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

%% @doc ODE-based LTC evaluation.
%%
%% Accurate but slower evaluation using Euler integration.
-spec evaluate_ode(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    Dt :: float()
) -> {float(), float()}.
evaluate_ode(Input, State, Tau, Bound, Dt) ->
    try nif_evaluate_ode(Input, State, Tau, Bound, Dt)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_ode(Input, State, Tau, Bound, Dt)
    end.

nif_evaluate_ode(_Input, _State, _Tau, _Bound, _Dt) ->
    erlang:nif_error(nif_not_loaded).

%% @doc ODE evaluation with custom weights.
-spec evaluate_ode_with_weights(
    Input :: float(),
    State :: float(),
    Tau :: float(),
    Bound :: float(),
    Dt :: float(),
    BackboneWeights :: [float()],
    HeadWeights :: [float()]
) -> {float(), float()}.
evaluate_ode_with_weights(Input, State, Tau, Bound, Dt, BackboneWeights, HeadWeights) ->
    try nif_evaluate_ode_with_weights(Input, State, Tau, Bound, Dt, BackboneWeights, HeadWeights)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_ode_with_weights(Input, State, Tau, Bound, Dt, BackboneWeights, HeadWeights)
    end.

nif_evaluate_ode_with_weights(_Input, _State, _Tau, _Bound, _Dt, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch CfC evaluation for time series.
-spec evaluate_cfc_batch(
    Inputs :: [float()],
    InitialState :: float(),
    Tau :: float(),
    Bound :: float()
) -> [{float(), float()}].
evaluate_cfc_batch(Inputs, InitialState, Tau, Bound) ->
    try nif_evaluate_cfc_batch(Inputs, InitialState, Tau, Bound)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:evaluate_cfc_batch(Inputs, InitialState, Tau, Bound)
    end.

nif_evaluate_cfc_batch(_Inputs, _InitialState, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% Distance and KNN Functions (Novelty Search)
%%==============================================================================

%% @doc Compute Euclidean distance between two behavior vectors.
-spec euclidean_distance(V1 :: [float()], V2 :: [float()]) -> float().
euclidean_distance(V1, V2) ->
    try nif_euclidean_distance(V1, V2)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:euclidean_distance(V1, V2)
    end.

nif_euclidean_distance(_V1, _V2) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch Euclidean distance from one vector to many.
-spec euclidean_distance_batch(Target :: [float()], Others :: [[float()]]) ->
    [{non_neg_integer(), float()}].
euclidean_distance_batch(Target, Others) ->
    try nif_euclidean_distance_batch(Target, Others)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:euclidean_distance_batch(Target, Others)
    end.

nif_euclidean_distance_batch(_Target, _Others) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute k-nearest neighbor novelty score.
-spec knn_novelty(
    Target :: [float()],
    Population :: [[float()]],
    Archive :: [[float()]],
    K :: pos_integer()
) -> float().
knn_novelty(Target, Population, Archive, K) ->
    try nif_knn_novelty(Target, Population, Archive, K)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:knn_novelty(Target, Population, Archive, K)
    end.

nif_knn_novelty(_Target, _Population, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch kNN novelty for multiple behaviors.
-spec knn_novelty_batch(
    Behaviors :: [[float()]],
    Archive :: [[float()]],
    K :: pos_integer()
) -> [float()].
knn_novelty_batch(Behaviors, Archive, K) ->
    try nif_knn_novelty_batch(Behaviors, Archive, K)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:knn_novelty_batch(Behaviors, Archive, K)
    end.

nif_knn_novelty_batch(_Behaviors, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% Statistics Functions
%%==============================================================================

%% @doc Compute fitness statistics in single pass.
%%
%% Returns {Min, Max, Mean, Variance, StdDev, Sum}.
-spec fitness_stats(Fitnesses :: [float()]) ->
    {float(), float(), float(), float(), float(), float()}.
fitness_stats(Fitnesses) ->
    try nif_fitness_stats(Fitnesses)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:fitness_stats(Fitnesses)
    end.

nif_fitness_stats(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute weighted moving average.
-spec weighted_moving_average(Values :: [float()], Decay :: float()) -> float().
weighted_moving_average(Values, Decay) ->
    try nif_weighted_moving_average(Values, Decay)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:weighted_moving_average(Values, Decay)
    end.

nif_weighted_moving_average(_Values, _Decay) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute Shannon entropy.
-spec shannon_entropy(Values :: [float()]) -> float().
shannon_entropy(Values) ->
    try nif_shannon_entropy(Values)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:shannon_entropy(Values)
    end.

nif_shannon_entropy(_Values) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Create histogram bins.
-spec histogram(
    Values :: [float()],
    NumBins :: pos_integer(),
    MinVal :: float(),
    MaxVal :: float()
) -> [non_neg_integer()].
histogram(Values, NumBins, MinVal, MaxVal) ->
    try nif_histogram(Values, NumBins, MinVal, MaxVal)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:histogram(Values, NumBins, MinVal, MaxVal)
    end.

nif_histogram(_Values, _NumBins, _MinVal, _MaxVal) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% Selection Functions
%%==============================================================================

%% @doc Build cumulative fitness array for roulette selection.
-spec build_cumulative_fitness(Fitnesses :: [float()]) -> {[float()], float()}.
build_cumulative_fitness(Fitnesses) ->
    try nif_build_cumulative_fitness(Fitnesses)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:build_cumulative_fitness(Fitnesses)
    end.

nif_build_cumulative_fitness(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Roulette wheel selection with binary search.
-spec roulette_select(Cumulative :: [float()], Total :: float(), RandomVal :: float()) ->
    non_neg_integer().
roulette_select(Cumulative, Total, RandomVal) ->
    try nif_roulette_select(Cumulative, Total, RandomVal)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:roulette_select(Cumulative, Total, RandomVal)
    end.

nif_roulette_select(_Cumulative, _Total, _RandomVal) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch roulette selection.
-spec roulette_select_batch(
    Cumulative :: [float()],
    Total :: float(),
    RandomVals :: [float()]
) -> [non_neg_integer()].
roulette_select_batch(Cumulative, Total, RandomVals) ->
    try nif_roulette_select_batch(Cumulative, Total, RandomVals)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:roulette_select_batch(Cumulative, Total, RandomVals)
    end.

nif_roulette_select_batch(_Cumulative, _Total, _RandomVals) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Tournament selection.
-spec tournament_select(Contestants :: [non_neg_integer()], Fitnesses :: [float()]) ->
    non_neg_integer().
tournament_select(Contestants, Fitnesses) ->
    try nif_tournament_select(Contestants, Fitnesses)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:tournament_select(Contestants, Fitnesses)
    end.

nif_tournament_select(_Contestants, _Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%%==============================================================================
%% Reward and Meta-Controller Functions
%%==============================================================================

%% @doc Compute z-score normalization.
-spec z_score(Value :: float(), Mean :: float(), StdDev :: float()) -> float().
z_score(Value, Mean, StdDev) ->
    try nif_z_score(Value, Mean, StdDev)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:z_score(Value, Mean, StdDev)
    end.

nif_z_score(_Value, _Mean, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute reward component with normalization.
%%
%% Returns {RawComponent, NormalizedComponent, ZScore}.
-spec compute_reward_component(History :: [float()], Current :: float()) ->
    {float(), float(), float()}.
compute_reward_component(History, Current) ->
    try nif_compute_reward_component(History, Current)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:compute_reward_component(History, Current)
    end.

nif_compute_reward_component(_History, _Current) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch compute weighted reward.
-spec compute_weighted_reward(Components :: [{[float()], float(), float()}]) -> float().
compute_weighted_reward(Components) ->
    try nif_compute_weighted_reward(Components)
    catch error:nif_not_loaded ->
        tweann_nif_fallback:compute_weighted_reward(Components)
    end.

nif_compute_weighted_reward(_Components) ->
    erlang:nif_error(nif_not_loaded).
