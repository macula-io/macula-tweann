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
%% Signal Aggregation NIF Functions
%% ============================================================================

%% @doc Fast dot product for signal aggregation.
%%
%% Computes: sum(signals[i] * weights[i]) + bias
%%
%% This is a hot path function for neural network forward propagation.
%% Expects pre-flattened data - use signal_aggregator:flatten_for_nif/2
%% to convert from the standard tuple format.
%%
%% @param Signals Flattened list of signal values
%% @param Weights Flattened list of weight values (same length as Signals)
%% @param Bias Bias value to add to result
%% @returns Aggregated scalar value
-spec dot_product_flat(
    Signals :: [float()],
    Weights :: [float()],
    Bias :: float()
) -> float().
dot_product_flat(_Signals, _Weights, _Bias) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch dot product for multiple neurons.
%%
%% More efficient than calling dot_product_flat/3 multiple times.
%% Processes multiple neurons in one NIF call to amortize overhead.
%% Uses dirty scheduler for large batches to avoid blocking.
%%
%% @param Batch List of {Signals, Weights, Bias} tuples
%% @returns List of dot product results
-spec dot_product_batch(
    Batch :: [{[float()], [float()], float()}]
) -> [float()].
dot_product_batch(_Batch) ->
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

%% ============================================================================
%% Weight Flattening NIF Functions
%% ============================================================================

%% @doc Flatten weights for efficient dot product.
%%
%% Converts nested weight structure to flat arrays.
%% Returns {FlatWeights, CountsPerSource}.
-spec flatten_weights(WeightedInputs :: [{non_neg_integer(), [{float(), float(), float(), [float()]}]}]) ->
    {[float()], [non_neg_integer()]}.
flatten_weights(_WeightedInputs) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Dot product with pre-flattened data.
-spec dot_product_preflattened(Signals :: [float()], Weights :: [float()], Bias :: float()) -> float().
dot_product_preflattened(_Signals, _Weights, _Bias) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Distance and KNN NIF Functions (Novelty Search)
%% ============================================================================

%% @doc Compute Euclidean distance between two behavior vectors.
%%
%% Hot path function for novelty search.
%% @param V1 First behavior vector
%% @param V2 Second behavior vector
%% @returns Euclidean distance
-spec euclidean_distance(V1 :: [float()], V2 :: [float()]) -> float().
euclidean_distance(_V1, _V2) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch Euclidean distance from one vector to many.
%%
%% Returns list of {Index, Distance} sorted by distance ascending.
%% @param Target Target behavior vector
%% @param Others List of other behavior vectors
%% @returns Sorted list of {Index, Distance}
-spec euclidean_distance_batch(Target :: [float()], Others :: [[float()]]) ->
    [{non_neg_integer(), float()}].
euclidean_distance_batch(_Target, _Others) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute k-nearest neighbor novelty score.
%%
%% Returns average distance to k nearest neighbors in population + archive.
%% @param Target Target behavior vector
%% @param Population Current population behaviors
%% @param Archive Historical behavior archive
%% @param K Number of nearest neighbors
%% @returns Average distance to k nearest
-spec knn_novelty(
    Target :: [float()],
    Population :: [[float()]],
    Archive :: [[float()]],
    K :: pos_integer()
) -> float().
knn_novelty(_Target, _Population, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch kNN novelty for multiple behaviors.
%%
%% More efficient than calling knn_novelty repeatedly.
%% @param Behaviors List of behavior vectors
%% @param Archive Historical behavior archive
%% @param K Number of nearest neighbors
%% @returns List of novelty scores
-spec knn_novelty_batch(
    Behaviors :: [[float()]],
    Archive :: [[float()]],
    K :: pos_integer()
) -> [float()].
knn_novelty_batch(_Behaviors, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Statistics NIF Functions
%% ============================================================================

%% @doc Compute fitness statistics in single pass.
%%
%% Returns {Min, Max, Mean, Variance, StdDev, Sum}.
-spec fitness_stats(Fitnesses :: [float()]) ->
    {float(), float(), float(), float(), float(), float()}.
fitness_stats(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute weighted moving average.
%%
%% Uses exponential decay weights.
%% @param Values List of values (most recent first)
%% @param Decay Decay factor (0-1)
%% @returns Weighted average
-spec weighted_moving_average(Values :: [float()], Decay :: float()) -> float().
weighted_moving_average(_Values, _Decay) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute Shannon entropy.
%%
%% Values are normalized to a probability distribution.
-spec shannon_entropy(Values :: [float()]) -> float().
shannon_entropy(_Values) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Create histogram bins.
%%
%% @param Values Values to bin
%% @param NumBins Number of bins
%% @param MinVal Minimum value
%% @param MaxVal Maximum value
%% @returns List of bin counts
-spec histogram(
    Values :: [float()],
    NumBins :: pos_integer(),
    MinVal :: float(),
    MaxVal :: float()
) -> [non_neg_integer()].
histogram(_Values, _NumBins, _MinVal, _MaxVal) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Selection NIF Functions
%% ============================================================================

%% @doc Build cumulative fitness array for roulette selection.
%%
%% Shifts fitnesses to ensure all positive.
%% Returns {CumulativeFitnesses, TotalFitness}.
-spec build_cumulative_fitness(Fitnesses :: [float()]) -> {[float()], float()}.
build_cumulative_fitness(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Roulette wheel selection with binary search.
%%
%% O(log n) selection using pre-built cumulative array.
%% @param Cumulative Cumulative fitness array from build_cumulative_fitness/1
%% @param Total Total fitness
%% @param RandomVal Random value in [0, 1]
%% @returns Selected index
-spec roulette_select(Cumulative :: [float()], Total :: float(), RandomVal :: float()) ->
    non_neg_integer().
roulette_select(_Cumulative, _Total, _RandomVal) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch roulette selection.
%%
%% Select multiple individuals efficiently.
%% @param Cumulative Cumulative fitness array
%% @param Total Total fitness
%% @param RandomVals List of random values in [0, 1]
%% @returns List of selected indices
-spec roulette_select_batch(
    Cumulative :: [float()],
    Total :: float(),
    RandomVals :: [float()]
) -> [non_neg_integer()].
roulette_select_batch(_Cumulative, _Total, _RandomVals) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Tournament selection.
%%
%% Select best from random subset.
%% @param Contestants List of contestant indices
%% @param Fitnesses All fitness values
%% @returns Index of winner
-spec tournament_select(Contestants :: [non_neg_integer()], Fitnesses :: [float()]) ->
    non_neg_integer().
tournament_select(_Contestants, _Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Reward and Meta-Controller NIF Functions
%% ============================================================================

%% @doc Compute z-score normalization.
%%
%% Returns 0 if std_dev is too small.
-spec z_score(Value :: float(), Mean :: float(), StdDev :: float()) -> float().
z_score(_Value, _Mean, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Compute reward component with normalization.
%%
%% Returns {RawComponent, NormalizedComponent, ZScore}.
%% @param History Recent values for baseline
%% @param Current Current value
-spec compute_reward_component(History :: [float()], Current :: float()) ->
    {float(), float(), float()}.
compute_reward_component(_History, _Current) ->
    erlang:nif_error(nif_not_loaded).

%% @doc Batch compute weighted reward.
%%
%% Each component is {History, CurrentValue, Weight}.
%% Returns weighted sum of normalized components.
-spec compute_weighted_reward(Components :: [{[float()], float(), float()}]) -> float().
compute_weighted_reward(_Components) ->
    erlang:nif_error(nif_not_loaded).
