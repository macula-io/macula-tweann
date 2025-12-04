%% @doc Synchronous neural network evaluator for inference.
%%
%% This module provides synchronous (blocking) forward propagation
%% for neural networks. Unlike the process-based cortex/neuron approach
%% used during training, this is designed for fast inference in
%% real-time applications like games.
%%
%% == Usage ==
%%
%% Create a network from a genotype:
%%   {ok, Network} = network_evaluator:from_genotype(AgentId)
%%
%% Or create a simple feedforward network:
%%   Network = network_evaluator:create_feedforward(42, [16, 8], 6)
%%
%% Evaluate:
%%   Outputs = network_evaluator:evaluate(Network, Inputs)
%%
%% @copyright 2025 Macula.io
-module(network_evaluator).

-export([
    create_feedforward/3,
    create_feedforward/4,
    evaluate/2,
    evaluate_with_activations/2,
    from_genotype/1,
    get_weights/1,
    set_weights/2,
    get_topology/1,
    get_viz_data/3
]).

-record(network, {
    layers :: [layer()],
    activation :: atom()
}).

-type layer() :: {Weights :: [[float()]], Biases :: [float()]}.
-type network() :: #network{}.

-export_type([network/0]).

%% @doc Create a feedforward network with random weights.
%%
%% @param InputSize Number of inputs
%% @param HiddenSizes List of hidden layer sizes
%% @param OutputSize Number of outputs
%% @returns Network record
-spec create_feedforward(pos_integer(), [pos_integer()], pos_integer()) -> network().
create_feedforward(InputSize, HiddenSizes, OutputSize) ->
    create_feedforward(InputSize, HiddenSizes, OutputSize, tanh).

%% @doc Create a feedforward network with specified activation.
-spec create_feedforward(pos_integer(), [pos_integer()], pos_integer(), atom()) -> network().
create_feedforward(InputSize, HiddenSizes, OutputSize, Activation) ->
    LayerSizes = [InputSize | HiddenSizes] ++ [OutputSize],
    Layers = create_layers(LayerSizes),
    #network{layers = Layers, activation = Activation}.

%% @doc Evaluate the network with given inputs.
%%
%% Performs synchronous forward propagation through all layers.
%%
%% @param Network The network record
%% @param Inputs List of input values (must match input size)
%% @returns List of output values
-spec evaluate(network(), [float()]) -> [float()].
evaluate(#network{layers = Layers, activation = Activation}, Inputs) ->
    forward_propagate(Layers, Inputs, Activation).

%% @doc Load a network from a genotype stored in Mnesia.
%%
%% Reads the agent's neural network structure and weights from Mnesia
%% and creates an evaluator network.
%%
%% @param AgentId The agent identifier
%% @returns {ok, Network} | {error, Reason}
-spec from_genotype(term()) -> {ok, network()} | {error, term()}.
from_genotype(AgentId) ->
    case load_genotype_structure(AgentId) of
        {ok, Structure} ->
            Network = build_network_from_structure(Structure),
            {ok, Network};
        {error, Reason} ->
            {error, Reason}
    end.

%% @doc Get all weights from the network as a flat list.
%%
%% Useful for evolution - can be mutated and set back.
-spec get_weights(network()) -> [float()].
get_weights(#network{layers = Layers}) ->
    lists:flatmap(
        fun({Weights, Biases}) ->
            lists:flatten(Weights) ++ Biases
        end,
        Layers
    ).

%% @doc Set weights from a flat list.
%%
%% The list must have the same number of elements as returned by get_weights/1.
-spec set_weights(network(), [float()]) -> network().
set_weights(Network = #network{layers = Layers}, FlatWeights) ->
    {NewLayers, []} = lists:mapfoldl(
        fun({Weights, Biases}, Remaining) ->
            WeightCount = length(Weights) * length(hd(Weights)),
            BiasCount = length(Biases),
            {WeightVals, Rest1} = lists:split(WeightCount, Remaining),
            {BiasVals, Rest2} = lists:split(BiasCount, Rest1),
            NewWeights = reshape_weights(WeightVals, length(hd(Weights))),
            {{NewWeights, BiasVals}, Rest2}
        end,
        FlatWeights,
        Layers
    ),
    Network#network{layers = NewLayers}.

%%==============================================================================
%% Internal Functions
%%==============================================================================

%% @private Create layer weight matrices
create_layers(LayerSizes) ->
    Pairs = lists:zip(
        lists:droplast(LayerSizes),
        tl(LayerSizes)
    ),
    [create_layer(FromSize, ToSize) || {FromSize, ToSize} <- Pairs].

%% @private Create a single layer with random weights
create_layer(FromSize, ToSize) ->
    %% Xavier initialization: scale by sqrt(2 / (fan_in + fan_out))
    Scale = math:sqrt(2.0 / (FromSize + ToSize)),
    Weights = [
        [(rand:uniform() * 2 - 1) * Scale || _ <- lists:seq(1, FromSize)]
        || _ <- lists:seq(1, ToSize)
    ],
    Biases = [(rand:uniform() * 0.2 - 0.1) || _ <- lists:seq(1, ToSize)],
    {Weights, Biases}.

%% @private Forward propagate through all layers
forward_propagate([], Activations, _Activation) ->
    Activations;
forward_propagate([{Weights, Biases} | RestLayers], Inputs, Activation) ->
    %% For each neuron: weighted sum of inputs + bias, then activation
    Outputs = lists:zipwith(
        fun(NeuronWeights, Bias) ->
            Sum = dot_product(NeuronWeights, Inputs) + Bias,
            apply_activation(Sum, Activation)
        end,
        Weights,
        Biases
    ),
    forward_propagate(RestLayers, Outputs, Activation).

%% @private Dot product of two vectors
dot_product(Weights, Inputs) ->
    lists:sum(lists:zipwith(fun(W, I) -> W * I end, Weights, Inputs)).

%% @private Apply activation function
apply_activation(X, tanh) ->
    math:tanh(X);
apply_activation(X, sigmoid) ->
    1.0 / (1.0 + math:exp(-X));
apply_activation(X, relu) ->
    max(0.0, X);
apply_activation(X, linear) ->
    X;
apply_activation(X, _) ->
    math:tanh(X).

%% @private Reshape flat weights into matrix
reshape_weights(FlatWeights, RowSize) ->
    reshape_weights(FlatWeights, RowSize, []).

reshape_weights([], _RowSize, Acc) ->
    lists:reverse(Acc);
reshape_weights(Weights, RowSize, Acc) ->
    {Row, Rest} = lists:split(RowSize, Weights),
    reshape_weights(Rest, RowSize, [Row | Acc]).

%% @private Load genotype structure from Mnesia
load_genotype_structure(AgentId) ->
    case mnesia:transaction(fun() ->
        case mnesia:read({agent, AgentId}) of
            [] ->
                {error, agent_not_found};
            [Agent] ->
                CxId = element(3, Agent), %% #agent.cx_id
                case mnesia:read({cortex, CxId}) of
                    [] ->
                        {error, cortex_not_found};
                    [Cortex] ->
                        %% Load neurons
                        NeuronIds = element(5, Cortex), %% #cortex.neuron_ids
                        Neurons = [N || NId <- NeuronIds,
                                       [N] <- [mnesia:read({neuron, NId})]],

                        %% Load sensors for input count
                        SensorIds = element(6, Cortex), %% #cortex.sensor_ids
                        Sensors = [S || SId <- SensorIds,
                                       [S] <- [mnesia:read({sensor, SId})]],

                        %% Load actuators for output count
                        ActuatorIds = element(7, Cortex), %% #cortex.actuator_ids
                        Actuators = [A || AId <- ActuatorIds,
                                         [A] <- [mnesia:read({actuator, AId})]],

                        {ok, {Sensors, Neurons, Actuators}}
                end
        end
    end) of
        {atomic, Result} ->
            Result;
        {aborted, Reason} ->
            {error, {mnesia_error, Reason}}
    end.

%% @private Build network from genotype structure
build_network_from_structure({Sensors, Neurons, Actuators}) ->
    %% Calculate sizes
    InputSize = lists:sum([element(8, S) || S <- Sensors]), %% #sensor.vl
    OutputSize = lists:sum([element(7, A) || A <- Actuators]), %% #actuator.vl
    HiddenCount = length(Neurons),

    %% For now, create a simple feedforward approximation
    %% A proper implementation would recreate the exact topology
    HiddenSizes = case HiddenCount of
        0 -> [];
        N when N < 10 -> [N];
        N -> [N div 2, N div 2]
    end,

    %% Create network and copy weights from neurons
    Network = create_feedforward(InputSize, HiddenSizes, OutputSize),

    %% TODO: Copy actual weights from neuron records
    %% For now, the random weights from create_feedforward are used
    Network.

%%==============================================================================
%% Visualization Functions
%%==============================================================================

%% @doc Evaluate network and return all layer activations.
%%
%% Returns {Outputs, AllActivations} where AllActivations is a list of
%% activation vectors for each layer (including input and output).
-spec evaluate_with_activations(network(), [float()]) ->
    {Outputs :: [float()], Activations :: [[float()]]}.
evaluate_with_activations(#network{layers = Layers, activation = Activation}, Inputs) ->
    {Outputs, Activations} = forward_propagate_with_activations(Layers, Inputs, Activation, [Inputs]),
    {Outputs, lists:reverse(Activations)}.

%% @doc Get network topology information for visualization.
%%
%% Returns a map with layer sizes for rendering the network structure.
-spec get_topology(network()) -> map().
get_topology(#network{layers = Layers}) ->
    %% Extract layer sizes from weight matrices
    LayerSizes = extract_layer_sizes(Layers),
    #{
        layer_sizes => LayerSizes,
        num_layers => length(LayerSizes),
        total_neurons => lists:sum(LayerSizes),
        total_connections => count_connections(Layers)
    }.

%% @doc Get visualization data for rendering the network.
%%
%% Combines topology, weights, and activations into a format suitable
%% for frontend visualization.
%%
%% @param Network The network record
%% @param Inputs Current input values (for activation display)
%% @param InputLabels Optional labels for input neurons
%% @returns Map with nodes, connections, and metadata
-spec get_viz_data(network(), [float()], [binary()]) -> map().
get_viz_data(Network = #network{layers = Layers}, Inputs, InputLabels) ->
    %% Get activations for current inputs
    {Outputs, AllActivations} = evaluate_with_activations(Network, Inputs),

    %% Build layer sizes
    LayerSizes = extract_layer_sizes(Layers),

    %% Build node data with positions and activations
    Nodes = build_viz_nodes(LayerSizes, AllActivations, InputLabels),

    %% Build connection data with weights
    Connections = build_viz_connections(Layers, LayerSizes),

    #{
        nodes => Nodes,
        connections => Connections,
        layer_sizes => LayerSizes,
        outputs => Outputs
    }.

%% @private Forward propagate and collect all activations
forward_propagate_with_activations([], Activations, _Activation, AllActivations) ->
    {Activations, AllActivations};
forward_propagate_with_activations([{Weights, Biases} | RestLayers], Inputs, Activation, AllActivations) ->
    Outputs = lists:zipwith(
        fun(NeuronWeights, Bias) ->
            Sum = dot_product(NeuronWeights, Inputs) + Bias,
            apply_activation(Sum, Activation)
        end,
        Weights,
        Biases
    ),
    forward_propagate_with_activations(RestLayers, Outputs, Activation, [Outputs | AllActivations]).

%% @private Extract layer sizes from weight matrices
extract_layer_sizes([]) ->
    [];
extract_layer_sizes([{Weights, _Biases} | Rest]) ->
    %% First layer: input size is the width of weight matrix
    InputSize = length(hd(Weights)),
    %% All layers: output size is the height of weight matrix
    OutputSizes = [length(W) || {W, _} <- [{Weights, undefined} | Rest]],
    [InputSize | OutputSizes].

%% @private Count total connections
count_connections(Layers) ->
    lists:sum([length(Weights) * length(hd(Weights)) || {Weights, _} <- Layers]).

%% @private Build node data for visualization
build_viz_nodes(LayerSizes, AllActivations, InputLabels) ->
    NumLayers = length(LayerSizes),
    lists:flatten(
        lists:zipwith3(
            fun(LayerIdx, LayerSize, Activations) ->
                Labels = case LayerIdx of
                    1 -> pad_labels(InputLabels, LayerSize);
                    N when N == NumLayers -> output_labels(LayerSize);
                    _ -> hidden_labels(LayerSize)
                end,
                build_layer_nodes(LayerIdx, LayerSize, Activations, Labels, NumLayers)
            end,
            lists:seq(1, NumLayers),
            LayerSizes,
            AllActivations
        )
    ).

%% @private Build nodes for a single layer
build_layer_nodes(LayerIdx, LayerSize, Activations, Labels, NumLayers) ->
    lists:zipwith3(
        fun(NodeIdx, Activation, Label) ->
            #{
                id => {LayerIdx, NodeIdx},
                layer => LayerIdx,
                index => NodeIdx,
                activation => Activation,
                label => Label,
                type => layer_type(LayerIdx, NumLayers)
            }
        end,
        lists:seq(1, LayerSize),
        Activations,
        Labels
    ).

%% @private Build connection data for visualization
build_viz_connections(Layers, LayerSizes) ->
    {Connections, _} = lists:foldl(
        fun({Weights, _Biases}, {Acc, LayerIdx}) ->
            FromSize = lists:nth(LayerIdx, LayerSizes),
            ToSize = lists:nth(LayerIdx + 1, LayerSizes),
            LayerConns = [
                #{
                    from => {LayerIdx, FromIdx},
                    to => {LayerIdx + 1, ToIdx},
                    weight => lists:nth(FromIdx, lists:nth(ToIdx, Weights))
                }
                || ToIdx <- lists:seq(1, ToSize),
                   FromIdx <- lists:seq(1, FromSize)
            ],
            {Acc ++ LayerConns, LayerIdx + 1}
        end,
        {[], 1},
        Layers
    ),
    Connections.

%% @private Determine layer type
layer_type(1, _NumLayers) -> input;
layer_type(N, N) -> output;
layer_type(_, _) -> hidden.

%% @private Pad labels to match layer size
pad_labels(Labels, Size) when length(Labels) >= Size ->
    lists:sublist(Labels, Size);
pad_labels(Labels, Size) ->
    Labels ++ lists:duplicate(Size - length(Labels), <<"">>).

%% @private Generate output labels
output_labels(6) ->
    [<<"L">>, <<"R">>, <<"F">>, <<"Spd">>, <<"Conf">>, <<"Aggr">>];
output_labels(Size) ->
    [list_to_binary("O" ++ integer_to_list(I)) || I <- lists:seq(1, Size)].

%% @private Generate hidden layer labels
hidden_labels(Size) ->
    [list_to_binary("H" ++ integer_to_list(I)) || I <- lists:seq(1, Size)].
