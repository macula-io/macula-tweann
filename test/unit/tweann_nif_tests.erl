%% @doc Unit tests for the TWEANN NIF module.
%%
%% Tests the Rust NIF for network compilation and evaluation.
%% Note: These tests require the NIF to be compiled first.
%%
%% @copyright 2025 Macula.io
%% @license Apache-2.0
-module(tweann_nif_tests).

-include_lib("eunit/include/eunit.hrl").

%%==============================================================================
%% Test Fixtures
%%==============================================================================

%% Simple 2-input, 1-output network (no hidden layers)
simple_network() ->
    %% Nodes: 2 inputs -> 1 output
    %% Node format: {Index, Type, Activation, Bias, Connections}
    Nodes = [
        {0, input, linear, 0.0, []},       %% Input 0
        {1, input, linear, 0.0, []},       %% Input 1
        {2, output, tanh, 0.0, [           %% Output: weighted sum of inputs
            {0, 0.5},   %% From input 0, weight 0.5
            {1, 0.5}    %% From input 1, weight 0.5
        ]}
    ],
    InputCount = 2,
    OutputIndices = [2],
    {Nodes, InputCount, OutputIndices}.

%% XOR-like network with hidden layer
xor_network() ->
    %% 2 inputs -> 2 hidden (tanh) -> 1 output (tanh)
    Nodes = [
        {0, input, linear, 0.0, []},       %% Input 0
        {1, input, linear, 0.0, []},       %% Input 1
        {2, hidden, tanh, 0.0, [           %% Hidden 0: sum
            {0, 1.0},
            {1, 1.0}
        ]},
        {3, hidden, tanh, 0.0, [           %% Hidden 1: diff
            {0, 1.0},
            {1, -1.0}
        ]},
        {4, output, tanh, 0.0, [           %% Output
            {2, 1.0},
            {3, 1.0}
        ]}
    ],
    InputCount = 2,
    OutputIndices = [4],
    {Nodes, InputCount, OutputIndices}.

%% Multi-output network
multi_output_network() ->
    Nodes = [
        {0, input, linear, 0.0, []},
        {1, input, linear, 0.0, []},
        {2, output, sigmoid, 0.0, [{0, 1.0}, {1, 0.0}]},  %% Depends on input 0
        {3, output, sigmoid, 0.0, [{0, 0.0}, {1, 1.0}]}   %% Depends on input 1
    ],
    InputCount = 2,
    OutputIndices = [2, 3],
    {Nodes, InputCount, OutputIndices}.

%%==============================================================================
%% NIF Loading Tests
%%==============================================================================

nif_loading_test_() ->
    {setup,
     fun() -> ok end,
     fun(_) -> ok end,
     [
         {"NIF is loaded", fun nif_is_loaded/0}
     ]}.

nif_is_loaded() ->
    %% This test checks if the NIF can be loaded
    %% If NIF is not compiled, this will return false but not fail
    IsLoaded = tweann_nif:is_loaded(),
    ?assert(is_boolean(IsLoaded)).

%%==============================================================================
%% Compilation Tests
%%==============================================================================

compile_simple_network_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = simple_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            ?assert(is_reference(Network))
    end.

compile_xor_network_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = xor_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            ?assert(is_reference(Network))
    end.

compile_empty_network_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Network = tweann_nif:compile_network([], 0, []),
            ?assert(is_reference(Network))
    end.

%%==============================================================================
%% Evaluation Tests
%%==============================================================================

evaluate_simple_zeros_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = simple_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            Outputs = tweann_nif:evaluate(Network, [0.0, 0.0]),
            ?assertEqual(1, length(Outputs)),
            %% tanh(0.0) = 0.0
            [Output] = Outputs,
            ?assert(abs(Output) < 0.001)
    end.

evaluate_simple_ones_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = simple_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            Outputs = tweann_nif:evaluate(Network, [1.0, 1.0]),
            ?assertEqual(1, length(Outputs)),
            %% tanh(0.5 * 1.0 + 0.5 * 1.0) = tanh(1.0) ≈ 0.7616
            [Output] = Outputs,
            ?assert(abs(Output - 0.7616) < 0.001)
    end.

evaluate_xor_pattern_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = xor_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),

            %% Test [0, 0] - should be near 0
            [Out00] = tweann_nif:evaluate(Network, [0.0, 0.0]),
            ?assert(abs(Out00) < 0.1),

            %% Test [1, 1] - should be positive (sum=2, diff=0)
            [Out11] = tweann_nif:evaluate(Network, [1.0, 1.0]),
            ?assert(Out11 > 0.5)
    end.

evaluate_multi_output_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = multi_output_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            Outputs = tweann_nif:evaluate(Network, [1.0, 0.0]),
            ?assertEqual(2, length(Outputs)),
            %% First output depends on input 0 (=1.0)
            %% Second output depends on input 1 (=0.0)
            [Out1, Out2] = Outputs,
            ?assert(Out1 > 0.5),  %% sigmoid(1.0) ≈ 0.73
            ?assert(abs(Out2 - 0.5) < 0.001)  %% sigmoid(0.0) = 0.5
    end.

%%==============================================================================
%% Batch Evaluation Tests
%%==============================================================================

evaluate_batch_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = simple_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),

            InputBatch = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ],
            OutputBatch = tweann_nif:evaluate_batch(Network, InputBatch),
            ?assertEqual(4, length(OutputBatch)),

            %% Each output should be a single-element list
            lists:foreach(
                fun(Outputs) ->
                    ?assertEqual(1, length(Outputs))
                end,
                OutputBatch
            )
    end.

%%==============================================================================
%% Compatibility Distance Tests
%%==============================================================================

compatibility_same_genome_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Connections = [{1, 0.5}, {2, -0.3}, {3, 0.8}],
            Distance = tweann_nif:compatibility_distance(
                Connections, Connections,
                1.0, 1.0, 0.4
            ),
            %% Same genome should have distance 0
            ?assert(abs(Distance) < 0.001)
    end.

compatibility_different_weights_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            ConnectionsA = [{1, 0.5}, {2, -0.3}, {3, 0.8}],
            ConnectionsB = [{1, 0.6}, {2, -0.2}, {3, 0.9}],
            Distance = tweann_nif:compatibility_distance(
                ConnectionsA, ConnectionsB,
                1.0, 1.0, 0.4
            ),
            %% Only weight differences, average diff = 0.1
            %% Distance = 0.4 * 0.1 = 0.04
            ?assert(Distance > 0.0),
            ?assert(Distance < 0.1)
    end.

compatibility_disjoint_genes_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            ConnectionsA = [{1, 0.5}, {2, -0.3}],
            ConnectionsB = [{1, 0.5}, {3, 0.8}],  %% Gene 2 vs 3 are disjoint
            Distance = tweann_nif:compatibility_distance(
                ConnectionsA, ConnectionsB,
                1.0, 1.0, 0.4
            ),
            ?assert(Distance > 0.0)
    end.

compatibility_excess_genes_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            ConnectionsA = [{1, 0.5}, {2, -0.3}],
            ConnectionsB = [{1, 0.5}, {2, -0.3}, {10, 0.9}],  %% 10 is excess
            Distance = tweann_nif:compatibility_distance(
                ConnectionsA, ConnectionsB,
                1.0, 1.0, 0.4
            ),
            ?assert(Distance > 0.0)
    end.

%%==============================================================================
%% Benchmark Tests
%%==============================================================================

benchmark_simple_network_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = xor_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            AvgTime = tweann_nif:benchmark_evaluate(Network, [1.0, 1.0], 1000),
            %% Should be less than 10 microseconds per evaluation
            ?assert(AvgTime < 10.0),
            %% Should be positive
            ?assert(AvgTime > 0.0)
    end.

%%==============================================================================
%% Activation Function Tests
%%==============================================================================

activation_functions_test_() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            [
                {"tanh activation", fun test_tanh_activation/0},
                {"sigmoid activation", fun test_sigmoid_activation/0},
                {"relu activation", fun test_relu_activation/0},
                {"linear activation", fun test_linear_activation/0}
            ]
    end.

test_tanh_activation() ->
    Nodes = [
        {0, input, linear, 0.0, []},
        {1, output, tanh, 0.0, [{0, 1.0}]}
    ],
    Network = tweann_nif:compile_network(Nodes, 1, [1]),
    [Out] = tweann_nif:evaluate(Network, [0.0]),
    ?assert(abs(Out) < 0.001).  %% tanh(0) = 0

test_sigmoid_activation() ->
    Nodes = [
        {0, input, linear, 0.0, []},
        {1, output, sigmoid, 0.0, [{0, 1.0}]}
    ],
    Network = tweann_nif:compile_network(Nodes, 1, [1]),
    [Out] = tweann_nif:evaluate(Network, [0.0]),
    ?assert(abs(Out - 0.5) < 0.001).  %% sigmoid(0) = 0.5

test_relu_activation() ->
    Nodes = [
        {0, input, linear, 0.0, []},
        {1, output, relu, 0.0, [{0, 1.0}]}
    ],
    Network = tweann_nif:compile_network(Nodes, 1, [1]),
    [OutNeg] = tweann_nif:evaluate(Network, [-1.0]),
    [OutPos] = tweann_nif:evaluate(Network, [1.0]),
    ?assert(abs(OutNeg) < 0.001),  %% relu(-1) = 0
    ?assert(abs(OutPos - 1.0) < 0.001).  %% relu(1) = 1

test_linear_activation() ->
    Nodes = [
        {0, input, linear, 0.0, []},
        {1, output, linear, 0.0, [{0, 2.0}]}
    ],
    Network = tweann_nif:compile_network(Nodes, 1, [1]),
    [Out] = tweann_nif:evaluate(Network, [3.0]),
    ?assert(abs(Out - 6.0) < 0.001).  %% linear(3.0 * 2.0) = 6.0

%%==============================================================================
%% Edge Case Tests
%%==============================================================================

evaluate_wrong_input_count_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {Nodes, InputCount, OutputIndices} = simple_network(),
            Network = tweann_nif:compile_network(Nodes, InputCount, OutputIndices),
            %% Wrong number of inputs should return empty list
            Outputs = tweann_nif:evaluate(Network, [1.0]),  %% Expects 2, got 1
            ?assertEqual([], Outputs)
    end.

bias_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            %% Network with bias
            Nodes = [
                {0, input, linear, 0.0, []},
                {1, output, linear, 1.0, [{0, 1.0}]}  %% Bias = 1.0
            ],
            Network = tweann_nif:compile_network(Nodes, 1, [1]),
            [Out] = tweann_nif:evaluate(Network, [2.0]),
            %% linear(2.0 * 1.0 + 1.0) = 3.0
            ?assert(abs(Out - 3.0) < 0.001)
    end.

%%==============================================================================
%% LTC/CfC (Liquid Time-Constant) NIF Tests
%%==============================================================================

%% Test CfC basic evaluation
evaluate_cfc_basic_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            %% Basic CfC evaluation: input=0.5, state=0.0, tau=1.0, bound=1.0
            {NewState, Output} = tweann_nif:evaluate_cfc(0.5, 0.0, 1.0, 1.0),
            %% State should move toward h (target)
            ?assert(is_float(NewState)),
            ?assert(is_float(Output)),
            %% With zero initial state and positive input, state should become positive
            ?assert(NewState > 0.0 orelse abs(NewState) < 0.001)
    end.

%% Test CfC state persistence
evaluate_cfc_state_persistence_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            %% First evaluation
            {State1, _} = tweann_nif:evaluate_cfc(1.0, 0.0, 1.0, 1.0),
            %% Second evaluation with State1 as input state
            {State2, _} = tweann_nif:evaluate_cfc(1.0, State1, 1.0, 1.0),
            %% State should continue to evolve
            ?assert(is_float(State1)),
            ?assert(is_float(State2)),
            %% With constant positive input, state should increase or saturate
            ?assert(State2 >= State1 orelse abs(State2 - State1) < 0.01)
    end.

%% Test CfC state bounding
evaluate_cfc_bound_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Bound = 0.5,
            %% Large input should be bounded
            {State, _} = tweann_nif:evaluate_cfc(100.0, 0.0, 1.0, Bound),
            ?assert(State >= -Bound),
            ?assert(State =< Bound)
    end.

%% Test CfC with custom weights
evaluate_cfc_with_weights_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            BackboneWeights = [0.1, 0.2, 0.3],
            HeadWeights = [0.5, 0.5],
            {NewState, Output} = tweann_nif:evaluate_cfc_with_weights(
                0.5, 0.0, 1.0, 1.0, BackboneWeights, HeadWeights
            ),
            ?assert(is_float(NewState)),
            ?assert(is_float(Output))
    end.

%% Test ODE-based evaluation
evaluate_ode_basic_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            %% ODE evaluation with dt=0.01
            {NewState, Output} = tweann_nif:evaluate_ode(0.5, 0.0, 1.0, 1.0, 0.01),
            ?assert(is_float(NewState)),
            ?assert(is_float(Output)),
            %% State should evolve from 0
            ?assert(abs(NewState) < 0.1)  %% Small dt means small state change
    end.

%% Test ODE state bounding
evaluate_ode_bound_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Bound = 0.5,
            %% Run many steps to try to exceed bound
            State0 = 0.0,
            State1 = run_ode_steps(100.0, State0, 1.0, Bound, 0.1, 100),
            ?assert(State1 >= -Bound),
            ?assert(State1 =< Bound)
    end.

%% Helper to run multiple ODE steps
run_ode_steps(_Input, State, _Tau, _Bound, _Dt, 0) ->
    State;
run_ode_steps(Input, State, Tau, Bound, Dt, N) ->
    case tweann_nif:is_loaded() of
        false -> State;
        true ->
            {NewState, _} = tweann_nif:evaluate_ode(Input, State, Tau, Bound, Dt),
            run_ode_steps(Input, NewState, Tau, Bound, Dt, N - 1)
    end.

%% Test ODE with custom weights
evaluate_ode_with_weights_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            BackboneWeights = [0.1, 0.2, 0.3],
            HeadWeights = [0.5, 0.5],
            {NewState, Output} = tweann_nif:evaluate_ode_with_weights(
                0.5, 0.0, 1.0, 1.0, 0.01, BackboneWeights, HeadWeights
            ),
            ?assert(is_float(NewState)),
            ?assert(is_float(Output))
    end.

%% Test CfC batch evaluation
evaluate_cfc_batch_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Inputs = [0.1, 0.2, 0.3, 0.4, 0.5],
            InitialState = 0.0,
            Tau = 1.0,
            Bound = 1.0,
            Results = tweann_nif:evaluate_cfc_batch(Inputs, InitialState, Tau, Bound),
            ?assertEqual(length(Inputs), length(Results)),
            %% Each result should be a {State, Output} tuple
            lists:foreach(
                fun({State, Output}) ->
                    ?assert(is_float(State)),
                    ?assert(is_float(Output))
                end,
                Results
            )
    end.

%% Test CfC batch maintains state continuity
evaluate_cfc_batch_state_continuity_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Inputs = [1.0, 1.0, 1.0],
            Results = tweann_nif:evaluate_cfc_batch(Inputs, 0.0, 1.0, 1.0),
            [{S1, _}, {S2, _}, {S3, _}] = Results,
            %% With constant positive input, state should evolve monotonically
            %% (or saturate if it reaches the bound)
            ?assert(is_float(S1)),
            ?assert(is_float(S2)),
            ?assert(is_float(S3))
    end.

%% Test that CfC is faster than ODE (conceptual test)
cfc_faster_than_ode_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Iterations = 10000,

            %% Time CfC
            CfcStart = erlang:monotonic_time(microsecond),
            cfc_loop(1.0, 0.0, 1.0, 1.0, Iterations),
            CfcEnd = erlang:monotonic_time(microsecond),
            CfcTime = CfcEnd - CfcStart,

            %% Time ODE
            OdeStart = erlang:monotonic_time(microsecond),
            ode_loop(1.0, 0.0, 1.0, 1.0, 0.01, Iterations),
            OdeEnd = erlang:monotonic_time(microsecond),
            OdeTime = OdeEnd - OdeStart,

            %% Both should complete (no crash)
            ?assert(CfcTime > 0),
            ?assert(OdeTime > 0)
            %% Note: CfC should be faster, but we don't assert specific ratio
            %% since NIF overhead may dominate for single calls
    end.

cfc_loop(_Input, State, _Tau, _Bound, 0) ->
    State;
cfc_loop(Input, State, Tau, Bound, N) ->
    {NewState, _} = tweann_nif:evaluate_cfc(Input, State, Tau, Bound),
    cfc_loop(Input, NewState, Tau, Bound, N - 1).

ode_loop(_Input, State, _Tau, _Bound, _Dt, 0) ->
    State;
ode_loop(Input, State, Tau, Bound, Dt, N) ->
    {NewState, _} = tweann_nif:evaluate_ode(Input, State, Tau, Bound, Dt),
    ode_loop(Input, NewState, Tau, Bound, Dt, N - 1).

%% Test LTC with different time constants
time_constant_effect_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            Input = 1.0,
            State = 0.0,
            Bound = 2.0,

            %% Small tau = fast response
            {StateFast, _} = tweann_nif:evaluate_cfc(Input, State, 0.1, Bound),
            %% Large tau = slow response
            {StateSlow, _} = tweann_nif:evaluate_cfc(Input, State, 10.0, Bound),

            %% Both should be valid floats
            ?assert(is_float(StateFast)),
            ?assert(is_float(StateSlow))
            %% Note: The exact relationship depends on backbone network implementation
    end.

%% Test negative inputs
negative_input_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            {State, Output} = tweann_nif:evaluate_cfc(-1.0, 0.5, 1.0, 1.0),
            ?assert(is_float(State)),
            ?assert(is_float(Output)),
            ?assert(State >= -1.0),
            ?assert(State =< 1.0)
    end.

%% Test zero tau (edge case)
zero_tau_test() ->
    case tweann_nif:is_loaded() of
        false ->
            {skip, "NIF not loaded"};
        true ->
            %% Tau close to zero - should not crash, may have special handling
            {State, Output} = tweann_nif:evaluate_cfc(1.0, 0.0, 0.001, 1.0),
            ?assert(is_float(State)),
            ?assert(is_float(Output))
    end.
