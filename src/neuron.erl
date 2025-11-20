%% @doc Neural processing unit for TWEANN networks.
%%
%% The neuron is the fundamental processing element in a neural network.
%% It receives signals from sensors or other neurons, aggregates them,
%% applies an activation function, and forwards the result to connected
%% neurons or actuators.
%%
%% == Neuron Lifecycle ==
%%
%% 1. Spawned by cortex with initial state
%% 2. Waits for signals from input connections
%% 3. Aggregates all inputs when complete
%% 4. Applies activation function
%% 5. Forwards output to all output connections
%% 6. Repeats from step 2
%%
%% == State ==
%%
%% The neuron maintains:
%%
%% - Input connections with weights
%% - Output connections (PIDs)
%% - Accumulated input signals
%% - Activation function
%% - Aggregation function
%%
%% @author Macula.io
%% @copyright 2025 Macula.io, Apache-2.0
-module(neuron).

-export([
    start_link/1,
    init/1,
    forward/3,
    backup/1
]).

-record(state, {
    id :: term(),
    cortex_pid :: pid(),
    activation_function :: atom(),
    aggregation_function :: atom(),
    input_pids :: [pid()],
    output_pids :: [pid()],
    ro_pids :: [pid()],  % recurrent output PIDs
    input_weights :: #{pid() => [{float(), float(), float(), list()}]},
    bias :: float(),
    acc_input :: #{pid() => [float()]},
    expected_inputs :: non_neg_integer()
}).

%% @doc Start a neuron process.
%%
%% Options:
%% - `id' - Unique identifier for this neuron
%% - `cortex_pid' - PID of the controlling cortex
%% - `activation_function' - Atom naming the activation function (e.g., tanh)
%% - `aggregation_function' - Atom naming the aggregation function (e.g., dot_product)
%% - `input_pids' - List of PIDs that send input to this neuron
%% - `output_pids' - List of PIDs to forward output to
%% - `ro_pids' - List of recurrent output PIDs
%% - `input_weights' - Map of PID to list of weight tuples
%% - `bias' - Bias value for this neuron
-spec start_link(map()) -> {ok, pid()}.
start_link(Opts) ->
    Pid = spawn_link(?MODULE, init, [Opts]),
    {ok, Pid}.

%% @doc Initialize the neuron and enter the main loop.
-spec init(map()) -> no_return().
init(Opts) ->
    Id = maps:get(id, Opts),
    CortexPid = maps:get(cortex_pid, Opts),
    ActivationFn = maps:get(activation_function, Opts, tanh),
    AggregationFn = maps:get(aggregation_function, Opts, dot_product),
    InputPids = maps:get(input_pids, Opts, []),
    OutputPids = maps:get(output_pids, Opts, []),
    RoPids = maps:get(ro_pids, Opts, []),
    InputWeights = maps:get(input_weights, Opts, #{}),
    Bias = maps:get(bias, Opts, 0.0),

    State = #state{
        id = Id,
        cortex_pid = CortexPid,
        activation_function = ActivationFn,
        aggregation_function = AggregationFn,
        input_pids = InputPids,
        output_pids = OutputPids,
        ro_pids = RoPids,
        input_weights = InputWeights,
        bias = Bias,
        acc_input = #{},
        expected_inputs = length(InputPids)
    },

    loop(State).

%% @doc Send a signal to a neuron.
%%
%% Called by sensors or other neurons to forward their output.
-spec forward(pid(), pid(), [float()]) -> ok.
forward(NeuronPid, FromPid, Signal) ->
    NeuronPid ! {forward, FromPid, Signal},
    ok.

%% @doc Request the neuron to backup its current weights.
%%
%% The neuron will send its weights to the cortex for storage.
-spec backup(pid()) -> ok.
backup(NeuronPid) ->
    NeuronPid ! backup,
    ok.

%% Internal functions

loop(State) ->
    receive
        {forward, FromPid, Signal} ->
            NewState = handle_forward(FromPid, Signal, State),
            loop(NewState);

        backup ->
            _ = handle_backup(State),
            loop(State);

        {cortex, terminate} ->
            ok;

        {update_weights, NewWeights, NewBias} ->
            NewState = State#state{
                input_weights = NewWeights,
                bias = NewBias
            },
            loop(NewState);

        %% Dynamic linking from constructor
        {link, input_pids, InputPids} ->
            NewState = State#state{
                input_pids = InputPids,
                expected_inputs = length(InputPids)
            },
            loop(NewState);

        {link, output_pids, OutputPids} ->
            loop(State#state{output_pids = OutputPids});

        {link, ro_pids, RoPids} ->
            loop(State#state{ro_pids = RoPids});

        {link, input_weights, InputWeights} ->
            loop(State#state{input_weights = InputWeights})
    end.

handle_forward(FromPid, Signal, State) ->
    #state{
        acc_input = AccInput,
        expected_inputs = ExpectedInputs
    } = State,

    %% Accumulate the signal
    NewAccInput = maps:put(FromPid, Signal, AccInput),
    ReceivedCount = maps:size(NewAccInput),

    %% Check if we have all inputs
    case ReceivedCount >= ExpectedInputs of
        true ->
            process_and_forward(State#state{acc_input = NewAccInput});
        false ->
            State#state{acc_input = NewAccInput}
    end.

process_and_forward(State) ->
    #state{
        activation_function = ActivationFn,
        aggregation_function = AggregationFn,
        output_pids = OutputPids,
        ro_pids = RoPids,
        input_weights = InputWeights,
        bias = Bias,
        acc_input = AccInput,
        input_pids = InputPids
    } = State,

    %% Build input list in correct order
    Inputs = build_inputs(InputPids, AccInput),

    %% Build weights list in correct order
    Weights = build_weights(InputPids, InputWeights),

    %% Aggregate inputs
    Aggregated = aggregate(AggregationFn, Inputs, Weights),

    %% Add bias and apply activation
    Output = activate(ActivationFn, Aggregated + Bias),

    %% Forward to all output connections
    lists:foreach(
        fun(OutputPid) ->
            OutputPid ! {forward, self(), [Output]}
        end,
        OutputPids
    ),

    %% Forward to recurrent outputs
    lists:foreach(
        fun(RoPid) ->
            RoPid ! {forward, self(), [Output]}
        end,
        RoPids
    ),

    %% Reset accumulated inputs
    State#state{acc_input = #{}}.

build_inputs(InputPids, AccInput) ->
    [{Pid, maps:get(Pid, AccInput, [0.0])} || Pid <- InputPids].

build_weights(InputPids, InputWeights) ->
    [{Pid, maps:get(Pid, InputWeights, [{1.0, 0.0, 0.1, []}])} || Pid <- InputPids].

aggregate(dot_product, Inputs, Weights) ->
    signal_aggregator:dot_product(Inputs, Weights);
aggregate(mult_product, Inputs, Weights) ->
    signal_aggregator:mult_product(Inputs, Weights);
aggregate(diff_product, Inputs, Weights) ->
    signal_aggregator:diff_product(Inputs, Weights);
aggregate(Function, Inputs, Weights) ->
    signal_aggregator:Function(Inputs, Weights).

activate(tanh, X) -> functions:tanh(X);
activate(sigmoid, X) -> functions:sigmoid(X);
activate(sigmoid1, X) -> functions:sigmoid1(X);
activate(sin, X) -> functions:sin(X);
activate(cos, X) -> functions:cos(X);
activate(gaussian, X) -> functions:gaussian(X);
activate(linear, X) -> functions:linear(X);
activate(sgn, X) -> functions:sgn(X);
activate(bin, X) -> functions:bin(X);
activate(trinary, X) -> functions:trinary(X);
activate(multiquadric, X) -> functions:multiquadric(X);
activate(quadratic, X) -> functions:quadratic(X);
activate(cubic, X) -> functions:cubic(X);
activate(absolute, X) -> functions:absolute(X);
activate(sqrt, X) -> functions:sqrt(X);
activate(log, X) -> functions:log(X);
activate(relu, X) -> functions:relu(X);
activate(Function, X) -> functions:Function(X).

handle_backup(State) ->
    #state{
        id = Id,
        cortex_pid = CortexPid,
        input_weights = InputWeights,
        bias = Bias
    } = State,

    CortexPid ! {backup, Id, InputWeights, Bias}.
