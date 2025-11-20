%% @doc Genetic mutation operators for neural network evolution.
%%
%% This module provides mutation operators that modify network genotypes
%% to explore the solution space. Mutations are categorized as:
%%
%% == Topological Mutations ==
%% Modify network structure:
%% - add_neuron: Insert neuron into existing connection
%% - add_outlink: Add output connection from neuron
%% - add_inlink: Add input connection to neuron
%% - add_sensorlink: Connect sensor to neuron
%% - add_actuatorlink: Connect neuron to actuator
%% - outsplice: Split output connection with new neuron
%%
%% == Parametric Mutations ==
%% Modify values without changing structure:
%% - mutate_weights: Perturb synaptic weights
%% - mutate_af: Change activation function
%% - mutate_plasticity: Modify learning parameters
%% - mutate_aggr_f: Change aggregation function
%%
%% == Evolutionary Strategy Mutations ==
%% Meta-evolution of search parameters:
%% - mutate_tuning_selection: Change weight selection strategy
%% - mutate_annealing: Modify simulated annealing schedule
%% - mutate_heredity_type: Switch between darwinian/lamarckian
%%
%% == Mutation Selection ==
%% Mutations are selected using roulette wheel selection
%% weighted by mutation probabilities from the constraint.
%%
%% @author Macula.io
%% @copyright 2025 Macula.io, Apache-2.0
-module(genome_mutator).

-include("records.hrl").

%% Suppress supertype warnings for polymorphic functions
-dialyzer({nowarn_function, [
    mutate_agent_parameter/3,
    mutate/1,
    apply_mutation/2,
    get_agent_field/2,
    set_agent_field/3,
    mutate_af/1,
    mutate_aggr_f/1,
    add_bias/1,
    add_outlink/1,
    add_inlink/1,
    add_neuron/1,
    outsplice/1,
    add_sensorlink/1,
    add_actuatorlink/1,
    add_sensor/1,
    add_actuator/1,
    update_source_output/3,
    update_target_input/4
]}).

-export([
    %% Main mutation interface
    mutate/1,
    mutate/2,

    %% Parametric mutations
    mutate_agent_parameter/3,
    mutate_tuning_selection/1,
    mutate_tuning_annealing/1,
    mutate_tot_topological_mutations/1,
    mutate_heredity_type/1,
    mutate_weights/1,
    mutate_af/1,
    mutate_aggr_f/1,

    %% Topological mutations
    add_bias/1,
    add_outlink/1,
    add_inlink/1,
    add_neuron/1,
    outsplice/1,
    add_sensorlink/1,
    add_actuatorlink/1,
    add_sensor/1,
    add_actuator/1,

    %% Utility
    select_random_neuron/1,
    calculate_mutation_count/1
]).

%% Delta multiplier for weight perturbation.
%% Using 2*pi (~6.28) as base perturbation range provides
%% sufficient exploration while maintaining stability.
-define(DELTA_MULTIPLIER, math:pi() * 2).

%% Search parameters mutation probability.
%% Probability of mutating evolutionary strategy parameters
%% (tuning selection, annealing, etc.) during each mutation cycle.
-define(SEARCH_PARAMETERS_MUTATION_PROBABILITY, 0).

%% ============================================================================
%% Main Mutation Interface
%% ============================================================================

%% @doc Apply mutations to an agent.
%%
%% Selects and applies mutations based on the agent's constraint.
%% The number of mutations is determined by the tot_topological_mutations_f.
%%
%% @param AgentId the agent to mutate
%% @returns ok
-spec mutate(term()) -> ok.
mutate(AgentId) ->
    MutationCount = calculate_mutation_count(AgentId),
    mutate(AgentId, MutationCount).

%% @doc Apply a specific number of mutations to an agent.
%%
%% @param AgentId the agent to mutate
%% @param Count number of mutations to apply
%% @returns ok
-spec mutate(term(), non_neg_integer()) -> ok.
mutate(AgentId, Count) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    MutationOperators = Agent#agent.mutation_operators,

    lists:foreach(
        fun(_) ->
            Operator = selection_utils:roulette_wheel(MutationOperators),
            apply_mutation(AgentId, Operator)
        end,
        lists:seq(1, Count)
    ),
    ok.

%% @doc Calculate number of mutations based on agent's mutation function.
-spec calculate_mutation_count(term()) -> pos_integer().
calculate_mutation_count(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),
    NeuronCount = length(Cortex#cortex.neuron_ids),

    %% Get mutation function from agent (stored as single {MutationF, Param} tuple)
    {MutationF, Param} = Agent#agent.tot_topological_mutations_f,

    calculate_count(MutationF, NeuronCount, Param).

%% @private Calculate count based on mutation function
-spec calculate_count(atom(), non_neg_integer(), float()) -> pos_integer().
calculate_count(ncount_exponential, NeuronCount, Param) ->
    %% Exponential decay based on network size
    max(1, round(NeuronCount * math:pow(Param, NeuronCount)));
calculate_count(ncount_linear, NeuronCount, Param) ->
    %% Linear scaling
    max(1, round(NeuronCount * Param));
calculate_count(_Default, _NeuronCount, _Param) ->
    1.

%% @private Apply a specific mutation operator
-spec apply_mutation(term(), atom()) -> ok | {error, term()}.
apply_mutation(AgentId, add_bias) -> add_bias(AgentId);
apply_mutation(AgentId, add_outlink) -> add_outlink(AgentId);
apply_mutation(AgentId, add_inlink) -> add_inlink(AgentId);
apply_mutation(AgentId, add_neuron) -> add_neuron(AgentId);
apply_mutation(AgentId, outsplice) -> outsplice(AgentId);
apply_mutation(AgentId, add_sensorlink) -> add_sensorlink(AgentId);
apply_mutation(AgentId, add_actuatorlink) -> add_actuatorlink(AgentId);
apply_mutation(AgentId, add_sensor) -> add_sensor(AgentId);
apply_mutation(AgentId, add_actuator) -> add_actuator(AgentId);
apply_mutation(AgentId, mutate_weights) -> mutate_weights(AgentId);
apply_mutation(AgentId, mutate_af) -> mutate_af(AgentId);
apply_mutation(AgentId, mutate_aggr_f) -> mutate_aggr_f(AgentId);
apply_mutation(_AgentId, add_cpp) -> ok; % Substrate - not implemented
apply_mutation(_AgentId, add_cep) -> ok; % Substrate - not implemented
apply_mutation(_AgentId, _Unknown) -> ok.

%% ============================================================================
%% Parametric Mutations (Evolutionary Strategy)
%% ============================================================================

%% @doc Generic function to mutate an agent parameter.
%%
%% Reads the current value of a field, gets alternatives from constraint,
%% and selects a new random value.
%%
%% @param AgentId the agent to mutate
%% @param FieldName the agent record field to mutate
%% @param ConstraintField the constraint field with alternatives
%% @returns ok or {error, no_alternatives}
-spec mutate_agent_parameter(term(), atom(), atom()) -> ok | {error, no_alternatives}.
mutate_agent_parameter(AgentId, FieldName, ConstraintField) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Constraint = Agent#agent.constraint,

    CurrentValue = get_agent_field(Agent, FieldName),
    AvailableValues = get_constraint_field(Constraint, ConstraintField),
    Alternatives = AvailableValues -- [CurrentValue],

    case Alternatives of
        [] ->
            {error, no_alternatives};
        Values ->
            NewValue = selection_utils:random_select(Values),
            UpdatedAgent = set_agent_field(Agent, FieldName, NewValue),
            genotype:write(UpdatedAgent),
            ok
    end.

%% @doc Mutate tuning selection function.
-spec mutate_tuning_selection(term()) -> ok | {error, no_alternatives}.
mutate_tuning_selection(AgentId) ->
    mutate_agent_parameter(AgentId, tuning_selection_f, tuning_selection_fs).

%% @doc Mutate annealing parameter.
-spec mutate_tuning_annealing(term()) -> ok | {error, no_alternatives}.
mutate_tuning_annealing(AgentId) ->
    mutate_agent_parameter(AgentId, annealing_parameter, annealing_parameters).

%% @doc Mutate total topological mutations function.
-spec mutate_tot_topological_mutations(term()) -> ok | {error, no_alternatives}.
mutate_tot_topological_mutations(AgentId) ->
    mutate_agent_parameter(AgentId, tot_topological_mutations_f, tot_topological_mutations_fs).

%% @doc Mutate heredity type (darwinian/lamarckian).
-spec mutate_heredity_type(term()) -> ok | {error, no_alternatives}.
mutate_heredity_type(AgentId) ->
    mutate_agent_parameter(AgentId, heredity_type, heredity_types).

%% @private Get field value from agent record
-spec get_agent_field(#agent{}, atom()) -> term().
get_agent_field(Agent, tuning_selection_f) -> Agent#agent.tuning_selection_f;
get_agent_field(Agent, annealing_parameter) -> Agent#agent.annealing_parameter;
get_agent_field(Agent, tot_topological_mutations_f) -> Agent#agent.tot_topological_mutations_f;
get_agent_field(Agent, heredity_type) -> Agent#agent.heredity_type;
get_agent_field(Agent, perturbation_range) -> Agent#agent.perturbation_range.

%% @private Set field value in agent record
-spec set_agent_field(#agent{}, atom(), term()) -> #agent{}.
set_agent_field(Agent, tuning_selection_f, Value) ->
    Agent#agent{tuning_selection_f = Value};
set_agent_field(Agent, annealing_parameter, Value) ->
    Agent#agent{annealing_parameter = Value};
set_agent_field(Agent, tot_topological_mutations_f, Value) ->
    Agent#agent{tot_topological_mutations_f = Value};
set_agent_field(Agent, heredity_type, Value) ->
    Agent#agent{heredity_type = Value};
set_agent_field(Agent, perturbation_range, Value) ->
    Agent#agent{perturbation_range = Value}.

%% @private Get field value from constraint record
-spec get_constraint_field(#constraint{}, atom()) -> list().
get_constraint_field(C, tuning_selection_fs) -> C#constraint.tuning_selection_fs;
get_constraint_field(C, annealing_parameters) -> C#constraint.annealing_parameters;
get_constraint_field(C, tot_topological_mutations_fs) -> C#constraint.tot_topological_mutations_fs;
get_constraint_field(C, heredity_types) -> C#constraint.heredity_types;
get_constraint_field(C, perturbation_ranges) -> C#constraint.perturbation_ranges.

%% ============================================================================
%% Parametric Mutations (Network Parameters)
%% ============================================================================

%% @doc Mutate weights of a random neuron.
%%
%% Selects a random neuron and perturbs its input weights
%% using the agent's perturbation range.
%%
%% @param AgentId the agent to mutate
%% @returns ok
-spec mutate_weights(term()) -> ok.
mutate_weights(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            ok;
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),
            PerturbRange = Agent#agent.perturbation_range,

            %% Perturb all weights
            NewInputIdps = [
                {InputId, perturbation_utils:perturb_weights(Weights, PerturbRange * ?DELTA_MULTIPLIER)}
                || {InputId, Weights} <- Neuron#neuron.input_idps
            ],

            UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
            genotype:write(UpdatedNeuron),
            ok
    end.

%% @doc Mutate activation function of a random neuron.
%%
%% Selects a random neuron and changes its activation function
%% to another available function from the constraint.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, no_alternatives}
-spec mutate_af(term()) -> ok | {error, term()}.
mutate_af(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Constraint = Agent#agent.constraint,
    AvailableAFs = Constraint#constraint.neural_afs,

    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            {error, no_neurons};
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),
            CurrentAF = Neuron#neuron.af,
            Alternatives = AvailableAFs -- [CurrentAF],

            case Alternatives of
                [] ->
                    {error, no_alternatives};
                AFs ->
                    NewAF = selection_utils:random_select(AFs),
                    UpdatedNeuron = Neuron#neuron{af = NewAF},
                    genotype:write(UpdatedNeuron),
                    ok
            end
    end.

%% @doc Mutate aggregation function of a random neuron.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, no_alternatives}
-spec mutate_aggr_f(term()) -> ok | {error, term()}.
mutate_aggr_f(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Constraint = Agent#agent.constraint,
    AvailableAggrFs = Constraint#constraint.neural_aggr_fs,

    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            {error, no_neurons};
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),
            CurrentAggrF = Neuron#neuron.aggr_f,
            Alternatives = AvailableAggrFs -- [CurrentAggrF],

            case Alternatives of
                [] ->
                    {error, no_alternatives};
                AggrFs ->
                    NewAggrF = selection_utils:random_select(AggrFs),
                    UpdatedNeuron = Neuron#neuron{aggr_f = NewAggrF},
                    genotype:write(UpdatedNeuron),
                    ok
            end
    end.

%% ============================================================================
%% Topological Mutations
%% ============================================================================

%% @doc Add bias input to a random neuron.
%%
%% Adds a bias connection (self-connection) to a neuron that
%% doesn't already have one.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_bias(term()) -> ok | {error, term()}.
add_bias(AgentId) ->
    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            {error, no_neurons};
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),

            %% Check if bias already exists
            HasBias = lists:any(
                fun({InputId, _}) -> InputId == bias end,
                Neuron#neuron.input_idps
            ),

            case HasBias of
                true ->
                    {error, already_has_bias};
                false ->
                    %% Add bias connection
                    BiasWeight = {rand:uniform() - 0.5, 0.0, 0.1, []},
                    NewInputIdps = [{bias, [BiasWeight]} | Neuron#neuron.input_idps],
                    UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
                    genotype:write(UpdatedNeuron),
                    ok
            end
    end.

%% @doc Add output link from a random neuron.
%%
%% Connects a neuron to another neuron or actuator that it's
%% not currently connected to.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_outlink(term()) -> ok | {error, term()}.
add_outlink(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            {error, no_neurons};
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),

            %% Find potential targets (neurons and actuators not already connected)
            AllTargets = Cortex#cortex.neuron_ids ++ Cortex#cortex.actuator_ids,
            CurrentOutputs = Neuron#neuron.output_ids,
            AvailableTargets = AllTargets -- CurrentOutputs -- [NeuronId],

            case AvailableTargets of
                [] ->
                    {error, no_available_targets};
                Targets ->
                    TargetId = selection_utils:random_select(Targets),
                    link_neuron_to_target(NeuronId, Neuron, TargetId),
                    ok
            end
    end.

%% @doc Add input link to a random neuron.
%%
%% Connects a sensor or another neuron to a neuron that it's
%% not currently connected to.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_inlink(term()) -> ok | {error, term()}.
add_inlink(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case select_random_neuron(AgentId) of
        {error, no_neurons} ->
            {error, no_neurons};
        NeuronId ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),

            %% Find potential sources (sensors and neurons not already connected)
            AllSources = Cortex#cortex.sensor_ids ++ Cortex#cortex.neuron_ids,
            CurrentInputIds = [InputId || {InputId, _} <- Neuron#neuron.input_idps],
            AvailableSources = AllSources -- CurrentInputIds -- [NeuronId],

            case AvailableSources of
                [] ->
                    {error, no_available_sources};
                Sources ->
                    SourceId = selection_utils:random_select(Sources),
                    link_source_to_neuron(SourceId, NeuronId, Neuron),
                    ok
            end
    end.

%% @doc Add a new neuron by splitting a connection.
%%
%% Selects a random connection, removes it, and inserts a new
%% neuron in the middle.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_neuron(term()) -> ok | {error, term()}.
add_neuron(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case find_splittable_link(AgentId) of
        {error, no_links} ->
            {error, cannot_add_neuron};
        {FromId, ToId, Weight} ->
            %% Create new neuron
            NewNeuronId = genotype:generate_id(neuron),
            Constraint = Agent#agent.constraint,
            AF = selection_utils:random_select(Constraint#constraint.neural_afs),
            AggrF = selection_utils:random_select(Constraint#constraint.neural_aggr_fs),

            %% Calculate layer coordinate (between from and to)
            %% Note: Layer coordinate not used in ID generation currently
            %% but preserved for future use
            _FromLayer = get_layer_coord(FromId),
            _ToLayer = get_layer_coord(ToId),

            %% Create neuron with connections
            NewNeuron = #neuron{
                id = NewNeuronId,
                generation = Agent#agent.generation,
                cx_id = Agent#agent.cx_id,
                af = AF,
                aggr_f = AggrF,
                input_idps = [{FromId, [Weight]}],
                output_ids = [ToId],
                ro_ids = []
            },

            %% Update source to point to new neuron instead of target
            update_source_output(FromId, ToId, NewNeuronId),

            %% Update target to receive from new neuron instead of source
            update_target_input(ToId, FromId, NewNeuronId, Weight),

            %% Write new neuron
            genotype:write(NewNeuron),

            %% Update cortex
            NewNeuronIds = [NewNeuronId | Cortex#cortex.neuron_ids],
            UpdatedCortex = Cortex#cortex{neuron_ids = NewNeuronIds},
            genotype:write(UpdatedCortex),

            ok
    end.

%% @doc Add neuron by outsplicing (split output connection).
%%
%% Similar to add_neuron but specifically targets output connections.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec outsplice(term()) -> ok | {error, term()}.
outsplice(AgentId) ->
    %% For now, delegate to add_neuron
    add_neuron(AgentId).

%% @doc Add link from a sensor to a neuron.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_sensorlink(term()) -> ok | {error, term()}.
add_sensorlink(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case Cortex#cortex.sensor_ids of
        [] ->
            {error, no_sensors};
        SensorIds ->
            SensorId = selection_utils:random_select(SensorIds),
            Sensor = genotype:dirty_read({sensor, SensorId}),

            %% Find neurons not connected to this sensor
            AvailableNeurons = Cortex#cortex.neuron_ids -- Sensor#sensor.fanout_ids,

            case AvailableNeurons of
                [] ->
                    {error, no_available_neurons};
                Neurons ->
                    NeuronId = selection_utils:random_select(Neurons),
                    link_sensor_to_neuron(SensorId, Sensor, NeuronId),
                    ok
            end
    end.

%% @doc Add link from a neuron to an actuator.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_actuatorlink(term()) -> ok | {error, term()}.
add_actuatorlink(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case Cortex#cortex.actuator_ids of
        [] ->
            {error, no_actuators};
        ActuatorIds ->
            ActuatorId = selection_utils:random_select(ActuatorIds),
            Actuator = genotype:dirty_read({actuator, ActuatorId}),

            %% Find neurons not connected to this actuator
            AvailableNeurons = Cortex#cortex.neuron_ids -- Actuator#actuator.fanin_ids,

            case AvailableNeurons of
                [] ->
                    {error, no_available_neurons};
                Neurons ->
                    NeuronId = selection_utils:random_select(Neurons),
                    link_neuron_to_actuator(NeuronId, ActuatorId, Actuator),
                    ok
            end
    end.

%% @doc Add a new sensor to the network.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_sensor(term()) -> ok | {error, term()}.
add_sensor(_AgentId) ->
    %% TODO: Implement sensor addition
    {error, not_implemented}.

%% @doc Add a new actuator to the network.
%%
%% @param AgentId the agent to mutate
%% @returns ok or {error, term()}
-spec add_actuator(term()) -> ok | {error, term()}.
add_actuator(_AgentId) ->
    %% TODO: Implement actuator addition
    {error, not_implemented}.

%% ============================================================================
%% Helper Functions
%% ============================================================================

%% @doc Select a random neuron from the agent's network.
%%
%% @param AgentId the agent
%% @returns NeuronId or {error, no_neurons}
-spec select_random_neuron(term()) -> term() | {error, no_neurons}.
select_random_neuron(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    case Cortex#cortex.neuron_ids of
        [] -> {error, no_neurons};
        NeuronIds -> selection_utils:random_select(NeuronIds)
    end.

%% @private Find a link that can be split to insert a neuron.
-spec find_splittable_link(term()) -> {term(), term(), {float(), float(), float(), list()}} | {error, no_links}.
find_splittable_link(AgentId) ->
    Agent = genotype:dirty_read({agent, AgentId}),
    Cortex = genotype:dirty_read({cortex, Agent#agent.cx_id}),

    %% Collect all links from neurons
    Links = lists:flatmap(
        fun(NeuronId) ->
            Neuron = genotype:dirty_read({neuron, NeuronId}),
            [{NeuronId, OutputId} || OutputId <- Neuron#neuron.output_ids]
        end,
        Cortex#cortex.neuron_ids
    ),

    case Links of
        [] ->
            {error, no_links};
        _ ->
            {FromId, ToId} = selection_utils:random_select(Links),
            %% Get the weight from the target's input
            Weight = get_link_weight(FromId, ToId),
            {FromId, ToId, Weight}
    end.

%% @private Get weight of a link
-spec get_link_weight(term(), term()) -> {float(), float(), float(), list()}.
get_link_weight(FromId, ToId) ->
    %% Try to get from target neuron
    case genotype:dirty_read({neuron, ToId}) of
        undefined ->
            %% Target is actuator - create new weight
            {rand:uniform() - 0.5, 0.0, 0.1, []};
        Neuron ->
            case lists:keyfind(FromId, 1, Neuron#neuron.input_idps) of
                {FromId, [Weight | _]} -> Weight;
                _ -> {rand:uniform() - 0.5, 0.0, 0.1, []}
            end
    end.

%% @private Get layer coordinate from element ID
-spec get_layer_coord(term()) -> float().
get_layer_coord({{Layer, _}, _Type}) -> Layer;
get_layer_coord(_) -> 0.5.

%% @private Update source element to output to new target
-spec update_source_output(term(), term(), term()) -> ok.
update_source_output(FromId, OldToId, NewToId) ->
    case genotype:dirty_read({neuron, FromId}) of
        undefined ->
            %% From is sensor
            Sensor = genotype:dirty_read({sensor, FromId}),
            NewFanoutIds = [NewToId | (Sensor#sensor.fanout_ids -- [OldToId])],
            UpdatedSensor = Sensor#sensor{fanout_ids = NewFanoutIds},
            genotype:write(UpdatedSensor);
        Neuron ->
            NewOutputIds = [NewToId | (Neuron#neuron.output_ids -- [OldToId])],
            UpdatedNeuron = Neuron#neuron{output_ids = NewOutputIds},
            genotype:write(UpdatedNeuron)
    end,
    ok.

%% @private Update target element to receive from new source
-spec update_target_input(term(), term(), term(), {float(), float(), float(), list()}) -> ok.
update_target_input(ToId, OldFromId, NewFromId, Weight) ->
    case genotype:dirty_read({neuron, ToId}) of
        undefined ->
            %% To is actuator
            Actuator = genotype:dirty_read({actuator, ToId}),
            NewFaninIds = [NewFromId | (Actuator#actuator.fanin_ids -- [OldFromId])],
            UpdatedActuator = Actuator#actuator{fanin_ids = NewFaninIds},
            genotype:write(UpdatedActuator);
        Neuron ->
            %% Remove old input, add new
            FilteredInputs = [{Id, W} || {Id, W} <- Neuron#neuron.input_idps, Id /= OldFromId],
            NewInputIdps = [{NewFromId, [Weight]} | FilteredInputs],
            UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
            genotype:write(UpdatedNeuron)
    end,
    ok.

%% @private Link a neuron to a target (neuron or actuator)
-spec link_neuron_to_target(term(), #neuron{}, term()) -> ok.
link_neuron_to_target(NeuronId, Neuron, TargetId) ->
    %% Update source neuron's outputs
    NewOutputIds = [TargetId | Neuron#neuron.output_ids],
    UpdatedNeuron = Neuron#neuron{output_ids = NewOutputIds},
    genotype:write(UpdatedNeuron),

    %% Update target's inputs
    case genotype:dirty_read({neuron, TargetId}) of
        undefined ->
            %% Target is actuator
            Actuator = genotype:dirty_read({actuator, TargetId}),
            NewFaninIds = [NeuronId | Actuator#actuator.fanin_ids],
            UpdatedActuator = Actuator#actuator{fanin_ids = NewFaninIds},
            genotype:write(UpdatedActuator);
        TargetNeuron ->
            NewWeight = {rand:uniform() - 0.5, 0.0, 0.1, []},
            NewInputIdps = [{NeuronId, [NewWeight]} | TargetNeuron#neuron.input_idps],
            UpdatedTarget = TargetNeuron#neuron{input_idps = NewInputIdps},
            genotype:write(UpdatedTarget)
    end,
    ok.

%% @private Link a source (sensor or neuron) to a neuron
-spec link_source_to_neuron(term(), term(), #neuron{}) -> ok.
link_source_to_neuron(SourceId, NeuronId, Neuron) ->
    %% Update neuron's inputs
    NewWeight = {rand:uniform() - 0.5, 0.0, 0.1, []},
    NewInputIdps = [{SourceId, [NewWeight]} | Neuron#neuron.input_idps],
    UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
    genotype:write(UpdatedNeuron),

    %% Update source's outputs
    case genotype:dirty_read({neuron, SourceId}) of
        undefined ->
            %% Source is sensor
            Sensor = genotype:dirty_read({sensor, SourceId}),
            NewFanoutIds = [NeuronId | Sensor#sensor.fanout_ids],
            UpdatedSensor = Sensor#sensor{fanout_ids = NewFanoutIds},
            genotype:write(UpdatedSensor);
        SourceNeuron ->
            NewOutputIds = [NeuronId | SourceNeuron#neuron.output_ids],
            UpdatedSource = SourceNeuron#neuron{output_ids = NewOutputIds},
            genotype:write(UpdatedSource)
    end,
    ok.

%% @private Link a sensor to a neuron
-spec link_sensor_to_neuron(term(), #sensor{}, term()) -> ok.
link_sensor_to_neuron(SensorId, Sensor, NeuronId) ->
    %% Update sensor's fanout
    NewFanoutIds = [NeuronId | Sensor#sensor.fanout_ids],
    UpdatedSensor = Sensor#sensor{fanout_ids = NewFanoutIds},
    genotype:write(UpdatedSensor),

    %% Update neuron's inputs
    Neuron = genotype:dirty_read({neuron, NeuronId}),
    NewWeight = {rand:uniform() - 0.5, 0.0, 0.1, []},
    NewInputIdps = [{SensorId, [NewWeight]} | Neuron#neuron.input_idps],
    UpdatedNeuron = Neuron#neuron{input_idps = NewInputIdps},
    genotype:write(UpdatedNeuron),
    ok.

%% @private Link a neuron to an actuator
-spec link_neuron_to_actuator(term(), term(), #actuator{}) -> ok.
link_neuron_to_actuator(NeuronId, ActuatorId, Actuator) ->
    %% Update actuator's fanin
    NewFaninIds = [NeuronId | Actuator#actuator.fanin_ids],
    UpdatedActuator = Actuator#actuator{fanin_ids = NewFaninIds},
    genotype:write(UpdatedActuator),

    %% Update neuron's outputs
    Neuron = genotype:dirty_read({neuron, NeuronId}),
    NewOutputIds = [ActuatorId | Neuron#neuron.output_ids],
    UpdatedNeuron = Neuron#neuron{output_ids = NewOutputIds},
    genotype:write(UpdatedNeuron),
    ok.
