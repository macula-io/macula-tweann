%% @doc Unit tests for genome_mutator module.
-module(genome_mutator_tests).

-include_lib("eunit/include/eunit.hrl").
-include("records.hrl").

%% ============================================================================
%% Module Export Tests
%% ============================================================================

genome_mutator_exports_test() ->
    Exports = genome_mutator:module_info(exports),
    ?assert(lists:member({mutate, 1}, Exports)),
    ?assert(lists:member({mutate, 2}, Exports)),
    ?assert(lists:member({add_neuron, 1}, Exports)),
    ?assert(lists:member({add_bias, 1}, Exports)),
    ?assert(lists:member({mutate_weights, 1}, Exports)),
    ?assert(lists:member({select_random_neuron, 1}, Exports)).

%% ============================================================================
%% Integration Tests with Mnesia
%% ============================================================================

%% Helper to setup each test
setup_test() ->
    application:ensure_all_started(macula_tweann),
    test_helper:register_all_example_morphologies(),
    genotype:init_db().

calculate_mutation_count_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Count = genome_mutator:calculate_mutation_count(AgentId),
        ?assert(Count >= 1)
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

select_random_neuron_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        NeuronId = genome_mutator:select_random_neuron(AgentId),
        ?assert(NeuronId /= {error, no_neurons}),

        Neuron = genotype:dirty_read({neuron, NeuronId}),
        ?assert(is_record(Neuron, neuron))
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

add_bias_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:add_bias(AgentId),
        case Result of
            ok -> ok;
            {error, already_has_bias} -> ok;
            {error, no_neurons} -> ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

mutate_weights_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:mutate_weights(AgentId),
        ?assertEqual(ok, Result)
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

mutate_af_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:mutate_af(AgentId),
        case Result of
            ok -> ok;
            {error, no_alternatives} -> ok;
            {error, no_neurons} -> ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

mutate_tuning_selection_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:mutate_tuning_selection(AgentId),
        case Result of
            ok -> ok;
            {error, no_alternatives} -> ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

add_outlink_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:add_outlink(AgentId),
        case Result of
            ok -> ok;
            {error, no_available_targets} -> ok;
            {error, no_neurons} -> ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

add_inlink_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:add_inlink(AgentId),
        case Result of
            ok -> ok;
            {error, no_available_sources} -> ok;
            {error, no_neurons} -> ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

add_neuron_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Agent1 = genotype:dirty_read({agent, AgentId}),
        Cortex1 = genotype:dirty_read({cortex, Agent1#agent.cx_id}),
        InitialCount = length(Cortex1#cortex.neuron_ids),

        Result = genome_mutator:add_neuron(AgentId),
        case Result of
            ok ->
                Cortex2 = genotype:dirty_read({cortex, Agent1#agent.cx_id}),
                NewCount = length(Cortex2#cortex.neuron_ids),
                ?assertEqual(InitialCount + 1, NewCount);
            {error, cannot_add_neuron} ->
                ok
        end
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

mutate_single_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:mutate(AgentId, 1),
        ?assertEqual(ok, Result)
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.

mutate_multiple_test() ->
    setup_test(),
    try
        SpecieId = test_specie,
        AgentId = genotype:generate_id(agent),
        Constraint = #constraint{morphology = xor_mimic},
        genotype:construct_Agent(SpecieId, AgentId, Constraint),

        Result = genome_mutator:mutate(AgentId, 3),
        ?assertEqual(ok, Result)
    after
        genotype:reset_db(),
        application:stop(mnesia)
    end.
