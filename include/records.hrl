%% @doc Core data structure definitions for Macula TWEANN
%%
%% This file contains all record definitions for the TWEANN system.
%% Each record is fully documented with field-level comments explaining
%% purpose, format, and usage.
%%
%% == Record Organization ==
%% - Network components: sensor, neuron, actuator, cortex
%% - Evolution components: agent, specie, population
%% - Substrate: substrate (for HyperNEAT)
%% - Environment: polis, scape
%% - Configuration: constraint, pmp (population manager parameters)
%%
%% @copyright 2025 Macula.io
%% @license Apache-2.0

-ifndef(MACULA_TWEANN_RECORDS_HRL).
-define(MACULA_TWEANN_RECORDS_HRL, true).

-include("types.hrl").

%%==============================================================================
%% Configuration Defines
%%==============================================================================

%% @doc Enable behavioral tracing for debugging
-define(BEHAVIORAL_TRACE, false).

%% @doc Enable interactive selection mode
-define(INTERACTIVE_SELECTION, false).

%%==============================================================================
%% Network Component Records
%%==============================================================================

%% @doc Sensor record - Input interface to the neural network
%%
%% Sensors gather information from the environment (scape) and feed
%% signals to neurons in the network.
%%
%% Field Documentation:
%%   id          - Unique identifier: {{-1.0, UniqueFloat}, sensor}
%%                 The -1.0 layer coordinate indicates sensor layer
%%   name        - Atom naming the sensor type (e.g., rng, distance_scanner)
%%   type        - Sensor category (standard, etc.)
%%   cx_id       - Parent cortex identifier for ownership
%%   scape       - Environment spec: {private|public, ScapeName}
%%   vl          - Vector length: number of signals produced per sense
%%   fanout_ids  - List of neuron IDs that receive signals from this sensor
%%   generation  - Generation when this sensor was added to the network
%%   format      - Signal format: {no_geo|geo, [Resolution...]}
%%   parameters  - Sensor-specific configuration parameters
%%   gt_parameters - Genotype parameters (for substrate sensors)
%%   phys_rep    - Physical representation in the environment
%%   vis_rep     - Visual representation for rendering
%%   pre_f       - Pre-processing function name (atom)
%%   post_f      - Post-processing function name (atom)
-record(sensor, {
    id,
    name,
    type,
    cx_id,
    scape,
    vl,
    fanout_ids = [],
    generation,
    format,
    parameters,
    gt_parameters,
    phys_rep,
    vis_rep,
    pre_f,
    post_f
}).

%% @doc Actuator record - Output interface from the neural network
%%
%% Actuators receive signals from the network and perform actions
%% in the environment (scape).
%%
%% Field Documentation:
%%   id          - Unique identifier: {{1.0, UniqueFloat}, actuator}
%%                 The 1.0 layer coordinate indicates actuator layer
%%   name        - Atom naming the actuator type (e.g., pts, differential_drive)
%%   type        - Actuator category (standard, etc.)
%%   cx_id       - Parent cortex identifier for ownership
%%   scape       - Environment spec: {private|public, ScapeName}
%%   vl          - Vector length: number of input signals expected
%%   fanin_ids   - List of neuron IDs that send signals to this actuator
%%   generation  - Generation when this actuator was added
%%   format      - Signal format: {no_geo|geo, [Resolution...]}
%%   parameters  - Actuator-specific configuration parameters
%%   gt_parameters - Genotype parameters (for substrate actuators)
%%   phys_rep    - Physical representation in the environment
%%   vis_rep     - Visual representation for rendering
%%   pre_f       - Pre-processing function name (atom)
%%   post_f      - Post-processing function name (atom)
-record(actuator, {
    id,
    name,
    type,
    cx_id,
    scape,
    vl,
    fanin_ids = [],
    generation,
    format,
    parameters,
    gt_parameters,
    phys_rep,
    vis_rep,
    pre_f,
    post_f
}).

%% @doc Neuron record - Core computational unit of the network
%%
%% Neurons receive weighted inputs, aggregate them, apply an activation
%% function, and forward the output. They can also apply plasticity
%% rules to modify their weights during operation.
%%
%% Field Documentation:
%%   id                    - Unique identifier: {{LayerCoord, UniqueFloat}, neuron}
%%                           LayerCoord is between 0.0 and 1.0 for hidden layers
%%   generation            - Generation when this neuron was created
%%   cx_id                 - Parent cortex identifier for ownership
%%   pre_processor         - Pre-processing function for inputs (atom or undefined)
%%   signal_integrator     - How to combine multiple signals (atom or undefined)
%%   af                    - Activation function: tanh|cos|sin|gaussian|etc.
%%                           (Old name, should be activation_function)
%%   post_processor        - Post-processing function for output (atom or undefined)
%%   pf                    - Plasticity function: {FunctionName, Parameters}
%%                           (Old name, should be plasticity_function)
%%   aggr_f                - Aggregation function: dot_product|mult_product|diff
%%                           (Old name, should be aggregation_function)
%%   input_idps            - Weighted inputs list:
%%                           [{SourceId, [{Weight, DeltaWeight, LearningRate, ParamList}...]}...]
%%                           This is the main weight storage using weight_spec() format
%%                           (Old name, should be weighted_inputs)
%%   input_idps_modulation - Modulatory inputs for neuromodulation
%%                           (Old name, should be weighted_inputs_modulation)
%%   output_ids            - List of neuron/actuator IDs to send output to
%%   ro_ids                - Recurrent output IDs (feedback connections)
%%                           (Old name, should be recurrent_output_ids)
-record(neuron, {
    id,
    generation,
    cx_id,
    pre_processor,
    signal_integrator,
    af,                         %% activation_function
    post_processor,
    pf,                         %% plasticity_function
    aggr_f,                     %% aggregation_function
    input_idps = [],            %% weighted_inputs
    input_idps_modulation = [], %% weighted_inputs_modulation
    output_ids = [],
    ro_ids = []                 %% recurrent_output_ids
}).

%% @doc Cortex record - Network coordinator
%%
%% The cortex coordinates the sense-think-act cycle of the neural network.
%% It triggers sensors, waits for neurons to process, and collects actuator outputs.
%%
%% Field Documentation:
%%   id           - Unique identifier: {{0.0, UniqueFloat}, cortex}
%%   agent_id     - Owning agent identifier
%%   neuron_ids   - List of all neuron IDs in this network
%%   sensor_ids   - List of all sensor IDs in this network
%%   actuator_ids - List of all actuator IDs in this network
-record(cortex, {
    id,
    agent_id,
    neuron_ids = [],
    sensor_ids = [],
    actuator_ids = []
}).

%%==============================================================================
%% Substrate Records (HyperNEAT)
%%==============================================================================

%% @doc Substrate record - Geometric neural network substrate
%%
%% Used in HyperNEAT for evolving larger-scale networks through
%% compositional pattern producing networks (CPPNs).
%%
%% Field Documentation:
%%   id         - Unique substrate identifier
%%   agent_id   - Owning agent identifier
%%   densities  - Substrate resolution/density parameters
%%   linkform   - Link pattern: l2l_feedforward|jordan_recurrent|fully_connected
%%   plasticity - Substrate plasticity type: none|hebbian|ojas
%%   cpp_ids    - Compositional Pattern Producing neuron IDs
%%   cep_ids    - Coordinate Encoder Pattern neuron IDs
-record(substrate, {
    id,
    agent_id,
    densities,
    linkform,
    plasticity = none,
    cpp_ids = [],
    cep_ids = []
}).

%%==============================================================================
%% Evolution Records
%%==============================================================================

%% @doc Agent record - Individual in the evolutionary population
%%
%% An agent represents one candidate solution with its neural network,
%% evolutionary history, fitness scores, and evolutionary parameters.
%%
%% Field Documentation:
%%   id                        - Unique identifier: {UniqueFloat, agent}
%%   encoding_type             - Network type: neural|substrate
%%   generation                - Current generation number
%%   population_id             - Parent population identifier
%%   specie_id                 - Assigned specie identifier
%%   cx_id                     - Cortex (neural network) identifier
%%   fingerprint               - Structural signature for speciation
%%   constraint                - Evolution constraint record
%%   evo_hist                  - List of applied mutation operators
%%                               [{MutationName, ElementIds...}...]
%%   fitness                   - Current fitness score
%%   innovation_factor         - Measure of topological novelty
%%   pattern                   - Network topology: [{LayerCoord, NeuronIds}...]
%%   tuning_selection_f        - Weight selection for tuning: all|recent|dynamic_random
%%   annealing_parameter       - Simulated annealing temperature
%%   tuning_duration_f         - Duration function: {FuncName, Param}
%%   perturbation_range        - Weight perturbation magnitude
%%   mutation_operators        - Available mutations: [{Name, Probability}...]
%%   tot_topological_mutations_f - Mutation count function: {FuncName, Param}
%%   heredity_type             - Evolution type: darwinian|lamarckian
%%   substrate_id              - Substrate identifier (for HyperNEAT)
%%   offspring_ids             - List of child agent IDs
%%   parent_ids                - List of parent agent IDs
%%   champion_flag             - Whether this is a specie champion
%%   evolvability              - Measure of evolutionary potential
%%   brittleness               - Sensitivity to weight perturbation
%%   robustness                - Stability across perturbations
%%   evolutionary_capacitance  - Ability to accumulate beneficial mutations
%%   behavioral_trace          - Recorded behavior for analysis
%%   fs                        - Fitness scaling factor
%%   main_fitness              - Primary optimization target
-record(agent, {
    id,
    encoding_type,
    generation,
    population_id,
    specie_id,
    cx_id,
    fingerprint,
    constraint,
    evo_hist = [],
    fitness = 0,
    innovation_factor = 0,
    pattern = [],
    tuning_selection_f,
    annealing_parameter,
    tuning_duration_f,
    perturbation_range,
    mutation_operators,
    tot_topological_mutations_f,
    heredity_type,
    substrate_id,
    offspring_ids = [],
    parent_ids = [],
    champion_flag = [false],
    evolvability = 0,
    brittleness = 0,
    robustness = 0,
    evolutionary_capacitance = 0,
    behavioral_trace,
    fs = 1,
    main_fitness
}).

%% @doc Champion record - Hall of fame entry
%%
%% Records information about top-performing agents for the hall of fame.
%%
%% Field Documentation:
%%   hof_fingerprint       - Hall of fame categorization fingerprint
%%   id                    - Champion agent ID
%%   fitness               - Training fitness score
%%   validation_fitness    - Validation set fitness
%%   test_fitness          - Test set fitness
%%   main_fitness          - Primary fitness measure
%%   tot_n                 - Total number of neurons
%%   evolvability          - Evolutionary potential measure
%%   robustness            - Stability measure
%%   brittleness           - Sensitivity measure
%%   generation            - Generation when achieved
%%   behavioral_differences - Behavioral novelty scores
%%   fs                    - Fitness scaling factor
-record(champion, {
    hof_fingerprint,
    id,
    fitness,
    validation_fitness,
    test_fitness,
    main_fitness,
    tot_n,
    evolvability,
    robustness,
    brittleness,
    generation,
    behavioral_differences,
    fs
}).

%% @doc Specie record - Species for NEAT speciation
%%
%% Groups similar agents together for protected innovation.
%%
%% Field Documentation:
%%   id                    - Specie identifier
%%   population_id         - Parent population identifier
%%   fingerprint           - Representative structural fingerprint
%%   constraint            - Evolution constraint record
%%   all_agent_ids         - All agents ever in this specie
%%   agent_ids             - Currently active agents
%%   dead_pool             - Recently removed agents
%%   champion_ids          - Top performers in this specie
%%   fitness               - Specie-level fitness (usually average)
%%   innovation_factor     - Topological innovation measure: {Novelty, Count}
%%   stats                 - Historical statistics
%%   seed_agent_ids        - Initial seed agents
%%   hof_distinguishers    - Hall of fame categorization criteria
%%   specie_distinguishers - Speciation criteria
%%   hall_of_fame          - List of champion records
-record(specie, {
    id,
    population_id,
    fingerprint,
    constraint,
    all_agent_ids = [],
    agent_ids = [],
    dead_pool = [],
    champion_ids = [],
    fitness,
    innovation_factor = {0, 0},
    stats = [],
    seed_agent_ids = [],
    hof_distinguishers = [tot_n],
    specie_distinguishers = [tot_n],
    hall_of_fame = []
}).

%% @doc Trace record - Evolution statistics trace
%%
%% Tracks statistics across generations for analysis.
%%
%% Field Documentation:
%%   stats           - List of stat records
%%   tot_evaluations - Total fitness evaluations performed
%%   step_size       - Evaluation interval for recording stats
-record(trace, {
    stats = [],
    tot_evaluations = 0,
    step_size = 500
}).

%% @doc Population record - Top-level evolutionary container
%%
%% Contains all species and manages the overall evolutionary process.
%%
%% Field Documentation:
%%   id                       - Population identifier
%%   polis_id                 - Parent polis (world) identifier
%%   specie_ids               - List of specie identifiers
%%   morphologies             - Available network morphologies
%%   innovation_factor        - Population-level innovation measure
%%   evo_alg_f                - Evolution algorithm: generational|steady_state
%%   fitness_postprocessor_f  - Fitness adjustment: none|size_proportional
%%   selection_f              - Parent selection: competition|top3|hof_competition
%%   trace                    - Statistics trace record
%%   seed_agent_ids           - Initial population seeds
%%   seed_specie_ids          - Initial species seeds
-record(population, {
    id,
    polis_id,
    specie_ids = [],
    morphologies = [],
    innovation_factor,
    evo_alg_f,
    fitness_postprocessor_f,
    selection_f,
    trace = #trace{},
    seed_agent_ids = [],
    seed_specie_ids = []
}).

%% @doc Stat record - Generation statistics
%%
%% Records statistics for a single generation or evaluation point.
%%
%% Field Documentation:
%%   morphology         - Network morphology type
%%   specie_id          - Specie being measured
%%   avg_neurons        - Average neuron count
%%   std_neurons        - Standard deviation of neuron count
%%   avg_fitness        - Average fitness score
%%   std_fitness        - Standard deviation of fitness
%%   max_fitness        - Maximum fitness achieved
%%   min_fitness        - Minimum fitness in population
%%   validation_fitness - Fitness on validation set
%%   test_fitness       - Fitness on test set
%%   avg_diversity      - Average behavioral diversity
%%   evaluations        - Number of evaluations performed
%%   time_stamp         - When this stat was recorded
-record(stat, {
    morphology,
    specie_id,
    avg_neurons,
    std_neurons,
    avg_fitness,
    std_fitness,
    max_fitness,
    min_fitness,
    validation_fitness,
    test_fitness,
    avg_diversity,
    evaluations,
    time_stamp
}).

%%==============================================================================
%% Configuration Records
%%==============================================================================

%% @doc Constraint record - Evolution constraints
%%
%% Defines what evolutionary operations are allowed and what
%% network components can be used.
%%
%% (See original file for full default values)
-record(constraint, {
    morphology = xor_mimic,
    connection_architecture = recurrent,
    neural_afs = [tanh, cos, gaussian],
    neural_pfns = [none],
    substrate_plasticities = [none],
    substrate_linkforms = [l2l_feedforward],
    neural_aggr_fs = [dot_product],
    tuning_selection_fs = [dynamic_random],
    tuning_duration_f = {wsize_proportional, 0.5},
    annealing_parameters = [0.5],
    perturbation_ranges = [1],
    agent_encoding_types = [neural],
    heredity_types = [darwinian],
    mutation_operators = [
        {add_bias, 10},
        {add_outlink, 40},
        {add_inlink, 40},
        {add_neuron, 40},
        {outsplice, 40},
        {add_sensorlink, 1},
        {add_sensor, 1},
        {add_actuator, 1},
        {add_cpp, 1},
        {add_cep, 1}
    ],
    tot_topological_mutations_fs = [{ncount_exponential, 0.5}],
    population_evo_alg_f = generational,
    population_fitness_postprocessor_f = size_proportional,
    population_selection_f = hof_competition,
    specie_distinguishers = [tot_n],
    hof_distinguishers = [tot_n],
    objectives = [main_fitness, inverse_tot_n]
}).

%% @doc Experiment record - Experiment configuration
%%
%% Tracks an evolutionary experiment across multiple runs.
%%
%% Field Documentation:
%%   id               - Experiment identifier
%%   backup_flag      - Whether to backup population state
%%   pm_parameters    - Population manager parameters (pmp record)
%%   init_constraints - Initial constraint record
%%   progress_flag    - Status: in_progress|completed|interrupted
%%   trace_acc        - Accumulated traces from all runs
%%   run_index        - Current run number
%%   tot_runs         - Total runs to perform
%%   notes            - Experiment notes/description
%%   started          - Start timestamp: {date(), time()}
%%   completed        - Completion timestamp
%%   interruptions    - List of interruption events
-record(experiment, {
    id,
    backup_flag = true,
    pm_parameters,
    init_constraints,
    progress_flag = in_progress,
    trace_acc = [],
    run_index = 1,
    tot_runs = 10,
    notes,
    started = {date(), time()},
    completed,
    interruptions = []
}).

%% @doc PMP record - Population Manager Parameters
%%
%% Configuration for the population manager process.
%%
%% Field Documentation:
%%   op_mode              - Operation mode: gt (ground truth) |test
%%   population_id        - Population to manage
%%   survival_percentage  - Fraction of population that survives selection
%%   specie_size_limit    - Maximum agents per specie
%%   init_specie_size     - Initial specie population
%%   polis_id             - Parent polis identifier
%%   generation_limit     - Maximum generations to evolve
%%   evaluations_limit    - Maximum fitness evaluations
%%   fitness_goal         - Target fitness (inf for no limit)
%%   benchmarker_pid      - Process for benchmark reporting
%%   committee_pid        - Process for ensemble decisions
-record(pmp, {
    op_mode = gt,
    population_id = test,
    survival_percentage = 0.5,
    specie_size_limit = 10,
    init_specie_size = 20,
    polis_id = mathema,
    generation_limit = 100,
    evaluations_limit = 100000,
    fitness_goal = inf,
    benchmarker_pid,
    committee_pid
}).

%%==============================================================================
%% Environment/Scape Records
%%==============================================================================

%% @doc Polis record - World container
%%
%% A polis is a world that contains populations and scapes.
%%
%% Field Documentation:
%%   id             - Polis identifier
%%   scape_ids      - List of environment identifiers
%%   population_ids - List of population identifiers
%%   specie_ids     - List of specie identifiers
%%   dx_ids         - List of agent identifiers
%%   parameters     - Polis configuration parameters
-record(polis, {
    id,
    scape_ids = [],
    population_ids = [],
    specie_ids = [],
    dx_ids = [],
    parameters = []
}).

%% @doc Scape record - Environment/simulation
%%
%% A scape is the environment where agents are evaluated.
%%
%% Field Documentation:
%%   id              - Scape identifier
%%   type            - Scape type/category
%%   physics         - Physics simulation parameters
%%   metabolics      - Energy/metabolism rules
%%   sector2avatars  - Mapping of sectors to avatars
%%   avatars         - List of avatar entities
%%   plants          - List of plant entities
%%   walls           - List of wall obstacles
%%   pillars         - List of pillar obstacles
%%   laws            - Environmental rules
%%   anomolies       - Special environmental effects
%%   artifacts       - Interactive objects
%%   objects         - Generic objects
%%   elements        - Environmental elements
%%   atoms           - Basic particles
%%   scheduler       - Scheduling counter
-record(scape, {
    id,
    type,
    physics,
    metabolics,
    sector2avatars,
    avatars = [],
    plants = [],
    walls = [],
    pillars = [],
    laws = [],
    anomolies = [],
    artifacts = [],
    objects = [],
    elements = [],
    atoms = [],
    scheduler = 0
}).

%% @doc Avatar record - Agent's physical presence in scape
-record(avatar, {
    id,
    sector,
    morphology,
    type,
    specie,
    energy = 0,
    health = 0,
    food = 0,
    age = 0,
    kills = 0,
    loc,
    direction,
    r,
    mass,
    objects,
    vis = [],
    state,
    stats,
    actuators,
    sensors,
    sound,
    gestalt,
    spear
}).

%%==============================================================================
%% Topology Summary Records
%%==============================================================================

%% @doc Topology summary for fingerprinting
-record(topology_summary, {
    type,
    tot_neurons,
    tot_n_ils,       %% total input links
    tot_n_ols,       %% total output links
    tot_n_ros,       %% total recurrent outputs
    af_distribution  %% activation function distribution
}).

%% @doc Signature record for speciation
-record(signature, {
    generalized_Pattern,
    generalized_EvoHist,
    generalized_Sensors,
    generalized_Actuators,
    topology_summary
}).

%%==============================================================================
%% Circuit Records (Deep Learning Components)
%%==============================================================================

%% Note: Circuit, layer, and neurode records are used for
%% deep learning components. See original DXNN2 documentation.

-record(circuit, {
    id,
    i,                       %% input_idps
    ovl,                     %% output vector length
    ivl,                     %% input vector length
    training,
    output,
    parameters,
    dynamics,
    layers,
    type = standard,
    noise,
    noise_type = zero_mask,
    lp_decay = 0.999999,
    lp_min = 0.0000001,
    lp_max = 0.1,
    memory = [],
    memory_size = {0, 100000},
    validation,
    testing,
    receptive_field = full,
    step = 0,
    block_size = 100,
    err_acc = 0,
    backprop_tuning = off,
    training_length = 1000
}).

-record(layer, {
    id,
    type,
    noise,
    neurode_type = tanh,
    dynamics = dynamic,
    neurodes = [],
    tot_neurodes,
    input,
    output,
    ivl,
    encoder = [],
    decoder = [],
    backprop_tuning = off,
    index_start,
    index_end,
    parameters = []
}).

-record(neurode, {
    id,
    weights,
    i,
    af,
    bias,
    parameters = [],
    dot_product
}).

-endif. %% MACULA_TWEANN_RECORDS_HRL
