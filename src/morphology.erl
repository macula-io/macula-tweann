%% @doc Morphology module for sensor/actuator specifications.
%%
%% This module defines the available sensors and actuators for each morphology
%% (problem domain). A morphology specifies what I/O interface the neural network
%% will use to interact with its environment.
%%
%% Based on DXNN2 by Gene Sher ("Handbook of Neuroevolution through Erlang").
%%
%% == Morphology Pattern ==
%%
%% Each morphology is implemented as a function clause that returns either
%% sensors or actuators depending on the argument:
%%
%% ```
%% morphology_name(sensors) -> [#sensor{...}];
%% morphology_name(actuators) -> [#actuator{...}].
%% ```
%%
%% == Adding New Morphologies ==
%%
%% 1. Add export for the new morphology function
%% 2. Implement `morphology_name(sensors)` returning sensor list
%% 3. Implement `morphology_name(actuators)` returning actuator list
%% 4. Sensors need: name, type, scape, vl (vector length)
%% 5. Actuators need: name, type, scape, vl
%%
%% @author Macula.io
%% @copyright 2025 Macula.io, Apache-2.0
-module(morphology).

-include("records.hrl").

%% Suppress supertype warnings - specs are intentionally general for polymorphic returns
-dialyzer({nowarn_function, [
    xor_mimic/1,
    pole_balancing/1,
    discrete_tmaze/1,
    prey/1,
    predator/1,
    forex_trader/1
]}).

%% API
-export([
    get_InitSensors/1,
    get_InitActuators/1,
    get_Sensors/1,
    get_Actuators/1
]).

%% Morphology definitions
-export([
    xor_mimic/1,
    pole_balancing/1,
    discrete_tmaze/1,
    prey/1,
    predator/1,
    forex_trader/1
]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Get initial sensors for a morphology.
%%
%% Returns the first sensor from the morphology's sensor list.
%% Used when constructing a new agent to get the default sensor.
%%
%% @param Morphology The morphology name (atom) or {Module, Function} tuple
%% @returns List containing the first sensor record
-spec get_InitSensors(atom() | {module(), atom()}) -> [#sensor{}].
get_InitSensors({M, F}) ->
    Sensors = M:F(sensors),
    [hd(Sensors)];
get_InitSensors(Morphology) ->
    Sensors = ?MODULE:Morphology(sensors),
    [hd(Sensors)].

%% @doc Get initial actuators for a morphology.
%%
%% Returns the first actuator from the morphology's actuator list.
%% Used when constructing a new agent to get the default actuator.
%%
%% @param Morphology The morphology name (atom) or {Module, Function} tuple
%% @returns List containing the first actuator record
-spec get_InitActuators(atom() | {module(), atom()}) -> [#actuator{}].
get_InitActuators({M, F}) ->
    Actuators = M:F(actuators),
    [hd(Actuators)];
get_InitActuators(Morphology) ->
    Actuators = ?MODULE:Morphology(actuators),
    [hd(Actuators)].

%% @doc Get all sensors for a morphology.
%%
%% @param Morphology The morphology name (atom) or {Module, Function} tuple
%% @returns List of all sensor records for this morphology
-spec get_Sensors(atom() | {module(), atom()}) -> [#sensor{}].
get_Sensors({M, F}) ->
    M:F(sensors);
get_Sensors(Morphology) ->
    ?MODULE:Morphology(sensors).

%% @doc Get all actuators for a morphology.
%%
%% @param Morphology The morphology name (atom) or {Module, Function} tuple
%% @returns List of all actuator records for this morphology
-spec get_Actuators(atom() | {module(), atom()}) -> [#actuator{}].
get_Actuators({M, F}) ->
    M:F(actuators);
get_Actuators(Morphology) ->
    ?MODULE:Morphology(actuators).

%%==============================================================================
%% Morphology Definitions
%%==============================================================================

%% @doc XOR mimic morphology for testing.
%%
%% Simple morphology that interfaces with an XOR simulation.
%% The sensor receives 2 inputs (the XOR inputs).
%% The actuator outputs 1 value (the XOR result).
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec xor_mimic(sensors | actuators) -> [#sensor{}] | [#actuator{}].
xor_mimic(sensors) ->
    [
        #sensor{
            name = xor_GetInput,
            type = standard,
            scape = {private, xor_sim},
            vl = 2
        }
    ];
xor_mimic(actuators) ->
    [
        #actuator{
            name = xor_SendOutput,
            type = standard,
            scape = {private, xor_sim},
            vl = 1
        }
    ].

%% @doc Pole balancing morphology.
%%
%% Classic reinforcement learning benchmark where the agent balances
%% one or more poles on a cart.
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec pole_balancing(sensors | actuators) -> [#sensor{}] | [#actuator{}].
pole_balancing(sensors) ->
    [
        #sensor{
            name = pb_GetInput,
            type = standard,
            scape = {private, pb_sim},
            vl = 3,
            parameters = [3]
        }
    ];
pole_balancing(actuators) ->
    [
        #actuator{
            name = pb_SendOutput,
            type = standard,
            scape = {private, pb_sim},
            vl = 1,
            parameters = [with_damping, 1]
        }
    ].

%% @doc Discrete T-Maze morphology.
%%
%% Navigation task where agent must learn to navigate a T-shaped maze.
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec discrete_tmaze(sensors | actuators) -> [#sensor{}] | [#actuator{}].
discrete_tmaze(sensors) ->
    [
        #sensor{
            name = dtm_GetInput,
            type = standard,
            scape = {private, dtm_sim},
            vl = 4,
            parameters = [all]
        }
    ];
discrete_tmaze(actuators) ->
    [
        #actuator{
            name = dtm_SendOutput,
            type = standard,
            scape = {private, dtm_sim},
            vl = 1,
            parameters = []
        }
    ].

%% @doc Prey morphology for predator-prey simulation.
%%
%% Flatland agent that can sense distance and move with two wheels.
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec prey(sensors | actuators) -> [#sensor{}] | [#actuator{}].
prey(sensors) ->
    Pi = math:pi(),
    Spread = Pi / 2,
    Density = 5,
    ROffset = 0,
    [
        #sensor{
            name = distance_scanner,
            type = standard,
            scape = {public, flatland},
            format = no_geo,
            vl = Density,
            parameters = [Spread, Density, ROffset]
        }
    ];
prey(actuators) ->
    [
        #actuator{
            name = two_wheels,
            type = standard,
            scape = {public, flatland},
            format = no_geo,
            vl = 2,
            parameters = [2]
        }
    ].

%% @doc Predator morphology for predator-prey simulation.
%%
%% Same as prey morphology (predator and prey have same I/O interface).
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec predator(sensors | actuators) -> [#sensor{}] | [#actuator{}].
predator(sensors) ->
    prey(sensors);
predator(actuators) ->
    prey(actuators).

%% @doc Forex trader morphology.
%%
%% Financial trading agent that reads price indicators and makes trades.
%%
%% @param Type Either `sensors` or `actuators`
%% @returns List of sensor or actuator records
-spec forex_trader(sensors | actuators) -> [#sensor{}] | [#actuator{}].
forex_trader(sensors) ->
    HRes = 100,
    [
        #sensor{
            name = fx_PLI,
            type = standard,
            scape = {private, fx_sim},
            format = no_geo,
            vl = HRes,
            parameters = [HRes, close]
        }
    ];
forex_trader(actuators) ->
    [
        #actuator{
            name = fx_Trade,
            type = standard,
            scape = {private, fx_sim},
            format = no_geo,
            vl = 1,
            parameters = []
        }
    ].
