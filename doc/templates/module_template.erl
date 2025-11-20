%% @module module_name
%% @doc Brief description of the module (one line)
%%
%% Detailed description of module purpose and architecture.
%% Explain what this module does and how it fits into the system.
%%
%% == Architecture ==
%% How this module fits in the system. What components it interacts
%% with and what role it plays in the overall TWEANN architecture.
%%
%% == Process Model ==
%% If this module implements a process (gen_server, etc.), describe
%% the process lifecycle and state management.
%%
%% == Usage ==
%% Example usage patterns:
%% ```
%% %% Create and initialize
%% Result = module_name:init(Args),
%%
%% %% Perform operation
%% Output = module_name:operation(Input).
%% ```
%%
%% == Implementation Notes ==
%% Important implementation details that future developers should know.
%% Include algorithm descriptions, performance considerations, and
%% design decisions.
%%
%% == Weight Tuple Format ==
%% If this module handles weights, document the weight_spec format:
%%   {Weight, DeltaWeight, LearningRate, ParameterList}
%%
%% @copyright 2025 Macula.io
%% @license Apache-2.0

-module(module_name).

%% Include type definitions
-include("types.hrl").
-include("records.hrl").

%% API exports
-export([
    public_function/1,
    public_function/2
]).

%% Internal exports (for spawn, etc.)
-export([
    internal_loop/1
]).

%%==============================================================================
%% Type Definitions
%%==============================================================================

%% @doc Description of this type
-type my_type() :: atom() | {complex, term()}.

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Brief description of function
%%
%% Detailed description of what the function does.
%%
%% @param Arg1 Description of first argument
%% @returns Description of return value
%% @throws {error, reason} when condition
%%
%% Example:
%% ```
%% Result = module_name:public_function(Arg1).
%% ```
-spec public_function(Arg1) -> Result when
      Arg1 :: term(),
      Result :: ok | {error, term()}.
public_function(Arg1) ->
    %% Implementation
    ok.

%% @doc Brief description of function
%%
%% @param Arg1 Description of first argument
%% @param Arg2 Description of second argument
%% @returns Description of return value
-spec public_function(Arg1, Arg2) -> Result when
      Arg1 :: term(),
      Arg2 :: term(),
      Result :: ok | {error, term()}.
public_function(Arg1, Arg2) ->
    %% Implementation
    ok.

%%==============================================================================
%% Internal Functions
%%==============================================================================

%% @private
%% @doc Internal helper function description
-spec internal_helper(Input) -> Output when
      Input :: term(),
      Output :: term().
internal_helper(Input) ->
    %% Implementation
    Input.

%% @private
%% @doc Process loop for gen_server-style modules
-spec internal_loop(State) -> no_return() when
      State :: term().
internal_loop(State) ->
    receive
        {message, Data} ->
            NewState = handle_message(Data, State),
            internal_loop(NewState);
        terminate ->
            ok
    end.

%% @private
handle_message(Data, State) ->
    %% Process message
    State.
