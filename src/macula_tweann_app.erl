%% @doc Application behaviour for macula_tweann.
%%
%% Starts the morphology registry on application startup.
%%
%% @author Macula.io
%% @copyright 2025 Macula.io, Apache-2.0
-module(macula_tweann_app).

-behaviour(application).

%% application callbacks
-export([start/2, stop/1]).

%%==============================================================================
%% application Callbacks
%%==============================================================================

%% @doc Start the application.
%%
%% Starts the morphology registry supervisor.
-spec start(StartType :: normal | {takeover, node()} | {failover, node()},
            StartArgs :: term()) ->
    {ok, pid()} | {error, Reason :: term()}.
start(_StartType, _StartArgs) ->
    macula_tweann_sup:start_link().

%% @doc Stop the application.
-spec stop(State :: term()) -> ok.
stop(_State) ->
    ok.
