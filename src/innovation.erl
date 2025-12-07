%% @doc Innovation number tracking for NEAT-style evolution.
%%
%% This module manages innovation numbers that uniquely identify structural
%% changes in neural networks. Innovation numbers enable:
%% - Meaningful crossover between networks with different topologies
%% - Historical alignment of genes during reproduction
%% - Tracking which structural changes are the "same" across lineages
%%
%% Key concepts from NEAT (Stanley and Miikkulainen, 2002):
%% - Each new link gets a unique innovation number
%% - When a link is split to add a neuron, the new node and its links get tracked
%% - Same structural change (same from/to) always gets the same innovation
%%
%% @author Macula.io
%% @copyright 2025 Macula.io, Apache-2.0
-module(innovation).

-include_lib("stdlib/include/ms_transform.hrl").

%% API
-export([
    init/0,
    reset/0,
    next_innovation/0,
    get_or_create_link_innovation/2,
    get_or_create_node_innovation/2,
    get_innovation_info/1,
    get_link_innovation/2,
    get_node_innovation/2
]).

%% Internal records for Mnesia storage
-record(innovation_counter, {
    id = counter :: counter,
    value = 0 :: non_neg_integer()
}).

-record(link_innovation, {
    key :: {term(), term()},         % {FromId, ToId}
    innovation :: pos_integer()      % Unique innovation number
}).

-record(node_innovation, {
    key :: {term(), term()},         % {FromId, ToId} - the link being split
    node_innovation :: pos_integer(),    % Innovation for the new node
    in_innovation :: pos_integer(),      % Innovation for link to new node
    out_innovation :: pos_integer()      % Innovation for link from new node
}).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Initialize innovation tables in Mnesia.
%%
%% Creates the required tables for innovation tracking.
%% Should be called after genotype:init_db().
-spec init() -> ok.
init() ->
    Tables = [
        {innovation_counter, record_info(fields, innovation_counter), set},
        {link_innovation, record_info(fields, link_innovation), set},
        {node_innovation, record_info(fields, node_innovation), set}
    ],

    lists:foreach(
        fun({Name, Fields, Type}) ->
            case mnesia:create_table(Name, [
                {attributes, Fields},
                {type, Type},
                {ram_copies, [node()]}
            ]) of
                {atomic, ok} -> ok;
                {aborted, {already_exists, Name}} -> ok
            end
        end,
        Tables
    ),

    %% Wait for tables
    ok = mnesia:wait_for_tables([innovation_counter, link_innovation, node_innovation], 5000),

    %% Initialize counter if not exists
    case mnesia:dirty_read(innovation_counter, counter) of
        [] ->
            mnesia:dirty_write(#innovation_counter{id = counter, value = 0});
        _ ->
            ok
    end,
    ok.

%% @doc Reset all innovation tracking.
%%
%% Clears all innovation tables and resets the counter.
%% Useful for starting a fresh evolutionary run.
-spec reset() -> ok.
reset() ->
    lists:foreach(
        fun(Table) ->
            case mnesia:clear_table(Table) of
                {atomic, ok} -> ok;
                {aborted, {no_exists, Table}} -> ok
            end
        end,
        [innovation_counter, link_innovation, node_innovation]
    ),
    %% Reinitialize counter
    mnesia:dirty_write(#innovation_counter{id = counter, value = 0}),
    ok.

%% @doc Get the next innovation number.
%%
%% Atomically increments and returns the global innovation counter.
-spec next_innovation() -> pos_integer().
next_innovation() ->
    mnesia:dirty_update_counter(innovation_counter, counter, 1).

%% @doc Get or create innovation number for a link.
%%
%% If this exact link (from -> to) was seen before, returns the same
%% innovation number. Otherwise, creates a new one.
%% This ensures that the same structural change in different lineages
%% gets the same historical marker.
-spec get_or_create_link_innovation(FromId :: term(), ToId :: term()) -> pos_integer().
get_or_create_link_innovation(FromId, ToId) ->
    Key = {FromId, ToId},
    case mnesia:dirty_read(link_innovation, Key) of
        [#link_innovation{innovation = Inn}] ->
            Inn;
        [] ->
            Inn = next_innovation(),
            mnesia:dirty_write(#link_innovation{key = Key, innovation = Inn}),
            Inn
    end.

%% @doc Get or create innovation numbers for a node split.
%%
%% When a link is split to add a neuron, we need three innovation numbers:
%% 1. For the new node itself
%% 2. For the new link from the original source to the new node
%% 3. For the new link from the new node to the original target
%%
%% Returns {NodeInnovation, InLinkInnovation, OutLinkInnovation}
-spec get_or_create_node_innovation(FromId :: term(), ToId :: term()) ->
    {pos_integer(), pos_integer(), pos_integer()}.
get_or_create_node_innovation(FromId, ToId) ->
    Key = {FromId, ToId},
    case mnesia:dirty_read(node_innovation, Key) of
        [#node_innovation{node_innovation = NodeInn,
                          in_innovation = InInn,
                          out_innovation = OutInn}] ->
            {NodeInn, InInn, OutInn};
        [] ->
            NodeInn = next_innovation(),
            InInn = next_innovation(),
            OutInn = next_innovation(),
            mnesia:dirty_write(#node_innovation{
                key = Key,
                node_innovation = NodeInn,
                in_innovation = InInn,
                out_innovation = OutInn
            }),
            {NodeInn, InInn, OutInn}
    end.

%% @doc Get innovation info for a specific innovation number.
%%
%% Returns {link, FromId, ToId} or {node, FromId, ToId, InInn, OutInn}
%% or not_found if the innovation doesn't exist.
-spec get_innovation_info(pos_integer()) ->
    {link, term(), term()} |
    {node, term(), term(), pos_integer(), pos_integer()} |
    not_found.
get_innovation_info(InnovationNum) ->
    %% Check link innovations
    LinkMatch = ets:fun2ms(
        fun(#link_innovation{key = Key, innovation = Inn})
            when Inn =:= InnovationNum -> Key
        end
    ),
    case mnesia:dirty_select(link_innovation, LinkMatch) of
        [{FromId, ToId}] ->
            {link, FromId, ToId};
        [] ->
            %% Check node innovations
            NodeMatch = ets:fun2ms(
                fun(#node_innovation{key = Key,
                                     node_innovation = NodeInn,
                                     in_innovation = InInn,
                                     out_innovation = OutInn})
                    when NodeInn =:= InnovationNum -> {Key, InInn, OutInn}
                end
            ),
            case mnesia:dirty_select(node_innovation, NodeMatch) of
                [{{FromId, ToId}, InInn, OutInn}] ->
                    {node, FromId, ToId, InInn, OutInn};
                [] ->
                    not_found
            end
    end.

%% @doc Get existing link innovation without creating one.
%%
%% Returns the innovation number if the link exists, undefined otherwise.
-spec get_link_innovation(FromId :: term(), ToId :: term()) -> pos_integer() | undefined.
get_link_innovation(FromId, ToId) ->
    case mnesia:dirty_read(link_innovation, {FromId, ToId}) of
        [#link_innovation{innovation = Inn}] -> Inn;
        [] -> undefined
    end.

%% @doc Get existing node innovation without creating one.
%%
%% Returns {NodeInn, InInn, OutInn} if the node split exists, undefined otherwise.
-spec get_node_innovation(FromId :: term(), ToId :: term()) ->
    {pos_integer(), pos_integer(), pos_integer()} | undefined.
get_node_innovation(FromId, ToId) ->
    case mnesia:dirty_read(node_innovation, {FromId, ToId}) of
        [#node_innovation{node_innovation = NodeInn,
                          in_innovation = InInn,
                          out_innovation = OutInn}] ->
            {NodeInn, InInn, OutInn};
        [] ->
            undefined
    end.
