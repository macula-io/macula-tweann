# Function Documentation Template

Use this template when documenting functions in Macula TWEANN.

## EDoc Format

```erlang
%% @doc Brief description of function (one line)
%%
%% Detailed description of what the function does.
%% Include algorithm explanation if complex.
%%
%% @param ParamName Description of parameter
%% @param ParamName2 Description of second parameter
%% @returns Description of return value
%% @throws {error, reason} Description of when this error occurs
%%
%% Example:
%% ```
%% Result = module:function(Arg1, Arg2).
%% ```
-spec function_name(Param1, Param2) -> Result when
      Param1 :: param1_type(),
      Param2 :: param2_type(),
      Result :: result_type().
function_name(Param1, Param2) ->
    %% Implementation
    ok.
```

## Documentation Requirements

### 1. Brief Description
- One line summary of what the function does
- Uses imperative voice: "Calculate...", "Apply...", "Create..."

### 2. Detailed Description
- Explain the algorithm or logic
- Document any side effects
- Reference related functions

### 3. Parameters
- Use `@param` for each parameter
- Describe purpose, valid values, and units

### 4. Returns
- Use `@returns` to describe return value
- Document all possible return values

### 5. Errors
- Use `@throws` for each error case
- Describe when each error occurs

### 6. Examples
- Provide runnable code examples
- Show common use cases

## Type Specification (-spec)

Always include type specifications:

```erlang
%% Simple spec
-spec calculate_output(Signal :: signal_vector()) -> float().

%% Complex spec with when clause
-spec apply_plasticity(Rule, Inputs, Weights, Output) -> UpdatedWeights when
      Rule :: plasticity_function(),
      Inputs :: [{element_id(), signal_vector()}],
      Weights :: weighted_inputs(),
      Output :: float(),
      UpdatedWeights :: weighted_inputs().

%% Multiple clauses
-spec process(Input) -> Output when
      Input :: number(),
      Output :: number();
             (Input) -> Output when
      Input :: list(),
      Output :: list().
```

## Weight Tuple Documentation

When a function handles weights, document the tuple format:

```erlang
%% @doc Apply learning rule to update weights
%%
%% The weight tuple format is: {Weight, DeltaWeight, LearningRate, ParamList}
%% where:
%%   - Weight: current synaptic weight value
%%   - DeltaWeight: momentum term from previous update
%%   - LearningRate: plasticity rule learning rate
%%   - ParamList: additional parameters specific to learning rule
```

## Examples by Function Type

### Pure Function
```erlang
%% @doc Calculate dot product of signal vector and weights
%%
%% Computes: sum(Signal_i * Weight_i) for all i
%%
%% @param Signals List of input signals
%% @param Weights List of weight values
%% @returns Dot product result
-spec dot_product(Signals, Weights) -> float() when
      Signals :: [float()],
      Weights :: [float()].
dot_product(Signals, Weights) ->
    lists:sum([S * W || {S, W} <- lists:zip(Signals, Weights)]).
```

### Process Function
```erlang
%% @doc Start neuron process
%%
%% Spawns a new neuron process that will wait for signals and
%% forward outputs to connected neurons/actuators.
%%
%% @param ExoselfPid Parent exoself process for coordination
%% @param Node Target node for spawning
%% @returns Process ID of spawned neuron
-spec gen(ExoselfPid, Node) -> pid() when
      ExoselfPid :: pid(),
      Node :: node().
gen(ExoselfPid, Node) ->
    spawn(Node, ?MODULE, prep, [ExoselfPid]).
```

### Error-Returning Function
```erlang
%% @doc Load agent from database
%%
%% Retrieves agent record from Mnesia database.
%%
%% @param AgentId Agent identifier
%% @returns Agent record on success
%% @throws {error, not_found} when agent doesn't exist
%% @throws {error, database_error} when Mnesia transaction fails
-spec load_agent(AgentId) -> Agent | no_return() when
      AgentId :: agent_id(),
      Agent :: #agent{}.
load_agent(AgentId) ->
    case mnesia:read({agent, AgentId}) of
        [Agent] -> Agent;
        [] -> throw({error, not_found})
    end.
```
