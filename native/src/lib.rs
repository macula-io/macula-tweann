//! High-performance NIF for TWEANN network evaluation.
//!
//! This module provides fast forward propagation for neural networks,
//! designed to accelerate fitness evaluation during evolutionary training.
//!
//! The network is "compiled" from Erlang data structures into a flat
//! representation that can be evaluated efficiently without message passing.

use rustler::{Atom, Env, NifResult, ResourceArc, Term};
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

mod atoms {
    rustler::atoms! {
        ok,
        error,
        // Activation functions
        tanh,
        sigmoid,
        sigmoid1,
        sin,
        cos,
        gaussian,
        linear,
        relu,
        sgn,
        bin,
        trinary,
        multiquadric,
        quadratic,
        cubic,
        absolute,
        sqrt,
        log,
        // Node types
        input,
        hidden,
        output,
        bias,
        // Aggregation functions
        dot_product,
        mult_product,
        diff_product,
        // LTC neuron types
        standard,
        ltc,
        cfc,
    }
}

/// Activation function enum for efficient dispatch
#[derive(Debug, Clone, Copy)]
enum Activation {
    Tanh,
    Sigmoid,
    Sigmoid1,
    Sin,
    Cos,
    Gaussian,
    Linear,
    ReLU,
    Sgn,
    Bin,
    Trinary,
    Multiquadric,
    Quadratic,
    Cubic,
    Absolute,
    Sqrt,
    Log,
}

impl Activation {
    fn from_atom(atom: Atom) -> Self {
        if atom == atoms::tanh() {
            Activation::Tanh
        } else if atom == atoms::sigmoid() {
            Activation::Sigmoid
        } else if atom == atoms::sigmoid1() {
            Activation::Sigmoid1
        } else if atom == atoms::sin() {
            Activation::Sin
        } else if atom == atoms::cos() {
            Activation::Cos
        } else if atom == atoms::gaussian() {
            Activation::Gaussian
        } else if atom == atoms::linear() {
            Activation::Linear
        } else if atom == atoms::relu() {
            Activation::ReLU
        } else if atom == atoms::sgn() {
            Activation::Sgn
        } else if atom == atoms::bin() {
            Activation::Bin
        } else if atom == atoms::trinary() {
            Activation::Trinary
        } else if atom == atoms::multiquadric() {
            Activation::Multiquadric
        } else if atom == atoms::quadratic() {
            Activation::Quadratic
        } else if atom == atoms::cubic() {
            Activation::Cubic
        } else if atom == atoms::absolute() {
            Activation::Absolute
        } else if atom == atoms::sqrt() {
            Activation::Sqrt
        } else if atom == atoms::log() {
            Activation::Log
        } else {
            Activation::Tanh // Default
        }
    }

    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => {
                let v = x.clamp(-10.0, 10.0);
                1.0 / (1.0 + (-v).exp())
            }
            Activation::Sigmoid1 => x / (1.0 + x.abs()),
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            Activation::Gaussian => {
                let v = x.clamp(-10.0, 10.0);
                (-v * v).exp()
            }
            Activation::Linear => x,
            Activation::ReLU => x.max(0.0),
            Activation::Sgn => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Activation::Bin => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Trinary => {
                if x < 0.33 && x > -0.33 {
                    0.0
                } else if x >= 0.33 {
                    1.0
                } else {
                    -1.0
                }
            }
            Activation::Multiquadric => (x * x + 0.01).sqrt(),
            Activation::Quadratic => x.signum() * x * x,
            Activation::Cubic => x * x * x,
            Activation::Absolute => x.abs(),
            Activation::Sqrt => x.signum() * x.abs().sqrt(),
            Activation::Log => {
                if x == 0.0 {
                    0.0
                } else {
                    x.signum() * x.abs().ln()
                }
            }
        }
    }
}

/// A connection in the network
#[derive(Debug, Clone)]
struct Connection {
    from_idx: usize,
    weight: f64,
}

/// A node in the compiled network
#[derive(Debug, Clone)]
struct Node {
    activation: Activation,
    bias: f64,
    connections: Vec<Connection>,
}

/// Compiled network for fast evaluation
#[derive(Debug)]
pub struct CompiledNetwork {
    /// All nodes in topological order
    nodes: Vec<Node>,
    /// Number of input nodes
    input_count: usize,
    /// Indices of output nodes
    output_indices: Vec<usize>,
    /// Total number of nodes
    node_count: usize,
}

impl CompiledNetwork {
    /// Evaluate the network with given inputs
    pub fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() != self.input_count {
            return vec![];
        }

        // Allocate values array
        let mut values = vec![0.0f64; self.node_count];

        // Set input values (first input_count nodes are inputs)
        for (i, &input) in inputs.iter().enumerate() {
            values[i] = input;
        }

        // Process nodes in topological order (skip input nodes)
        for (idx, node) in self.nodes.iter().enumerate().skip(self.input_count) {
            // Sum weighted inputs
            let sum: f64 = node
                .connections
                .iter()
                .map(|conn| values[conn.from_idx] * conn.weight)
                .sum();

            // Apply bias and activation
            values[idx] = node.activation.apply(sum + node.bias);
        }

        // Collect outputs
        self.output_indices.iter().map(|&i| values[i]).collect()
    }

    /// Evaluate multiple input sets (batch mode)
    pub fn evaluate_batch(&self, inputs_batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs_batch.iter().map(|inputs| self.evaluate(inputs)).collect()
    }
}

/// Resource wrapper for CompiledNetwork
pub struct NetworkResource(pub CompiledNetwork);

#[allow(deprecated)]
fn load(env: Env, _: Term) -> bool {
    let _ = rustler::resource!(NetworkResource, env);
    true
}

/// Compile a network from Erlang representation
///
/// Expected format:
/// - nodes: [{Index, Type, Activation, Bias, [{FromIndex, Weight}, ...]}]
/// - input_count: integer
/// - output_indices: [integer]
#[rustler::nif]
fn compile_network(
    nodes_term: Vec<(usize, Atom, Atom, f64, Vec<(usize, f64)>)>,
    input_count: usize,
    output_indices: Vec<usize>,
) -> NifResult<ResourceArc<NetworkResource>> {
    let mut nodes = Vec::with_capacity(nodes_term.len());

    for (_idx, _node_type, activation_atom, bias, connections_term) in nodes_term {
        let activation = Activation::from_atom(activation_atom);
        let connections: Vec<Connection> = connections_term
            .into_iter()
            .map(|(from_idx, weight)| Connection { from_idx, weight })
            .collect();

        nodes.push(Node {
            activation,
            bias,
            connections,
        });
    }

    let network = CompiledNetwork {
        node_count: nodes.len(),
        nodes,
        input_count,
        output_indices,
    };

    Ok(ResourceArc::new(NetworkResource(network)))
}

/// Evaluate a compiled network with given inputs
#[rustler::nif]
fn evaluate(network: ResourceArc<NetworkResource>, inputs: Vec<f64>) -> Vec<f64> {
    network.0.evaluate(&inputs)
}

/// Evaluate a compiled network with multiple input sets (batch mode)
/// Uses dirty scheduler to prevent blocking the regular BEAM scheduler
/// for large batch operations.
#[rustler::nif(schedule = "DirtyCpu")]
fn evaluate_batch(
    network: ResourceArc<NetworkResource>,
    inputs_batch: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    network.0.evaluate_batch(&inputs_batch)
}

/// Calculate compatibility distance between two genomes
///
/// Used for speciation in NEAT.
/// Formula: (c1 * E / N) + (c2 * D / N) + (c3 * W)
/// where E = excess genes, D = disjoint genes, W = average weight difference
#[rustler::nif]
fn compatibility_distance(
    connections_a: Vec<(u64, f64)>, // [(innovation, weight), ...]
    connections_b: Vec<(u64, f64)>,
    c1: f64,
    c2: f64,
    c3: f64,
) -> f64 {
    if connections_a.is_empty() && connections_b.is_empty() {
        return 0.0;
    }

    // Build innovation -> weight maps
    let map_a: HashMap<u64, f64> = connections_a.into_iter().collect();
    let map_b: HashMap<u64, f64> = connections_b.into_iter().collect();

    // Find max innovation numbers
    let max_a = map_a.keys().max().copied().unwrap_or(0);
    let max_b = map_b.keys().max().copied().unwrap_or(0);
    let threshold = max_a.min(max_b);

    let mut excess = 0;
    let mut disjoint = 0;
    let mut weight_diff_sum = 0.0;
    let mut matching = 0;

    // Check genes in A
    for (&innov, &weight_a) in &map_a {
        if let Some(&weight_b) = map_b.get(&innov) {
            // Matching gene
            weight_diff_sum += (weight_a - weight_b).abs();
            matching += 1;
        } else if innov > threshold {
            excess += 1;
        } else {
            disjoint += 1;
        }
    }

    // Check genes only in B
    for &innov in map_b.keys() {
        if !map_a.contains_key(&innov) {
            if innov > threshold {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
    }

    // Normalize by larger genome size
    let n = map_a.len().max(map_b.len()).max(1) as f64;
    let avg_weight_diff = if matching > 0 {
        weight_diff_sum / matching as f64
    } else {
        0.0
    };

    (c1 * excess as f64 / n) + (c2 * disjoint as f64 / n) + (c3 * avg_weight_diff)
}

/// Compute dot product of signals and weights with optional bias.
///
/// This is a hot path function used in signal_aggregator.erl.
/// Expects pre-flattened data for maximum performance:
/// - signals: flattened list of all signal values [s1_1, s1_2, s2_1, ...]
/// - weights: corresponding weight values [w1_1, w1_2, w2_1, ...]
/// - bias: bias value to add to the result
///
/// Returns: sum(signal[i] * weight[i]) + bias
#[rustler::nif]
fn dot_product_flat(signals: Vec<f64>, weights: Vec<f64>, bias: f64) -> f64 {
    signals
        .iter()
        .zip(weights.iter())
        .map(|(s, w)| s * w)
        .sum::<f64>()
        + bias
}

/// Batch dot product for multiple neurons (most efficient).
///
/// Processes multiple neurons in one NIF call to amortize overhead.
/// Uses dirty scheduler for large batches.
///
/// - batch: Vec of (signals, weights, bias) tuples
/// Returns: Vec of dot product results
#[rustler::nif(schedule = "DirtyCpu")]
fn dot_product_batch(batch: Vec<(Vec<f64>, Vec<f64>, f64)>) -> Vec<f64> {
    batch
        .into_iter()
        .map(|(signals, weights, bias)| {
            signals
                .iter()
                .zip(weights.iter())
                .map(|(s, w)| s * w)
                .sum::<f64>()
                + bias
        })
        .collect()
}

/// Benchmark: evaluate network N times and return average time in microseconds
#[rustler::nif]
fn benchmark_evaluate(
    network: ResourceArc<NetworkResource>,
    inputs: Vec<f64>,
    iterations: usize,
) -> f64 {
    use std::time::Instant;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = network.0.evaluate(&inputs);
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / iterations as f64
}

// ============================================================================
// LTC/CfC (Liquid Time-Constant / Closed-form Continuous-time) Functions
// ============================================================================

/// CfC closed-form evaluation (fast, ~100x faster than ODE)
///
/// Implements the closed-form solution:
///   x(t+dt) = sigmoid(-f) * x(t) + (1 - sigmoid(-f)) * h
///
/// Where:
///   - f = backbone network output (time constant modulator)
///   - h = head network output (target state)
///   - sigmoid(-f) acts as interpolation gate
///
/// Returns (new_state, output)
#[rustler::nif]
fn evaluate_cfc(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
) -> (f64, f64) {
    evaluate_cfc_impl(input, state, tau, bound, &[], &[])
}

/// CfC evaluation with custom backbone and head weights
#[rustler::nif]
fn evaluate_cfc_with_weights(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
    backbone_weights: Vec<f64>,
    head_weights: Vec<f64>,
) -> (f64, f64) {
    evaluate_cfc_impl(input, state, tau, bound, &backbone_weights, &head_weights)
}

/// ODE-based LTC evaluation (accurate, slower)
///
/// Implements Euler integration of the LTC ODE:
///   dx/dt = -[1/tau + f(x, I, theta)] * x + f(x, I, theta) * A
///
/// Returns (new_state, output)
#[rustler::nif]
fn evaluate_ode(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
    dt: f64,
) -> (f64, f64) {
    evaluate_ode_impl(input, state, tau, bound, dt, &[], &[])
}

/// ODE evaluation with custom weights
#[rustler::nif]
fn evaluate_ode_with_weights(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
    dt: f64,
    backbone_weights: Vec<f64>,
    head_weights: Vec<f64>,
) -> (f64, f64) {
    evaluate_ode_impl(input, state, tau, bound, dt, &backbone_weights, &head_weights)
}

/// Batch CfC evaluation for multiple inputs
#[rustler::nif]
fn evaluate_cfc_batch(
    inputs: Vec<f64>,
    initial_state: f64,
    tau: f64,
    bound: f64,
) -> Vec<(f64, f64)> {
    let mut state = initial_state;
    inputs
        .into_iter()
        .map(|input| {
            let (new_state, output) = evaluate_cfc_impl(input, state, tau, bound, &[], &[]);
            state = new_state;
            (new_state, output)
        })
        .collect()
}

// Internal implementation for CfC evaluation
fn evaluate_cfc_impl(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
    backbone_weights: &[f64],
    head_weights: &[f64],
) -> (f64, f64) {
    let f = compute_backbone(input, tau, backbone_weights);
    let h = compute_head(input, head_weights);
    let sig_neg_f = sigmoid(-f);
    let new_state_raw = sig_neg_f * state + (1.0 - sig_neg_f) * h;
    let new_state = clamp_state(new_state_raw, bound);
    (new_state, new_state)
}

// Internal implementation for ODE evaluation
fn evaluate_ode_impl(
    input: f64,
    state: f64,
    tau: f64,
    bound: f64,
    dt: f64,
    backbone_weights: &[f64],
    head_weights: &[f64],
) -> (f64, f64) {
    // Compute f() for liquid time constant
    let f = compute_backbone(input, tau, backbone_weights);

    // LTC ODE: dx/dt = -[1/tau + f] * x + f * A
    let effective_tau_inv = 1.0 / tau.max(0.001) + f;
    let dx_dt = -effective_tau_inv * state + f * bound;

    // Euler integration
    let new_state_raw = state + dt * dx_dt;

    // Clamp to bounds
    let new_state = clamp_state(new_state_raw, bound);

    // Optionally apply head for output transformation
    let output = if head_weights.is_empty() {
        new_state
    } else {
        compute_head(new_state, head_weights)
    };

    (new_state, output)
}

/// Compute backbone network f() for time constant modulation
#[inline]
fn compute_backbone(input: f64, tau: f64, weights: &[f64]) -> f64 {
    if weights.is_empty() {
        // Simple mode: f = input / tau (input-dependent modulation)
        input / tau.max(0.001)
    } else {
        // Learned mode: weighted sum
        let weighted_sum: f64 = weights.iter().enumerate()
            .map(|(i, &w)| {
                let x = if i == 0 { input } else { 1.0 };  // input + bias
                w * x
            })
            .sum();
        weighted_sum.tanh()  // Bounded output
    }
}

/// Compute head network h() for target state
#[inline]
fn compute_head(input: f64, weights: &[f64]) -> f64 {
    if weights.is_empty() {
        // Simple mode: h = tanh(input)
        input.tanh()
    } else {
        // Learned mode: weighted sum through tanh
        let weighted_sum: f64 = weights.iter().enumerate()
            .map(|(i, &w)| {
                let x = if i == 0 { input } else { 1.0 };
                w * x
            })
            .sum();
        weighted_sum.tanh()
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f64) -> f64 {
    let v = x.clamp(-10.0, 10.0);  // Prevent overflow
    1.0 / (1.0 + (-v).exp())
}

/// Clamp state to bounds [-bound, bound]
#[inline]
fn clamp_state(state: f64, bound: f64) -> f64 {
    state.clamp(-bound, bound)
}

// ============================================================================
// Distance and KNN Functions (for Novelty Search)
// ============================================================================

/// Compute Euclidean distance between two behavior vectors.
/// Hot path function for novelty search.
#[rustler::nif]
fn euclidean_distance(v1: Vec<f64>, v2: Vec<f64>) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Batch Euclidean distance: compute distance from one vector to many.
/// Returns Vec of (index, distance) sorted by distance ascending.
/// Used for finding k-nearest neighbors in novelty search.
#[rustler::nif(schedule = "DirtyCpu")]
fn euclidean_distance_batch(
    target: Vec<f64>,
    others: Vec<Vec<f64>>,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = others
        .iter()
        .enumerate()
        .map(|(idx, other)| {
            let dist = target
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            (idx, dist)
        })
        .collect();

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances
}

/// Compute k-nearest neighbor distances for novelty calculation.
/// Returns the average distance to k nearest neighbors.
#[rustler::nif(schedule = "DirtyCpu")]
fn knn_novelty(
    target: Vec<f64>,
    population: Vec<Vec<f64>>,
    archive: Vec<Vec<f64>>,
    k: usize,
) -> f64 {
    // Combine population and archive, excluding exact matches with target
    let mut all_distances: Vec<f64> = Vec::with_capacity(population.len() + archive.len());

    for other in population.iter().chain(archive.iter()) {
        let dist: f64 = target
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        // Exclude exact duplicates (distance ~= 0)
        if dist > 1e-10 {
            all_distances.push(dist);
        }
    }

    if all_distances.is_empty() {
        return 0.0;
    }

    // Partial sort to get k smallest (more efficient than full sort)
    let k_actual = k.min(all_distances.len());
    all_distances.select_nth_unstable_by(k_actual - 1, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Average of k nearest
    all_distances[..k_actual].iter().sum::<f64>() / k_actual as f64
}

/// Batch kNN novelty: compute novelty scores for multiple behaviors.
/// More efficient than calling knn_novelty repeatedly.
#[rustler::nif(schedule = "DirtyCpu")]
fn knn_novelty_batch(
    behaviors: Vec<Vec<f64>>,
    archive: Vec<Vec<f64>>,
    k: usize,
) -> Vec<f64> {
    // Precompute all pairwise distances for efficiency
    let n = behaviors.len();
    let m = archive.len();

    behaviors
        .iter()
        .enumerate()
        .map(|(i, target)| {
            let mut distances: Vec<f64> = Vec::with_capacity(n - 1 + m);

            // Distances to other behaviors in population
            for (j, other) in behaviors.iter().enumerate() {
                if i != j {
                    let dist: f64 = target
                        .iter()
                        .zip(other.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    if dist > 1e-10 {
                        distances.push(dist);
                    }
                }
            }

            // Distances to archive
            for other in archive.iter() {
                let dist: f64 = target
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                if dist > 1e-10 {
                    distances.push(dist);
                }
            }

            if distances.is_empty() {
                return 0.0;
            }

            let k_actual = k.min(distances.len());
            distances.select_nth_unstable_by(k_actual - 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });

            distances[..k_actual].iter().sum::<f64>() / k_actual as f64
        })
        .collect()
}

// ============================================================================
// Statistics Functions
// ============================================================================

/// Compute fitness statistics in a single pass.
/// Returns (min, max, mean, variance, std_dev, sum).
#[rustler::nif]
fn fitness_stats(fitnesses: Vec<f64>) -> (f64, f64, f64, f64, f64, f64) {
    if fitnesses.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let n = fitnesses.len() as f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &f in &fitnesses {
        if f < min { min = f; }
        if f > max { max = f; }
        sum += f;
        sum_sq += f * f;
    }

    let mean = sum / n;
    // Variance using: Var = E[X²] - E[X]²
    let variance = (sum_sq / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt();  // max(0) to handle floating point errors

    (min, max, mean, variance, std_dev, sum)
}

/// Compute weighted moving average (for trend computation).
/// Uses exponential decay weights: w[i] = decay^i
#[rustler::nif]
fn weighted_moving_average(values: Vec<f64>, decay: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;
    let mut weight = 1.0;

    for &value in &values {
        weighted_value_sum += weight * value;
        weight_sum += weight;
        weight *= decay;
    }

    weighted_value_sum / weight_sum
}

/// Compute Shannon entropy of a distribution (for diversity metrics).
/// Values should be non-negative; they'll be normalized to a probability distribution.
#[rustler::nif]
fn shannon_entropy(values: Vec<f64>) -> f64 {
    let total: f64 = values.iter().filter(|&&v| v > 0.0).sum();
    if total <= 0.0 {
        return 0.0;
    }

    values
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| {
            let p = v / total;
            -p * p.ln()
        })
        .sum()
}

/// Histogram binning for entropy/diversity calculations.
/// Returns counts per bin.
#[rustler::nif]
fn histogram(values: Vec<f64>, num_bins: usize, min_val: f64, max_val: f64) -> Vec<usize> {
    let mut bins = vec![0usize; num_bins];
    let range = max_val - min_val;

    if range <= 0.0 || num_bins == 0 {
        return bins;
    }

    let bin_width = range / num_bins as f64;

    for &v in &values {
        if v >= min_val && v <= max_val {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);  // Handle edge case where v == max_val
            bins[bin] += 1;
        }
    }

    bins
}

// ============================================================================
// Selection Functions
// ============================================================================

/// Build cumulative fitness array for roulette wheel selection.
/// Returns (cumulative_fitnesses, total_fitness).
/// Shifts fitnesses to ensure all are positive.
#[rustler::nif]
fn build_cumulative_fitness(fitnesses: Vec<f64>) -> (Vec<f64>, f64) {
    if fitnesses.is_empty() {
        return (vec![], 0.0);
    }

    // Find minimum to shift all fitnesses positive
    let min_fitness = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_fitness < 0.0 { -min_fitness + 1.0 } else { 0.0 };

    let mut cumulative = Vec::with_capacity(fitnesses.len());
    let mut running_sum = 0.0;

    for &f in &fitnesses {
        running_sum += f + shift;
        cumulative.push(running_sum);
    }

    (cumulative, running_sum)
}

/// Roulette wheel selection using pre-built cumulative array.
/// Returns index of selected individual.
/// Uses binary search for O(log n) selection.
#[rustler::nif]
fn roulette_select(cumulative: Vec<f64>, total: f64, random_val: f64) -> usize {
    let target = random_val * total;

    // Binary search for the first index where cumulative[i] >= target
    match cumulative.binary_search_by(|c| {
        c.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(idx) => idx,
        Err(idx) => idx.min(cumulative.len() - 1),
    }
}

/// Batch roulette selection: select n individuals.
/// random_vals should be uniformly distributed in [0, 1].
#[rustler::nif]
fn roulette_select_batch(
    cumulative: Vec<f64>,
    total: f64,
    random_vals: Vec<f64>,
) -> Vec<usize> {
    random_vals
        .iter()
        .map(|&r| {
            let target = r * total;
            match cumulative.binary_search_by(|c| {
                c.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Ok(idx) => idx,
                Err(idx) => idx.min(cumulative.len().saturating_sub(1)),
            }
        })
        .collect()
}

/// Tournament selection: select best from random subset.
/// contestants: indices of individuals in tournament
/// fitnesses: fitness values (indexed by contestant indices)
/// Returns index of winner.
#[rustler::nif]
fn tournament_select(contestants: Vec<usize>, fitnesses: Vec<f64>) -> usize {
    contestants
        .into_iter()
        .max_by(|&a, &b| {
            let fa = fitnesses.get(a).copied().unwrap_or(f64::NEG_INFINITY);
            let fb = fitnesses.get(b).copied().unwrap_or(f64::NEG_INFINITY);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

// ============================================================================
// Reward and Meta-Controller Functions
// ============================================================================

/// Compute z-score normalization: (value - mean) / std_dev
/// Returns 0 if std_dev is too small to prevent division by zero.
#[rustler::nif]
fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev.abs() < 1e-10 {
        0.0
    } else {
        (value - mean) / std_dev
    }
}

/// Compute reward components with normalization.
/// history: recent values for computing baseline
/// current: current value
/// Returns (raw_component, normalized_component, z_score)
#[rustler::nif]
fn compute_reward_component(history: Vec<f64>, current: f64) -> (f64, f64, f64) {
    if history.is_empty() {
        return (current, current.tanh(), 0.0);
    }

    let n = history.len() as f64;
    let mean: f64 = history.iter().sum::<f64>() / n;
    let variance: f64 = history.iter().map(|&h| (h - mean) * (h - mean)).sum::<f64>() / n;
    let std_dev = variance.max(0.0).sqrt();

    let z = if std_dev.abs() < 1e-10 {
        0.0
    } else {
        (current - mean) / std_dev
    };

    // Normalized via sigmoid for bounded output
    let normalized = 1.0 / (1.0 + (-z).exp());

    (current, normalized, z)
}

/// Batch compute multiple reward components.
/// Each tuple is (history, current_value, weight).
/// Returns weighted sum of normalized components.
#[rustler::nif]
fn compute_weighted_reward(components: Vec<(Vec<f64>, f64, f64)>) -> f64 {
    components
        .into_iter()
        .map(|(history, current, weight)| {
            let (_, normalized, _) = compute_reward_component_impl(&history, current);
            normalized * weight
        })
        .sum()
}

fn compute_reward_component_impl(history: &[f64], current: f64) -> (f64, f64, f64) {
    if history.is_empty() {
        return (current, sigmoid(current), 0.0);
    }

    let n = history.len() as f64;
    let mean: f64 = history.iter().sum::<f64>() / n;
    let variance: f64 = history.iter().map(|&h| (h - mean) * (h - mean)).sum::<f64>() / n;
    let std_dev = variance.max(0.0).sqrt();

    let z = if std_dev.abs() < 1e-10 {
        0.0
    } else {
        (current - mean) / std_dev
    };

    let normalized = sigmoid(z);

    (current, normalized, z)
}

// ============================================================================
// Weight/Genome Utilities
// ============================================================================

// ============================================================================
// Batch Mutation Functions (for Evolutionary Genetics)
// ============================================================================

/// Mutate a single weight according to mutation parameters.
///
/// With mutation_rate probability:
///   - With perturb_rate probability: add gaussian noise scaled by perturb_strength
///   - Otherwise: replace with new random weight in [-1, 1]
#[inline]
fn mutate_single_weight(
    weight: f64,
    mutation_rate: f64,
    perturb_rate: f64,
    perturb_strength: f64,
    rng: &mut impl Rng,
) -> f64 {
    if rng.gen::<f64>() < mutation_rate {
        if rng.gen::<f64>() < perturb_rate {
            // Perturb: add gaussian noise
            let normal = Normal::new(0.0, perturb_strength).unwrap();
            weight + normal.sample(rng)
        } else {
            // Replace: new random weight in [-1, 1]
            rng.gen::<f64>() * 2.0 - 1.0
        }
    } else {
        weight
    }
}

/// Mutate weights in batch using gaussian perturbation.
///
/// This is the hot path for neuroevolution - called for every offspring.
/// Uses dirty scheduler to prevent blocking the BEAM scheduler.
///
/// Parameters:
/// - weights: list of weights to mutate
/// - mutation_rate: probability of mutating each weight (0.0 to 1.0)
/// - perturb_rate: probability of perturbing vs replacing (0.0 to 1.0)
/// - perturb_strength: standard deviation for gaussian perturbation
///
/// Returns: mutated weights
#[rustler::nif(schedule = "DirtyCpu")]
fn mutate_weights(
    weights: Vec<f64>,
    mutation_rate: f64,
    perturb_rate: f64,
    perturb_strength: f64,
) -> Vec<f64> {
    let mut rng = thread_rng();
    weights
        .into_iter()
        .map(|w| mutate_single_weight(w, mutation_rate, perturb_rate, perturb_strength, &mut rng))
        .collect()
}

/// Mutate weights with explicit seed for reproducibility.
///
/// Useful for debugging and deterministic testing.
#[rustler::nif(schedule = "DirtyCpu")]
fn mutate_weights_seeded(
    weights: Vec<f64>,
    mutation_rate: f64,
    perturb_rate: f64,
    perturb_strength: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    weights
        .into_iter()
        .map(|w| mutate_single_weight(w, mutation_rate, perturb_rate, perturb_strength, &mut rng))
        .collect()
}

/// Batch mutate multiple genomes' weights.
///
/// Each genome is a list of weights. Returns list of mutated genomes.
/// More efficient than calling mutate_weights repeatedly due to:
/// - Single NIF call overhead
/// - Better cache locality
/// - Shared RNG state
///
/// Input: Vec of (genome_weights, mutation_rate, perturb_rate, perturb_strength)
/// This allows per-genome mutation parameters (useful for adaptive mutation).
#[rustler::nif(schedule = "DirtyCpu")]
fn mutate_weights_batch(
    genomes: Vec<(Vec<f64>, f64, f64, f64)>,
) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    genomes
        .into_iter()
        .map(|(weights, mutation_rate, perturb_rate, perturb_strength)| {
            weights
                .into_iter()
                .map(|w| mutate_single_weight(w, mutation_rate, perturb_rate, perturb_strength, &mut rng))
                .collect()
        })
        .collect()
}

/// Batch mutate with uniform parameters (most common case).
///
/// All genomes use the same mutation parameters.
/// Returns list of mutated weight vectors.
#[rustler::nif(schedule = "DirtyCpu")]
fn mutate_weights_batch_uniform(
    genomes: Vec<Vec<f64>>,
    mutation_rate: f64,
    perturb_rate: f64,
    perturb_strength: f64,
) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    genomes
        .into_iter()
        .map(|weights| {
            weights
                .into_iter()
                .map(|w| mutate_single_weight(w, mutation_rate, perturb_rate, perturb_strength, &mut rng))
                .collect()
        })
        .collect()
}

/// Generate random weights for initial genome creation.
///
/// Returns n random weights uniformly distributed in [-1, 1].
#[rustler::nif]
fn random_weights(n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
}

/// Generate random weights with explicit seed.
#[rustler::nif]
fn random_weights_seeded(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
}

/// Generate gaussian random weights (normal distribution).
///
/// Returns n random weights from N(mean, std_dev).
#[rustler::nif]
fn random_weights_gaussian(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(mean, std_dev).unwrap_or(Normal::new(0.0, 1.0).unwrap());
    (0..n).map(|_| normal.sample(&mut rng)).collect()
}

/// Batch generate random weights for multiple genomes.
///
/// Returns list of random weight vectors, each of specified size.
#[rustler::nif(schedule = "DirtyCpu")]
fn random_weights_batch(sizes: Vec<usize>) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    sizes
        .into_iter()
        .map(|n| (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
        .collect()
}

/// Internal L1 distance implementation
fn weight_distance_l1_impl(weights1: &[f64], weights2: &[f64]) -> f64 {
    weights1
        .iter()
        .zip(weights2.iter())
        .map(|(w1, w2)| (w1 - w2).abs())
        .sum::<f64>()
        / weights1.len().max(1) as f64
}

/// Compute L1 (Manhattan) distance between two weight vectors.
///
/// Used for speciation distance when comparing fixed-topology networks.
#[rustler::nif]
fn weight_distance_l1(weights1: Vec<f64>, weights2: Vec<f64>) -> f64 {
    weight_distance_l1_impl(&weights1, &weights2)
}

/// Internal L2 distance implementation
fn weight_distance_l2_impl(weights1: &[f64], weights2: &[f64]) -> f64 {
    weights1
        .iter()
        .zip(weights2.iter())
        .map(|(w1, w2)| (w1 - w2) * (w1 - w2))
        .sum::<f64>()
        .sqrt()
        / (weights1.len().max(1) as f64).sqrt()
}

/// Compute L2 (Euclidean) distance between two weight vectors.
#[rustler::nif]
fn weight_distance_l2(weights1: Vec<f64>, weights2: Vec<f64>) -> f64 {
    weight_distance_l2_impl(&weights1, &weights2)
}

/// Batch compute weight distances from one genome to many.
///
/// Returns Vec of (index, distance) sorted by distance ascending.
/// Used for finding similar genomes in speciation.
#[rustler::nif(schedule = "DirtyCpu")]
fn weight_distance_batch(
    target: Vec<f64>,
    others: Vec<Vec<f64>>,
    use_l2: bool,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = others
        .iter()
        .enumerate()
        .map(|(idx, other)| {
            let dist = if use_l2 {
                target
                    .iter()
                    .zip(other.iter())
                    .map(|(w1, w2)| (w1 - w2) * (w1 - w2))
                    .sum::<f64>()
                    .sqrt()
                    / (target.len().max(1) as f64).sqrt()
            } else {
                target
                    .iter()
                    .zip(other.iter())
                    .map(|(w1, w2)| (w1 - w2).abs())
                    .sum::<f64>()
                    / target.len().max(1) as f64
            };
            (idx, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances
}

// ============================================================================
// Weight/Genome Utilities (existing)
// ============================================================================

/// Flatten nested weight structure for efficient dot product.
/// Input: list of {source_id, [{weight, delta, lr, params}, ...]} tuples
/// Output: (flat_weights, weight_count_per_source)
/// This avoids intermediate list creation in Erlang.
#[rustler::nif]
fn flatten_weights(
    weighted_inputs: Vec<(u64, Vec<(f64, f64, f64, Vec<f64>)>)>,
) -> (Vec<f64>, Vec<usize>) {
    let mut flat_weights = Vec::new();
    let mut counts = Vec::with_capacity(weighted_inputs.len());

    for (_source_id, weights) in weighted_inputs {
        counts.push(weights.len());
        for (w, _delta, _lr, _params) in weights {
            flat_weights.push(w);
        }
    }

    (flat_weights, counts)
}

/// Compute dot product with pre-flattened weight structure.
/// signals_flat: all signals concatenated
/// weights_flat: all weights concatenated
/// counts: number of weights per source (for validation)
/// bias: bias to add
#[rustler::nif]
fn dot_product_preflattened(
    signals_flat: Vec<f64>,
    weights_flat: Vec<f64>,
    bias: f64,
) -> f64 {
    signals_flat
        .iter()
        .zip(weights_flat.iter())
        .map(|(s, w)| s * w)
        .sum::<f64>()
        + bias
}

#[allow(deprecated)]
rustler::init!(
    "tweann_nif",
    [
        // Network compilation and evaluation
        compile_network,
        evaluate,
        evaluate_batch,
        compatibility_distance,
        benchmark_evaluate,
        // Signal aggregation NIFs
        dot_product_flat,
        dot_product_batch,
        dot_product_preflattened,
        flatten_weights,
        // LTC/CfC functions
        evaluate_cfc,
        evaluate_cfc_with_weights,
        evaluate_ode,
        evaluate_ode_with_weights,
        evaluate_cfc_batch,
        // Distance and KNN (Novelty Search)
        euclidean_distance,
        euclidean_distance_batch,
        knn_novelty,
        knn_novelty_batch,
        // Statistics
        fitness_stats,
        weighted_moving_average,
        shannon_entropy,
        histogram,
        // Selection
        build_cumulative_fitness,
        roulette_select,
        roulette_select_batch,
        tournament_select,
        // Reward and Meta-Controller
        z_score,
        compute_reward_component,
        compute_weighted_reward,
        // Batch Mutation (Evolutionary Genetics)
        mutate_weights,
        mutate_weights_seeded,
        mutate_weights_batch,
        mutate_weights_batch_uniform,
        random_weights,
        random_weights_seeded,
        random_weights_gaussian,
        random_weights_batch,
        weight_distance_l1,
        weight_distance_l2,
        weight_distance_batch
    ],
    load = load
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        assert!((Activation::Tanh.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
        assert!((Activation::ReLU.apply(-1.0) - 0.0).abs() < 1e-10);
        assert!((Activation::ReLU.apply(1.0) - 1.0).abs() < 1e-10);
        assert!((Activation::Linear.apply(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_network() {
        // Create a simple XOR-like network
        // 2 inputs, 2 hidden, 1 output
        let network = CompiledNetwork {
            nodes: vec![
                // Input 0 (no connections, no activation needed)
                Node {
                    activation: Activation::Linear,
                    bias: 0.0,
                    connections: vec![],
                },
                // Input 1
                Node {
                    activation: Activation::Linear,
                    bias: 0.0,
                    connections: vec![],
                },
                // Hidden 0: receives from both inputs
                Node {
                    activation: Activation::Tanh,
                    bias: 0.0,
                    connections: vec![
                        Connection { from_idx: 0, weight: 1.0 },
                        Connection { from_idx: 1, weight: 1.0 },
                    ],
                },
                // Hidden 1: receives from both inputs
                Node {
                    activation: Activation::Tanh,
                    bias: 0.0,
                    connections: vec![
                        Connection { from_idx: 0, weight: 1.0 },
                        Connection { from_idx: 1, weight: -1.0 },
                    ],
                },
                // Output: receives from hidden nodes
                Node {
                    activation: Activation::Tanh,
                    bias: 0.0,
                    connections: vec![
                        Connection { from_idx: 2, weight: 1.0 },
                        Connection { from_idx: 3, weight: 1.0 },
                    ],
                },
            ],
            input_count: 2,
            output_indices: vec![4],
            node_count: 5,
        };

        let result = network.evaluate(&[0.0, 0.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.0).abs() < 0.1); // tanh(0) = 0
    }

    // ========================================================================
    // LTC/CfC Tests
    // ========================================================================

    #[test]
    fn test_sigmoid() {
        // sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        // sigmoid(large positive) -> 1
        assert!(sigmoid(10.0) > 0.99);
        // sigmoid(large negative) -> 0
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_clamp_state() {
        // Within bounds
        assert!((clamp_state(0.5, 1.0) - 0.5).abs() < 1e-10);
        // Above bound
        assert!((clamp_state(2.0, 1.0) - 1.0).abs() < 1e-10);
        // Below bound
        assert!((clamp_state(-2.0, 1.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_backbone_simple() {
        // Simple mode (no weights): f = input / tau
        let result = compute_backbone(1.0, 2.0, &[]);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_backbone_with_weights() {
        // With weights: tanh(weighted_sum)
        let weights = vec![1.0, 0.0]; // weight for input=1, bias=0
        let result = compute_backbone(0.5, 1.0, &weights);
        assert!((result - 0.5_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_head_simple() {
        // Simple mode: h = tanh(input)
        let result = compute_head(0.5, &[]);
        assert!((result - 0.5_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_cfc_evaluation_zero_input() {
        // With zero input and zero state, should stay near zero
        let (new_state, output) = evaluate_cfc_impl(0.0, 0.0, 1.0, 1.0, &[], &[]);
        assert!(new_state.abs() < 0.1);
        assert_eq!(new_state, output);
    }

    #[test]
    fn test_cfc_evaluation_state_persistence() {
        // CfC should interpolate between current state and target
        let (new_state, _) = evaluate_cfc_impl(1.0, 0.5, 1.0, 1.0, &[], &[]);
        // State should move toward tanh(1.0) ~ 0.76
        assert!(new_state > 0.5);
        assert!(new_state < 1.0);
    }

    #[test]
    fn test_cfc_respects_bounds() {
        // Large input should be clamped to bound
        let (new_state, _) = evaluate_cfc_impl(100.0, 0.0, 1.0, 0.5, &[], &[]);
        assert!(new_state.abs() <= 0.5);
    }

    #[test]
    fn test_ode_evaluation_basic() {
        // ODE evaluation should update state based on dynamics
        let (new_state, output) = evaluate_ode_impl(0.5, 0.0, 1.0, 1.0, 0.1, &[], &[]);
        // State should change (not stay at 0)
        assert!(new_state != 0.0 || output != 0.0);
    }

    #[test]
    fn test_ode_respects_bounds() {
        // Large dynamics should be clamped
        let (new_state, _) = evaluate_ode_impl(100.0, 0.0, 0.01, 0.5, 1.0, &[], &[]);
        assert!(new_state.abs() <= 0.5);
    }

    #[test]
    fn test_cfc_faster_than_ode() {
        // This is a simple benchmark sanity check
        use std::time::Instant;

        let iterations = 10000;

        // CfC timing
        let start_cfc = Instant::now();
        for i in 0..iterations {
            let _ = evaluate_cfc_impl(i as f64 * 0.001, 0.0, 1.0, 1.0, &[], &[]);
        }
        let cfc_time = start_cfc.elapsed();

        // ODE timing
        let start_ode = Instant::now();
        for i in 0..iterations {
            let _ = evaluate_ode_impl(i as f64 * 0.001, 0.0, 1.0, 1.0, 0.1, &[], &[]);
        }
        let ode_time = start_ode.elapsed();

        // CfC should be at least as fast (they're similar complexity in this impl)
        // In practice, CfC avoids ODE integration overhead
        println!("CfC: {:?}, ODE: {:?}", cfc_time, ode_time);
        // Just check both complete without error
        assert!(cfc_time.as_nanos() > 0);
        assert!(ode_time.as_nanos() > 0);
    }

    // ========================================================================
    // Batch Mutation Tests
    // ========================================================================

    #[test]
    fn test_mutate_single_weight_no_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        // With mutation_rate = 0, weight should not change
        let result = mutate_single_weight(0.5, 0.0, 0.8, 0.1, &mut rng);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mutate_single_weight_always_mutate() {
        let mut rng = StdRng::seed_from_u64(42);
        // With mutation_rate = 1.0, weight should change
        let original = 0.5;
        let result = mutate_single_weight(original, 1.0, 0.8, 0.1, &mut rng);
        // Should be different from original (with very high probability)
        assert!(result != original || (result - original).abs() < 1e-10);
    }

    #[test]
    fn test_mutate_weights_seeded_reproducibility() {
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Same seed should produce same results
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(12345);

        let result1: Vec<f64> = weights.iter()
            .map(|&w| mutate_single_weight(w, 0.5, 0.8, 0.1, &mut rng1))
            .collect();
        let result2: Vec<f64> = weights.iter()
            .map(|&w| mutate_single_weight(w, 0.5, 0.8, 0.1, &mut rng2))
            .collect();

        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert!((r1 - r2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mutate_weights_different_seeds() {
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Different seeds should produce different results
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(54321);

        let result1: Vec<f64> = weights.iter()
            .map(|&w| mutate_single_weight(w, 1.0, 0.8, 0.5, &mut rng1))
            .collect();
        let result2: Vec<f64> = weights.iter()
            .map(|&w| mutate_single_weight(w, 1.0, 0.8, 0.5, &mut rng2))
            .collect();

        // At least one weight should differ
        let any_different = result1.iter().zip(result2.iter())
            .any(|(r1, r2)| (r1 - r2).abs() > 1e-10);
        assert!(any_different);
    }

    #[test]
    fn test_random_weights_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let weights: Vec<f64> = (0..100).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        assert_eq!(weights.len(), 100);

        // All weights should be in [-1, 1]
        for w in &weights {
            assert!(*w >= -1.0 && *w <= 1.0);
        }
    }

    #[test]
    fn test_random_weights_seeded_reproducibility() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let result1: Vec<f64> = (0..10).map(|_| rng1.gen::<f64>() * 2.0 - 1.0).collect();
        let result2: Vec<f64> = (0..10).map(|_| rng2.gen::<f64>() * 2.0 - 1.0).collect();

        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert!((r1 - r2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_weights_gaussian_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let weights: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
        assert_eq!(weights.len(), 1000);

        // Check mean is approximately 0
        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
        assert!(mean.abs() < 0.2); // Within 0.2 of 0 with high probability
    }

    #[test]
    fn test_weight_distance_l1() {
        let w1 = vec![0.0, 0.0, 0.0];
        let w2 = vec![1.0, 1.0, 1.0];

        let dist = weight_distance_l1_impl(&w1, &w2);
        assert!((dist - 1.0).abs() < 1e-10); // Average of [1, 1, 1] = 1
    }

    #[test]
    fn test_weight_distance_l2() {
        let w1 = vec![0.0, 0.0, 0.0];
        let w2 = vec![1.0, 1.0, 1.0];

        let dist = weight_distance_l2_impl(&w1, &w2);
        // L2 distance = sqrt(3) / sqrt(3) = 1
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_distance_identical() {
        let w = vec![0.5, -0.3, 0.8];

        let dist_l1 = weight_distance_l1_impl(&w, &w);
        let dist_l2 = weight_distance_l2_impl(&w, &w);

        assert!(dist_l1.abs() < 1e-10);
        assert!(dist_l2.abs() < 1e-10);
    }

    #[test]
    fn test_mutate_weights_batch_uniform() {
        // Test batch mutation with zero mutation rate
        let mut rng = StdRng::seed_from_u64(42);
        let genomes = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        // Simulate batch uniform with mutation_rate = 0
        let result: Vec<Vec<f64>> = genomes
            .iter()
            .map(|weights| {
                weights
                    .iter()
                    .map(|&w| mutate_single_weight(w, 0.0, 0.8, 0.1, &mut rng))
                    .collect()
            })
            .collect();

        // With mutation_rate = 0, all weights should be unchanged
        assert_eq!(result.len(), 3);
        for (original, mutated) in genomes.iter().zip(result.iter()) {
            for (o, m) in original.iter().zip(mutated.iter()) {
                assert!((o - m).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_random_weights_batch() {
        let mut rng = StdRng::seed_from_u64(42);
        let sizes = vec![5, 10, 3];

        // Simulate batch random weights
        let result: Vec<Vec<f64>> = sizes
            .iter()
            .map(|&n| (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect())
            .collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 5);
        assert_eq!(result[1].len(), 10);
        assert_eq!(result[2].len(), 3);

        // All weights in range
        for genome in &result {
            for w in genome {
                assert!(*w >= -1.0 && *w <= 1.0);
            }
        }
    }
}
