# Macula Neuroevolution Libraries - Publishing Plan

## Current State Analysis

### Package Versions

| Package | Current Version | Hex Published | Notes |
|---------|----------------|---------------|-------|
| macula | 0.14.1 | Yes | HTTP/3 mesh platform |
| macula_tweann | 0.10.0 | Yes | Neural network library |
| macula_neuroevolution | 0.1.0 | Yes | Evolution engine |

### Dependency Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    macula-neurolab                           │   │
│  │                    (Elixir/Phoenix)                          │   │
│  │                    NOT on Hex.pm                             │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             │ depends on (via _checkouts)           │
│                             ▼                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         LIBRARY LAYER                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  macula_neuroevolution                       │   │
│  │                      v0.1.0                                  │   │
│  │           Population management, selection,                  │   │
│  │           mutation, breeding, speciation                     │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             │ depends on (Hex: ~> 0.10.0)           │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    macula_tweann                             │   │
│  │                      v0.10.0                                 │   │
│  │           Neural networks, LTC neurons, ONNX export,         │   │
│  │           topology evolution, morphologies                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             │ NO dependency on macula               │
│                             ▼                                       │
│                         (standalone)                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        PLATFORM LAYER                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        macula                                │   │
│  │                      v0.14.1                                 │   │
│  │           HTTP/3 mesh, DHT, NAT traversal, PubSub, RPC       │   │
│  │                                                              │   │
│  │           INDEPENDENT - no dependency on tweann/neuroevo     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Dependency Analysis

**macula_tweann** (v0.10.0)
- Dependencies: `kernel`, `stdlib`, `mnesia` (OTP only)
- NO external Hex dependencies
- Standalone library

**macula_neuroevolution** (v0.1.0)
- Dependencies: `macula_tweann ~> 0.10.0` (via Hex)
- Uses `_checkouts` symlink for development

**macula** (v0.14.1)
- Dependencies: `quicer`, `msgpack`, `gproc` (via Hex)
- INDEPENDENT of tweann/neuroevolution
- Different problem domain (networking vs AI)

---

## Versioning Strategy

### Semantic Versioning (SemVer)

All packages follow SemVer: `MAJOR.MINOR.PATCH`

| Change Type | Bump | Example |
|-------------|------|---------|
| Breaking API change | MAJOR | 0.x → 1.0.0 |
| New feature (backwards compatible) | MINOR | 0.10.0 → 0.11.0 |
| Bug fix, docs, tests | PATCH | 0.10.0 → 0.10.1 |

### Version Alignment Strategy

**Option A: Independent Versioning** (RECOMMENDED)
- Each package versions independently
- Dependency constraints use compatible ranges (`~> 0.10.0`)
- Allows packages to evolve at their own pace

**Option B: Synchronized Versioning**
- All packages share major.minor version
- Only patch versions can differ
- Simpler for users, harder to maintain

**Recommendation: Option A** - These are distinct libraries with different release cadences.

### Next Versions to Publish

| Package | Current | Next | Changes |
|---------|---------|------|---------|
| macula_tweann | 0.10.0 | 0.11.0 | ONNX export, network serialization |
| macula_neuroevolution | 0.1.0 | 0.2.0 | Fitness threshold, training_complete event |
| macula | 0.14.1 | (no changes) | - |

---

## Code Quality Requirements

### Pre-Publication Checklist

For each package before publishing:

1. **Tests**
   - [ ] All unit tests pass: `rebar3 eunit`
   - [ ] All common tests pass: `rebar3 ct`
   - [ ] Coverage report generated: `rebar3 cover`
   - [ ] Minimum 80% coverage for core modules

2. **Static Analysis**
   - [ ] Dialyzer passes: `rebar3 dialyzer`
   - [ ] No warnings in compilation
   - [ ] XRef checks pass: `rebar3 xref`

3. **Documentation**
   - [ ] All exported functions have @doc
   - [ ] All exported functions have @spec
   - [ ] Module-level @moduledoc present
   - [ ] ExDoc generates without errors: `rebar3 ex_doc`
   - [ ] All links verified (internal and external)

4. **Idiomatic Erlang**
   - [ ] No `if` statements (use pattern matching)
   - [ ] No deep nesting (max 2 levels)
   - [ ] Guards preferred over `case` where possible
   - [ ] No `try..catch` unless absolutely necessary

---

## Documentation Requirements

### Required Sections per Package

1. **README.md**
   - Project description
   - Quick start example
   - Installation instructions
   - Link to full documentation
   - License

2. **Guides** (in `guides/` or `docs/`)
   - Overview
   - Getting Started
   - API Reference (auto-generated)
   - Architecture
   - Examples

3. **Academic References**
   - Papers cited for algorithms
   - Theoretical foundations
   - Comparison to prior work

### SVG Diagram Requirements

Each package needs high-quality SVG diagrams for:

1. **Architecture Overview** - Module relationships
2. **Data Flow** - How data moves through the system
3. **API Surface** - Public interface visualization
4. **Algorithm Diagrams** - For complex algorithms

Diagrams stored in: `docs/diagrams/` or `design_docs/diagrams/`

---

## Academic References to Include

### macula_tweann

| Algorithm/Concept | Primary Reference |
|-------------------|-------------------|
| TWEANN | Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies." Evolutionary Computation, 10(2), 99-127. |
| LTC Neurons | Hasani, R., et al. (2021). "Liquid Time-constant Networks." AAAI 2021. |
| CfC (Closed-form Continuous) | Hasani, R., et al. (2022). "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence. |
| NEAT | Stanley, K.O. (2004). "Efficient Evolution of Neural Network Topologies." CEC 2002. |
| Xavier Initialization | Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." AISTATS. |

### macula_neuroevolution

| Algorithm/Concept | Primary Reference |
|-------------------|-------------------|
| Genetic Algorithms | Holland, J.H. (1975). "Adaptation in Natural and Artificial Systems." MIT Press. |
| Neuroevolution | Yao, X. (1999). "Evolving Artificial Neural Networks." Proceedings of the IEEE. |
| Speciation | Stanley, K.O. & Miikkulainen, R. (2002). NEAT paper (speciation section). |
| Tournament Selection | Miller, B.L. & Goldberg, D.E. (1995). "Genetic Algorithms, Tournament Selection, and the Effects of Noise." Complex Systems. |
| Fitness Sharing | Goldberg, D.E. & Richardson, J. (1987). "Genetic algorithms with sharing for multimodal function optimization." ICGA. |

### macula (mesh platform)

| Algorithm/Concept | Primary Reference |
|-------------------|-------------------|
| Kademlia DHT | Maymounkov, P. & Mazieres, D. (2002). "Kademlia: A Peer-to-peer Information System Based on the XOR Metric." IPTPS. |
| QUIC Protocol | Iyengar, J. & Thomson, M. (2021). "QUIC: A UDP-Based Multiplexed and Secure Transport." RFC 9000. |
| CRDTs | Shapiro, M., et al. (2011). "A comprehensive study of Convergent and Commutative Replicated Data Types." INRIA RR-7506. |
| NAT Traversal | Ford, B., et al. (2005). "Peer-to-Peer Communication Across Network Address Translators." USENIX ATC. |

---

## Related Projects Section

Add to each package's documentation:

```markdown
## Related Projects

### Macula Ecosystem

- **[macula](https://hex.pm/packages/macula)** - HTTP/3 mesh networking platform with NAT traversal, Pub/Sub, and async RPC
- **[macula_tweann](https://hex.pm/packages/macula_tweann)** - Neural network library with LTC neurons and ONNX export
- **[macula_neuroevolution](https://hex.pm/packages/macula_neuroevolution)** - Population-based evolutionary training engine

### Inspiration & Related Work

- **[DXNN2](https://github.com/CorticalComputer/DXNN2)** - Gene Sher's original TWEANN implementation
- **[NEAT-Python](https://neat-python.readthedocs.io/)** - Python NEAT implementation
- **[SharpNEAT](http://sharpneat.sourceforge.net/)** - C# NEAT implementation
- **[pytorch-neat](https://github.com/uber-research/PyTorch-NEAT)** - Uber's PyTorch NEAT
```

---

## Link Quality Verification

### Automated Checks

Create script: `scripts/check-links.sh`

```bash
#!/bin/bash
# Check all markdown links in documentation

find . -name "*.md" -exec markdown-link-check {} \;
```

### Manual Verification

Before each release:
1. Run `rebar3 ex_doc`
2. Open generated docs in browser
3. Click every link manually
4. Fix any 404s or broken anchors

---

## Publishing Workflow

### Pre-release Checklist

```bash
# 1. Run all quality checks
rebar3 compile
rebar3 dialyzer
rebar3 xref
rebar3 eunit
rebar3 ct
rebar3 cover

# 2. Generate and verify docs
rebar3 ex_doc
# Open _build/default/lib/<package>/doc/index.html

# 3. Verify Hex package contents
rebar3 hex build

# 4. Publish (dry run first)
rebar3 hex publish --dry-run

# 5. Actual publish
rebar3 hex publish
```

### Post-release

1. Tag release in Git: `git tag -a v0.11.0 -m "Release 0.11.0"`
2. Push tag: `git push origin v0.11.0`
3. Update CHANGELOG.md
4. Announce on relevant channels

---

## Action Items

### Immediate (Before Next Release)

1. [ ] Add ONNX export to macula_tweann (DONE)
2. [ ] Add fitness threshold to macula_neuroevolution (DONE)
3. [ ] Create SVG architecture diagrams
4. [ ] Add academic references to all docs
5. [ ] Verify all documentation links
6. [ ] Run full quality checks
7. [ ] Bump versions and publish

### Short-term

1. [ ] Increase test coverage to 80%+
2. [ ] Add property-based tests (PropEr)
3. [ ] Create video tutorials
4. [ ] Write blog posts about the libraries

### Long-term

1. [ ] Integration examples with popular frameworks
2. [ ] Benchmark suite
3. [ ] Conference paper submission
