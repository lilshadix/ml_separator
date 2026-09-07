# Gen12.2 — coordination-topology descriptor specification

*Frozen with `config/coordination_smarts.json` version 1.1.0, SHA-256 `4b073660…`, before any
Gen12.2 level model was fitted. The digest is recorded beside every feature matrix and asserted at
scoring time, so a specification edited after a result was seen cannot silently reach a table.*

---

## 0. What these descriptors are, and what they are not

They are **topological descriptors over a 2D molecular graph**. Every one is a function of one
canonical SMILES string and of nothing else.

- `dist__donor_pair_min` is a count of **bonds**, not an interatomic distance in ångström.
- `arm__estimated_denticity_per_pocket` is the number of **potential** donor atoms in one graph
  neighbourhood. It is not a measured denticity and not a coordination number.
- `arm__local_donor_cluster_count` is a **graph-component count**. It is not a number of metal
  centres and it does not assert that the molecule binds more than one metal.
- `motif__n_dga_unit` is a **substructure match count**, not a statement about binding.

Names in the denticity family carry `potential_`, `estimated_` or `local_` for that reason. **No
descriptor in this block contains geometry, and none may be described as one.** Where 3D geometry
exists in the repository it is deliberately unused: it covers only 1,370 of the raw Eu rows and
the repository's verdict across gen2, gen3 and gen5 is that it adds no transferable signal over 2D.

## 1. Why this block exists

Gen12 found that the ten worst zero-shot extractants of its champion are bridged, tripodal or
multi-armed diglycolamides, **under-predicted by 2.6 to 4.2 decades each**, and reported that as a
representation failure rather than a data failure. A radius-2 Morgan fingerprint sets the same bits
for one diglycolamide unit and for three of them tied to a benzene core, up to the bits of the
linker. The hypothesis this block exists to test is that what the fingerprint cannot express is
**how many separate chelating pockets a molecule presents, and how its donors are arranged around
them**, and that this is what sets the extractant's intrinsic extraction level.

The block is deliberately small and interpretable. It is not a descriptor sweep: 114 columns in 5
named families, every one defined below, and none selected on any test result.

## 2. Families

| family | prefix | columns | what it encodes |
|---|---|---|---|
| donor census | `coord__donor__` | 23 | how many potential donor atoms of each functional class and each element |
| motif counts | `coord__motif__` | 23 | named chelating units and ring classes |
| binding-arm multiplicity | `coord__arm__` | 17 | how many separate chelating pockets, how many are repeats of one another |
| donor topology | `coord__dist__` | 22 | graph distances between donors, and their histogram |
| architecture | `coord__arch__` | 24 | size, branching, symmetry, and size per donor |

109 of the 114 vary across the cohort. The 5 constant ones — formal charge, the two
phosphine-phosphorus columns, ester carbonyl oxygen and the aryl-ether fraction — are kept in the
definition because they are part of the contract and could vary on another cohort, and they are
dropped fold-locally by the preprocessor. **A statement of the form "114 coordination features" is
wrong by five on this cohort** and the informative count is what is quoted.

## 3. Donor classes

A potential donor is an atom that could donate a lone pair to a hard trivalent lanthanide. Classes
are matched by SMARTS and **made disjoint in declaration order**, because `O_carboxyl` and
`O_hydroxyl` would otherwise both claim the same acidic oxygen.

| class | SMARTS | why |
|---|---|---|
| `O_amide_carbonyl` | `[OX1]=[CX3][NX3]` | the principal hard donor of every amide-type extractant here |
| `O_ester_carbonyl` | `[OX1]=[CX3][OX2][#6]` | weaker than an amide carbonyl; counted apart |
| `O_carboxyl` | `[CX3](=[OX1])[OX2H1,OX1-]` | strong anionic donor when deprotonated; both oxygens counted |
| `O_ketone` | `[OX1]=[CX3;!$([CX3][NX3]);!$([CX3][OX2]);!$([CX3][OX1-])]` | carbonyl that is neither amide, ester nor carboxyl |
| `O_ether` | `[OX2;!$([OX2][CX3]=[OX1]);!$([OX2][P,S]);!$([OX2H])]([CX4])[CX4]` | aliphatic ether: the bridging donor of a diglycolamide, a podand and a crown |
| `O_ether_aryl` | `[OX2;…]([c,C])[c]` | aryl ether: the lone pair is conjugated into the ring, so it is a weaker donor and is counted apart |
| `O_hydroxyl` | `[OX2H;!$([OX2H][CX3]=[OX1])]` | alcohol or phenol, excluding the acidic OH of a carboxyl |
| `O_phosphoryl` | `[OX1]=[PX4]` | CMPO, phosphine oxide, phosphonate, phosphate |
| `O_phosphate_ester` | `[OX2]([#6])[PX4]` | P–O–C ester oxygen, a weak donor |
| `O_nitroxide` | `[OX1-][n+,NX4+]` | one pyridine-N-oxide bis(phosphine oxide) is present |
| `N_aromatic` | `[nX2]` | **pyridine-type only.** Pyrrole-type `[nX3]` is excluded because its lone pair is in the ring π system and it does not coordinate. This is the single most important exclusion in the block: counting nitrogen by element would credit every pyrrole, indole and N-substituted pyrazole with a donor it does not have |
| `N_amine` | `[NX3;!$([NX3][CX3]=[O,S]);!$([NX3]=*);!$([NX3+]);!$([NX3][a]);!n]` | aliphatic amine that is not an amide, an anilide or charged |
| `S_thiophosphoryl` | `[SX1]=[PX4]` | soft donor of the dithiophosphinate family |
| `S_thiocarbonyl` | `[SX1]=[CX3]` | soft donor of the thioamide family |
| `S_thioether_thiol` | `[SX2]` | divalent sulfur |
| `P_phosphine` | `[PX3;!$([PX3]=*)]` | trivalent P lone pair; absent from this cohort, kept for completeness |

Element groups: `hard_O` (ten oxygen classes), `N`, `soft_S`, `P`. Derived: total potential donor
count, per-group counts and fractions, element diversity, and a soft-donor flag.

## 4. Chelating-unit motifs

Counted with `uniquify=True`, so a symmetric motif that matches in both directions over the same
atom set counts once.

| motif | SMARTS | note |
|---|---|---|
| `dga_unit` | `[N]C(=O)[#6;X4,a][#8X2][#6;X4,a]C(=O)[N]` | reused **verbatim** from `lanthanide_separation.ligand_descriptors`, so this count is the same object gen6 used to assign the diglycolamide family. Matches 74 of 183 structures, exactly the family size |
| `malonamide_unit` | `[N]C(=O)[CX4]C(=O)[N]` | reused verbatim; matches 11, exactly the family size |
| `cmpo_unit` | `[PX4](=[OX1])[CX4]C(=[OX1])[NX3]` | carbamoylmethylphosphine oxide |
| `phosphoryl_amide_extended` | `[PX4](=[OX1])[CX4][#7,#8][CX4]C(=[OX1])[NX3]` | the three-atom-bridge variant of the TWE-33…36 series |
| `bis_thiophosphoryl_unit`, `bis_thiophosphoryl_methylene` | see the JSON | the TWE-23/24/27/28/29/30 family |
| `picolinamide_arm` | `[NX3]C(=[OX1])[c;$(c[nX2])]` | an amide on a heteroaromatic carbon next to a pyridine-type N: the N,O-chelating arm of picolinamides, dipicolinamides and phenanthroline diamides |
| `carboxylate_arm` | `[NX3][CX4]C(=[OX1])[OX2H1,OX1-]` | glycinate arm of an aminopolycarboxylate |
| `beta_diketone_unit` | `[#6](=[OX1])[CX4][#6](=[OX1])` | a malonamide also matches this; the two counts are **reported separately rather than made exclusive**, and that overlap is documented rather than hidden |
| `thioamide_unit`, `aryl_amide`, `phosphine_oxide`, `phosphonate_ester`, `phosphate_ester` | see the JSON | |

Ring classes come from the symmetrised smallest set of smallest rings rather than SMARTS, so a
fused system is counted once per ring: pyridine-like, diazine-like, triazine-like, azole,
crown-like (ring of 9 or more with 3 or more oxygens) and lactam.

## 5. Binding-arm multiplicity — the central construct

**Donor cluster.** Build a graph whose nodes are the molecule's potential donor atoms and whose
edges join two donors at shortest-path distance ≤ 4 bonds on the heavy-atom molecular graph. Its
connected components are the **chelating pockets**.

**Why 4 bonds.** A donor pair *d* bonds apart closes a *(d+2)*-membered chelate ring with the
metal. Four bonds is a six-membered ring, the largest commonly quoted as chelate-stabilised; three
bonds is the diglycolamide amide-O to ether-O separation, a five-membered ring. The value is fixed
from that chemistry and is not tuned.

**A first draft used 3 and was wrong.** At 3 bonds every malonamide (amide O to amide O is 4
bonds) and every P(=S)–CH₂–P(=S) dithiophosphinate (S to S is 4 bonds) was split into two
pockets, although both are textbook single-pocket bidentates closing a six-membered ring. The
error was found by inventorying the descriptor over all 183 structures and reading the topicity of
the malonamide family; **no target value and no model output was consulted**, and the correction
is logged in the specification file as version 1.1.0. Sensitivity, against fifteen structures whose
topicity is known from their chemistry:

| chelate link | multi-arm extractants | reference structures misassigned |
|---|---|---|
| 3 bonds | 63 | 4 — tetrabutylmalonamide, TWE-23, TWE-24, LL9 |
| **4 bonds (frozen)** | **43** | **1 — LL9** |
| 5 bonds | 31 | 1 — LL9 |

LL9 is a worked example rather than a defect. It is a bis-diglycolamide bridged by a
3-oxapentyl chain whose central ether oxygen is itself a potential donor and sits more than four
bonds from every other donor, so the rule gives it three pockets where a chemist counting
diglycolamide units would say two. The rule is applied uniformly and its answer is reported as it
stands.

Derived from the clusters: pocket count, largest and mean pocket size, pockets with 2 or 3 or more
donors, `estimated_denticity_per_pocket` (the largest pocket's donor count), `is_multitopic`, and
`topicity` capped at 6.

**Repeated arms.** Atom symmetry orbits come from `Chem.CanonicalRankAtoms(breakTies=False)`,
which gives constitutionally equivalent atoms equal rank. Each pocket is signed by the sorted
ranks of its donors; `repeated_arm_count` is the largest number of pockets sharing a signature,
and `n_distinct_arm_types` the number of distinct signatures. TODGA has one pocket and one arm
type; a bridged bis-diglycolamide has two pockets, both the same type; a tris-diglycolamide has
three.

**Core.** The atom minimising its maximum graph distance to any donor is the core. Its heavy
degree is `core_branch_degree`, and the mean donor distance to it is `mean_donor_distance_to_core`.

## 6. Donor topology

All shortest-path distances over the heavy-atom graph, from `Chem.GetDistanceMatrix`.

Pairwise donor-donor distance minimum, maximum, mean, median, standard deviation and network
diameter; the histogram as counts of pairs within 2, 3, 4, 5 and 6 bonds and beyond 6; the fraction
within 3; the count within the chelate link distance; counts of pairs that would close four- and
five-membered chelate rings; donor eccentricity mean, maximum and minimum; and, for multi-pocket
molecules, minimum and mean inter-pocket donor distance and the implied linker length.

## 7. Denticity proxies, named as proxies

True denticity is not inferable from a 2D graph and no label claiming it is produced. The proxies
are `potential_donor_count`, `local_donor_cluster_count`, `max_connected_donor_motif_size`,
`donor_pair_proximity_count`, `repeated_arm_count` and `estimated_denticity_per_pocket`.

For orientation rather than validation: TODGA gets 3 potential donors in 1 pocket, and TODGA is
the tridentate benchmark of this literature; a malonamide gets 2 in 1; a bridged bis-diglycolamide
gets 6 in 2; a tris-diglycolamide gets 9 in 3.

## 8. Architecture

Molecular weight, heavy-atom count, carbon count, heteroatom count, rotatable bonds, rings,
aromatic rings, fraction sp3, TPSA, cLogP, formal charge, HBD, HBA, mean heavy degree, counts of
degree-3 and degree-4 atoms, graph diameter, symmetry-orbit count, largest-orbit fraction, and the
ratios heavy atoms, carbons, molecular weight and cLogP **per potential donor** and heavy atoms
**per pocket**. These ratios are the "how much grease per binding site" axis; they overlap in
purpose with Gen12's generic descriptors and are kept because they are the ones the coordination
reading needs, not because they are new.

## 9. Numerical safety

Gen12's neural arms diverged in 8 of 25 folds because `lig2d__rd__Ipc` spans 1.1e8 to 3.8e29 and a
held-out value standardised to 7.3e16 under a chemotype hold-out, and the pre-registered
prediction clip then hid it. Recorded for this block before it was used:

- every value on all 183 structures is finite;
- the largest ratio of maximum value to standard deviation over the 109 informative columns is
  **20.58**;
- rebuilding from a shuffled input list is bit-identical.

The block is counts and short graph distances and cannot produce that failure, but the check is a
guard rather than a claim.

## 10. Reproducing the matrix

```bash
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_build_features.py
```

Writes `features/coordination_descriptors.parquet` (content hash `da609be8fcd5f260`),
`features/coordination_audit.json`, `features/coordination_columns.json` and
`manifests/multi_arm_subgroup.csv`.
