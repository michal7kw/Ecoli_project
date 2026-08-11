-- The Ecomics compendium as SQLite.
--
-- Design notes
--   * `condition` is the hub: every measurement links to it through the
--     canonical 4-tuple (strain, medium, stress, canonical GP). See db/canon.py
--     -- without canonicalization the three published GP dialects never join.
--   * Meta-data feature tables (strain_feature, medium_component) are stored
--     LONG rather than as 152/120 wide columns, so the ontology can grow and so
--     the feature encoder can query it generically.
--   * Measurements are long too. The wide matrices the model actually trains on
--     are mirrored to Parquet by db/build.py -- SQLite is for provenance,
--     integrity and ad-hoc querying, Parquet is for throughput.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- meta-data

CREATE TABLE IF NOT EXISTS strain (
    name             TEXT PRIMARY KEY,
    link             TEXT,
    pmid             TEXT,
    alternate_names  TEXT
);

-- Long form of the 152 genotype/phenotype markers from strain.v5.json.
-- value is the raw cell ('no', 'relA1', 'F-', 'Lambda-', ...); `present` is the
-- boolean reading used by the feature encoder.
CREATE TABLE IF NOT EXISTS strain_feature (
    strain   TEXT NOT NULL REFERENCES strain(name),
    feature  TEXT NOT NULL,
    value    TEXT,
    present  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (strain, feature)
);

CREATE TABLE IF NOT EXISTS medium (
    medium_id    TEXT PRIMARY KEY,
    base_medium  TEXT,
    description  TEXT,
    link         TEXT,
    pmid         TEXT,
    defined      INTEGER
);

-- Long form of the 120 chemical components from medium.v5.json.
-- raw is the published string ('328mM', '0.40%', 'yes', '?'); amount/unit are
-- the parsed numeric reading, NULL where unparseable.
CREATE TABLE IF NOT EXISTS medium_component (
    medium_id  TEXT NOT NULL REFERENCES medium(medium_id),
    component  TEXT NOT NULL,
    raw        TEXT,
    amount     REAL,
    unit       TEXT,
    present    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (medium_id, component)
);

CREATE TABLE IF NOT EXISTS stress (
    name      TEXT PRIMARY KEY,
    n_profiles INTEGER DEFAULT 0
);

-- One row per distinct (gene, type) perturbation seen anywhere.
CREATE TABLE IF NOT EXISTS perturbation (
    id     TEXT PRIMARY KEY,        -- canonical 'b0756(KO)'
    gene   TEXT NOT NULL,
    type   TEXT NOT NULL            -- KO / OE / IN / VAR / RM / EX
);

-- ---------------------------------------------------------------- conditions

CREATE TABLE IF NOT EXISTS condition (
    id          INTEGER PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,   -- 'MG1655.MD018.none.none'
    strain      TEXT NOT NULL,
    medium_id   TEXT NOT NULL,
    stress      TEXT NOT NULL,
    gp          TEXT NOT NULL,          -- canonical, 'none' for wild type
    is_wildtype INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_condition_parts
    ON condition (strain, medium_id, stress, gp);

-- Many-to-many: which perturbations a condition carries.
CREATE TABLE IF NOT EXISTS condition_perturbation (
    condition_id   INTEGER NOT NULL REFERENCES condition(id),
    perturbation_id TEXT   NOT NULL REFERENCES perturbation(id),
    PRIMARY KEY (condition_id, perturbation_id)
);

-- ---------------------------------------------------------------- molecules

CREATE TABLE IF NOT EXISTS molecule (
    id          TEXT PRIMARY KEY,   -- b-number, KEGG C-id, HMDB id, ...
    kind        TEXT NOT NULL,      -- Gene / Protein / Metabolite
    name        TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS reaction (
    id          TEXT PRIMARY KEY,   -- R0001 .. R0120
    description TEXT,
    bigg        TEXT
);

-- ---------------------------------------------------------------- profiles

-- A profile is one measured sample. Transcriptome profiles are individual
-- (3,578 of them); the public proteome/metabolome/fluxome/phenome tables are
-- already condition-averaged, so there one profile == one condition.
CREATE TABLE IF NOT EXISTS profile (
    id            INTEGER PRIMARY KEY,
    source_id     TEXT,                     -- e.g. 'T0568', NULL if averaged
    layer         TEXT NOT NULL,            -- transcriptome/proteome/...
    condition_id  INTEGER NOT NULL REFERENCES condition(id),
    is_averaged   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_profile_layer     ON profile (layer);
CREATE INDEX IF NOT EXISTS ix_profile_condition ON profile (condition_id);

-- ---------------------------------------------------------------- measurements

CREATE TABLE IF NOT EXISTS measurement (
    profile_id  INTEGER NOT NULL REFERENCES profile(id),
    molecule_id TEXT NOT NULL,
    value       REAL,
    PRIMARY KEY (profile_id, molecule_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS flux (
    profile_id  INTEGER NOT NULL REFERENCES profile(id),
    reaction_id TEXT NOT NULL REFERENCES reaction(id),
    value       REAL,
    PRIMARY KEY (profile_id, reaction_id)
) WITHOUT ROWID;

-- The three phenotypes extracted by the paper's automated growth-curve script:
-- lag time, maximum growth rate, final optical density.
CREATE TABLE IF NOT EXISTS phenotype (
    profile_id   INTEGER PRIMARY KEY REFERENCES profile(id),
    lag_time     REAL,
    growth_rate  REAL,
    final_od     REAL
);

-- ---------------------------------------------------------------- provenance

CREATE TABLE IF NOT EXISTS source (
    name        TEXT PRIMARY KEY,
    path        TEXT,
    sha256      TEXT,
    n_records   INTEGER,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS build_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ---------------------------------------------------------------- views

-- Which layers each condition has, for cross-layer queries.
CREATE VIEW IF NOT EXISTS condition_layers AS
SELECT c.id, c.key, c.strain, c.medium_id, c.stress, c.gp,
       MAX(p.layer = 'transcriptome') AS has_transcriptome,
       MAX(p.layer = 'proteome')      AS has_proteome,
       MAX(p.layer = 'metabolome')    AS has_metabolome,
       MAX(p.layer = 'fluxome')       AS has_fluxome,
       MAX(p.layer = 'phenome')       AS has_phenome,
       COUNT(DISTINCT p.id)           AS n_profiles
FROM condition c
LEFT JOIN profile p ON p.condition_id = c.id
GROUP BY c.id;

-- Per-layer profile and condition counts.
CREATE VIEW IF NOT EXISTS layer_summary AS
SELECT layer,
       COUNT(*)                     AS n_profiles,
       COUNT(DISTINCT condition_id) AS n_conditions
FROM profile
GROUP BY layer;
