-- Relational schema for the clean data of Stages 1 to 3 (SQLite).
--
-- One row per respondent, one row per scale item answer, one row per choice
-- task and one row per profile shown, linked by keys. Lookup tables hold the
-- universities, scale items, conjoint attributes and their levels, so every
-- coded value in the data tables has a documented meaning.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Lookup tables
-- ---------------------------------------------------------------------------

CREATE TABLE university (
    university_id   INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    sector          TEXT NOT NULL CHECK (sector IN ('Public', 'Private'))
);

CREATE TABLE scale_item (
    item_code       TEXT PRIMARY KEY,
    scale           TEXT NOT NULL CHECK (scale IN ('Diversity and inclusion climate',
                                                   'Expected procedural justice',
                                                   'Attention check')),
    component       TEXT
);

CREATE TABLE attribute (
    attribute_code  TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    attribute_type  TEXT NOT NULL CHECK (attribute_type IN ('Governance', 'Efficiency')),
    display_order   INTEGER NOT NULL UNIQUE
);

CREATE TABLE attribute_level (
    attribute_code  TEXT NOT NULL REFERENCES attribute (attribute_code),
    level           INTEGER NOT NULL CHECK (level IN (1, 2, 3)),
    label           TEXT NOT NULL,
    PRIMARY KEY (attribute_code, level)
);

-- ---------------------------------------------------------------------------
-- Respondents (analytic sample, N = 317)
-- ---------------------------------------------------------------------------

CREATE TABLE respondent (
    response_id         TEXT PRIMARY KEY,
    university_id       INTEGER NOT NULL REFERENCES university (university_id),
    role                TEXT NOT NULL CHECK (role IN ('Academic staff', 'Professional staff', 'Leaders')),
    gender              TEXT NOT NULL,
    age_band            TEXT NOT NULL,
    education           TEXT NOT NULL,
    college_field       TEXT,
    recruit_exp         TEXT NOT NULL,
    ai_experience       TEXT NOT NULL,
    duration_sec        INTEGER NOT NULL CHECK (duration_sec > 0),
    fast_responder      INTEGER NOT NULL CHECK (fast_responder IN (0, 1)),
    straightline_both   INTEGER NOT NULL CHECK (straightline_both IN (0, 1)),
    regression_sample   INTEGER NOT NULL CHECK (regression_sample IN (0, 1)),
    climate_12          REAL NOT NULL CHECK (climate_12 BETWEEN 1 AND 5),
    climate_org         REAL NOT NULL CHECK (climate_org BETWEEN 1 AND 5),
    climate_pers        REAL NOT NULL CHECK (climate_pers BETWEEN 1 AND 5),
    pj                  REAL NOT NULL CHECK (pj BETWEEN 1 AND 5),
    pj_z                REAL NOT NULL,
    pj_group            TEXT NOT NULL CHECK (pj_group IN ('Lower expected PJ', 'Higher expected PJ')),
    design_version      INTEGER NOT NULL
);

CREATE TABLE item_response (
    response_id     TEXT NOT NULL REFERENCES respondent (response_id),
    item_code       TEXT NOT NULL REFERENCES scale_item (item_code),
    value           INTEGER NOT NULL CHECK (value BETWEEN 1 AND 5),
    PRIMARY KEY (response_id, item_code)
);

-- ---------------------------------------------------------------------------
-- Conjoint experiment
-- ---------------------------------------------------------------------------

-- Row of the profile table in which each attribute was listed, fixed for
-- each respondent across the nine tasks
CREATE TABLE attribute_position (
    response_id     TEXT NOT NULL REFERENCES respondent (response_id),
    attribute_code  TEXT NOT NULL REFERENCES attribute (attribute_code),
    row_position    INTEGER NOT NULL CHECK (row_position BETWEEN 1 AND 4),
    PRIMARY KEY (response_id, attribute_code),
    UNIQUE (response_id, row_position)
);

CREATE TABLE choice_task (
    response_id     TEXT NOT NULL REFERENCES respondent (response_id),
    task            INTEGER NOT NULL CHECK (task BETWEEN 1 AND 9),
    chosen_profile  INTEGER NOT NULL CHECK (chosen_profile IN (1, 2)),
    PRIMARY KEY (response_id, task)
);

CREATE TABLE profile (
    response_id     TEXT NOT NULL,
    task            INTEGER NOT NULL,
    profile         INTEGER NOT NULL CHECK (profile IN (1, 2)),
    privacy         INTEGER NOT NULL CHECK (privacy IN (1, 2, 3)),
    transparency    INTEGER NOT NULL CHECK (transparency IN (1, 2, 3)),
    usefulness      INTEGER NOT NULL CHECK (usefulness IN (1, 2, 3)),
    ease            INTEGER NOT NULL CHECK (ease IN (1, 2, 3)),
    PRIMARY KEY (response_id, task, profile),
    FOREIGN KEY (response_id, task) REFERENCES choice_task (response_id, task)
);

CREATE INDEX idx_respondent_university ON respondent (university_id);
CREATE INDEX idx_item_response_item ON item_response (item_code);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Every profile with the outcome: 1 if chosen, 0 if not
CREATE VIEW profile_choice AS
SELECT p.*,
       CASE WHEN p.profile = t.chosen_profile THEN 1 ELSE 0 END AS chosen
FROM profile AS p
JOIN choice_task AS t USING (response_id, task);

-- One row per profile and attribute, with the level label
CREATE VIEW profile_attribute_level AS
SELECT pc.response_id, pc.task, pc.profile, pc.chosen,
       al.attribute_code, al.level, al.label
FROM (
    SELECT response_id, task, profile, chosen, 'privacy' AS attribute_code, privacy AS level FROM profile_choice
    UNION ALL
    SELECT response_id, task, profile, chosen, 'transparency', transparency FROM profile_choice
    UNION ALL
    SELECT response_id, task, profile, chosen, 'usefulness', usefulness FROM profile_choice
    UNION ALL
    SELECT response_id, task, profile, chosen, 'ease', ease FROM profile_choice
) AS pc
JOIN attribute_level AS al USING (attribute_code, level);

-- Respondents with their university and sector
CREATE VIEW respondent_profile AS
SELECT r.*, u.name AS university, u.sector
FROM respondent AS r
JOIN university AS u USING (university_id);
