-- Example queries. Each reproduces a figure reported in Chapters 3 to 5.

-- name: Analytic sample by stakeholder role and university sector (Table 5.1)
SELECT role,
       SUM(sector = 'Public')  AS public,
       SUM(sector = 'Private') AS private,
       COUNT(*)                AS total
FROM respondent_profile
GROUP BY role
ORDER BY total DESC;

-- name: Mean expected procedural justice and climate by role (Chapter 3)
SELECT role,
       COUNT(*)                   AS n,
       ROUND(AVG(pj), 3)          AS mean_expected_pj,
       ROUND(AVG(climate_12), 3)  AS mean_climate
FROM respondent
GROUP BY role
ORDER BY mean_expected_pj;

-- name: Marginal mean of every attribute level (Table 4.6)
SELECT a.name                AS attribute,
       pal.label             AS level,
       COUNT(*)              AS profiles,
       ROUND(AVG(pal.chosen), 3) AS marginal_mean
FROM profile_attribute_level AS pal
JOIN attribute AS a USING (attribute_code)
GROUP BY a.display_order, pal.level
ORDER BY a.display_order, pal.level;

-- name: Share of tasks in which the left profile was chosen (Table 4.2)
SELECT COUNT(*) AS n, ROUND(100.0 * AVG(chosen_profile = 1), 1) AS left_chosen_pct
FROM choice_task;

-- name: Tasks trading governance against efficiency by the same number of levels (Chapter 4, complete systems and trade-offs)
WITH task_gap AS (
    SELECT p1.response_id, p1.task, t.chosen_profile,
           (p1.privacy + p1.transparency) - (p2.privacy + p2.transparency) AS governance_gap,
           (p1.usefulness + p1.ease) - (p2.usefulness + p2.ease)           AS efficiency_gap
    FROM profile AS p1
    JOIN profile AS p2 ON p2.response_id = p1.response_id AND p2.task = p1.task AND p2.profile = 2
    JOIN choice_task AS t ON t.response_id = p1.response_id AND t.task = p1.task
    WHERE p1.profile = 1
)
SELECT COUNT(*) AS tasks,
       ROUND(100.0 * AVG(CASE WHEN governance_gap > 0 THEN chosen_profile = 1
                              ELSE chosen_profile = 2 END), 1) AS governance_side_chosen_pct
FROM task_gap
WHERE governance_gap <> 0 AND governance_gap = -efficiency_gap;

-- name: Marginal means of algorithmic transparency by stakeholder role (Figure 5.2)
SELECT pal.label AS transparency,
       ROUND(AVG(CASE WHEN r.role = 'Academic staff'     THEN pal.chosen END), 3) AS academic_staff,
       ROUND(AVG(CASE WHEN r.role = 'Professional staff' THEN pal.chosen END), 3) AS professional_staff,
       ROUND(AVG(CASE WHEN r.role = 'Leaders'            THEN pal.chosen END), 3) AS leaders
FROM profile_attribute_level AS pal
JOIN respondent AS r USING (response_id)
WHERE pal.attribute_code = 'transparency'
GROUP BY pal.level
ORDER BY pal.level;

-- name: Item means of the expected procedural justice scale (Chapter 3)
SELECT ir.item_code,
       ROUND(AVG(ir.value), 2) AS mean,
       MIN(ir.value)           AS min,
       MAX(ir.value)           AS max
FROM item_response AS ir
JOIN scale_item AS si USING (item_code)
WHERE si.scale = 'Expected procedural justice'
GROUP BY ir.item_code
ORDER BY ir.item_code;
