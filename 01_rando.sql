WITH MonthlyClaims AS (
    SELECT
        PROV_BIL_ID,
        EXTRACT(YEAR FROM LDOS_CLM) AS year,
        EXTRACT(MONTH FROM LDOS_CLM) AS month,
        COUNT(*) AS total_claims,
        SUM(CASE WHEN CPT_PROC_CD = '90837' THEN 1 ELSE 0 END) AS cpt_90837_claims
    FROM MV_ALL_PROF_FAST
    WHERE LDOS_CLM >= DATEADD(MONTH, -6, CURRENT_DATE())  -- Adjusts the date range to the last 6 months
    GROUP BY PROV_BIL_ID, EXTRACT(YEAR FROM LDOS_CLM), EXTRACT(MONTH FROM LDOS_CLM)
),
PercentageCPT90837 AS (
    SELECT
        PROV_BIL_ID,
        year,
        month,
        total_claims,
        cpt_90837_claims,
        (100.0 * cpt_90837_claims / total_claims) AS percentage_90837
    FROM MonthlyClaims
)

SELECT PROV_BIL_ID, year, month, percentage_90837
FROM PercentageCPT90837
WHERE percentage_90837 > 60  -- To find where more than 60% of claims are for 90837
ORDER BY PROV_BIL_ID, year, month;
