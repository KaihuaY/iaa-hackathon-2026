# Assumptions (clean-room build)

1. **Claim cost definition.** cost = incurred_to_date - recoveries (floored at 0); paid_to_date is already net of excess; for closed claims incurred == paid; for open claims the case estimate is included (incurred = paid + case reserve). No capping of large losses (largest single claim ~11.9M incurred is kept).
   - Source: "the final cost of all claims with a loss date in the row's period, net of the policy excess and net of recoveries" (Description); "paid_to_date: Amount paid so far, net of excess"; "incurred_to_date: Paid to date plus the current case estimate for open claims" (data-description)

2. **Repudiated claims excluded.** status == 'repudiated' claims are removed from both cost and claim counts.
   - Source: "excluding declined claims" (Description); "repudiated (declined)" (data-description)

3. **Attribution by loss date.** A claim belongs to the row with the same policy_id (in the table of the claim's line) whose [period_start, period_end] (both inclusive) contains loss_date. Report date is irrelevant to attribution. Claims with no covering row (2020 losses, post-cancellation gaps) are dropped.
   - Source: "A policy keeps the same policy_id across years, so match each claim to a policy row using its loss_date" (data-description); "last day covered" for period_end

4. **Duplicate claims removed.** Claims identical in all fields except claim_id are treated as duplicates and the first is kept (604).
   - Source: "Duplicate claims" in the data-quality register (COMPETITION.md)

5. **IBNR / development method.** Chain-ladder on a monthly reporting-delay count triangle (loss month x months to report), per line x peril, age-to-age factors from the volume-weighted pooled history of Jan-2021..Dec-2024 loss months, tail = 1 beyond 36 months. Each reported claim's cost is multiplied by the cumulative factor of its loss-month age at 31-Dec-2024. Count development is applied to cost (assumes severity of late-reported claims equals severity of reported ones). Valuation date 31-Dec-2024 (end of December). Open-claim case reserves are NOT developed (no transaction history available), i.e. case estimates assumed unbiased.
   - Source: "Some 2024 claims had not been reported by then, and open claims may still develop" (data-description); "Ultimate: The final cost of claims once all are reported and settled" (glossary)

6. **Exposure.** Supplied exposure kept when in (0,1]; otherwise replaced by min(1, (period_end-period_start+1)/365). Row-level model: label = target/exposure, weight = exposure; prediction = rate x exposure. Test exposure used as supplied (all in (0,1]).
   - Source: "exposure is the length of the row in years" (data-description); "exposure in days" in the register (COMPETITION.md); "Predictions are per row, so they scale with exposure"

7. **Customer-id repair.** Policy customer_id left-padded with zeros to 8 characters before joining customers (motor only needed it).
   - Source: "ids that lost their leading zeros" (COMPETITION.md); "Identifiers ... are text" (data-description)

8. **Weather join year.** Weather joined on address_id and the calendar year of period_start (same-year weather; 2025 weather exists for test).
   - Source: "weather: address_id x year" (data-description)

9. **Feature exclusions (leakage).** customer_service_contacts and renewal_decision are never used as features; IDs, surname, e-mail domain, marketing consent, document preference, campaign code, branch and calendar year also excluded. written_premium kept (present in test).
   - Source: "Leakage: customer_service_contacts (includes claim calls) and renewal_decision. Never use them as features." (COMPETITION.md)

10. **Sum-insured and construction repairs.** sum_insured divided by 1000 when > 5M (buildings) / > 1M (contents); construction_type lower-cased and codes w,t mapped to timber; impossible years_licensed (> age-16) set to NaN.
   - Source: "sum insured x1000, messy construction labels, impossible licence years" (COMPETITION.md); construction classes "brick, stone, timber, non-standard" (data-description)

11. **Modelling and CV.** One LightGBM Tweedie (power 1.5) model per line; 5 folds by crc32(customer_id) % 5 so customers (and thus their policies across years and lines) never straddle folds; early stopping on validation fold (mildly optimistic OOF); D2 computed on row-level total cost vs rate x exposure.
   - Source: "Leaderboard metric: Tweedie D2 (power 1.5)" (COMPETITION.md)

12. **Unadjusted items.** No trend/inflation adjustment across 2021-2024 to 2025; no cross-line features; no adjustment of 2024 test-year differences. Rows without a claim have target 0 even if IBNR exists for them (IBNR is carried only by reported claims' weights).
   - Source: Task statement: predict 2025 rows (COMPETITION.md)
