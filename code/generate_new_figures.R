# generate_new_figures.R
# Reproduces the 9 figures in ../new_figures_2026-02-27/ at 600 dpi.
# Run from the Data-Drift/ directory:  Rscript code/generate_new_figures.R

suppressPackageStartupMessages({
  need <- c("arrow","data.table","ggplot2","pROC","ragg","scales","patchwork","stringr")
  miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")
  lapply(need, library, character.only = TRUE)
})

set.seed(42)
N_BOOT <- 1000

DATA   <- "data/external"
OUTDIR <- "figures"
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- helpers ----

age_bin <- function(a) {
  cut(a, breaks = c(-Inf, 15, 44, 64, 79, Inf),
      labels = c(NA, "16-44", "45-64", "65-79", "80+"))
}

eth_norm <- function(x) {
  x <- toupper(as.character(x))
  fcase(
    grepl("WHITE", x), "White",
    grepl("BLACK|AFRICAN", x), "Black",
    grepl("HISP|LATIN", x), "Hispanic",
    grepl("ASIAN", x), "Asian",
    default = NA_character_
  )
}

gender_norm <- function(g) {
  g <- as.character(g)
  fcase(
    g %in% c("M","Male","male","0","FALSE"), "Male",
    g %in% c("F","Female","female","1","TRUE"), "Female",
    default = NA_character_
  )
}

# Stratified bootstrap AUROC (preserves class balance per resample).
boot_auc <- function(y, p, n_boot = N_BOOT) {
  y <- as.integer(y); p <- as.numeric(p)
  ok <- !is.na(y) & !is.na(p)
  y <- y[ok]; p <- p[ok]
  if (length(unique(y)) < 2 || length(y) < 10) return(c(NA_real_, NA_real_, NA_real_))
  pos <- which(y == 1); neg <- which(y == 0)
  if (length(pos) < 3 || length(neg) < 3) return(c(NA_real_, NA_real_, NA_real_))
  point <- as.numeric(pROC::auc(pROC::roc(y, p, quiet = TRUE, direction = "<")))
  draws <- replicate(n_boot, {
    idx <- c(sample(pos, replace = TRUE), sample(neg, replace = TRUE))
    yy <- y[idx]; pp <- p[idx]
    if (length(unique(yy)) < 2) return(NA_real_)
    as.numeric(pROC::auc(pROC::roc(yy, pp, quiet = TRUE, direction = "<")))
  })
  ci <- quantile(draws, c(0.025, 0.975), na.rm = TRUE)
  c(point, ci[[1]], ci[[2]])
}

# DeLong p-value (paired predictors not relevant here -> two-sample via roc.test on independent samples).
delong_p <- function(y1, p1, y2, p2) {
  ok1 <- !is.na(y1) & !is.na(p1); ok2 <- !is.na(y2) & !is.na(p2)
  if (sum(ok1) < 10 || sum(ok2) < 10) return(NA_real_)
  if (length(unique(y1[ok1])) < 2 || length(unique(y2[ok2])) < 2) return(NA_real_)
  r1 <- pROC::roc(y1[ok1], p1[ok1], quiet = TRUE, direction = "<")
  r2 <- pROC::roc(y2[ok2], p2[ok2], quiet = TRUE, direction = "<")
  tryCatch(pROC::roc.test(r1, r2, method = "delong")$p.value, error = function(e) NA_real_)
}

save_png <- function(p, file, w, h) {
  ragg::agg_png(file.path(OUTDIR, file), width = w, height = h,
                units = "in", res = 600, scaling = 1)
  print(p); dev.off()
  message("wrote ", file)
}

ETH_LEVELS <- c("White","Black","Hispanic","Asian")
AGE_LEVELS <- c("16-44","45-64","65-79","80+")
PER_MIMIC  <- c("2008 - 2010","2011 - 2013","2014 - 2016","2017 - 2019","2020 - 2022")
PER_EICU   <- c("2014","2015","2020","2021")

eth_pal <- c(White="#1f77b4", Black="#2ca02c", Hispanic="#ff7f0e", Asian="#d62728")
gen_pal <- c(Male="#4C72B0", Female="#DD8452")

# =========================================================== SOFA Figure 1D ==
# Source: data/external/sofa_bias/mimiciv_ml-scores_bias.csv
# y = death_hosp, score = sofa, time = anchor_year_group.

sofa <- as.data.table(arrow::read_csv_arrow(
  file.path(DATA, "sofa_bias/mimiciv_ml-scores_bias.csv")))
sofa <- sofa[!is.na(sofa) & !is.na(death_hosp) & !is.na(anchor_year_group)]

# The SOFA bias CSV collapsed ethnicity into 5 buckets that drop Hispanic.
# Re-derive fine-grained ethnicity from the sepsis parquet
# (Patient_ID == stay_id for MIMIC) so Hispanic is preserved.
sepsis_eth <- as.data.table(arrow::read_parquet(
  file.path(DATA, "sepsis_prediction/mimiciv_model_shift_sepsis.parquet"),
  col_select = c("Patient_ID","ethnicity")))
sepsis_eth <- unique(sepsis_eth, by = "Patient_ID")
sofa <- merge(sofa, sepsis_eth, by.x = "stay_id", by.y = "Patient_ID",
              all.x = TRUE, suffixes = c("_bias","_fine"))
sofa[, ethnicity_use := fcoalesce(ethnicity_fine, ethnicity_bias)]

sofa[, AgeGroup  := as.character(age_bin(age))]
sofa[, Race      := eth_norm(ethnicity_use)]
sofa[, Gender    := fifelse(gender == "M", "Male",
                     fifelse(gender == "F", "Female", NA_character_))]
sofa[, Period    := factor(anchor_year_group, levels = PER_MIMIC)]

# Use the four periods that have all subgroups well-populated (matches reference figs).
SOFA_PERIODS <- c("2008 - 2010","2011 - 2013","2014 - 2016","2017 - 2019")
sofa <- sofa[Period %in% SOFA_PERIODS][, Period := droplevels(Period)]

sofa_subgroup_auc <- function(group_col, group_levels) {
  rows <- list()
  for (lvl in group_levels) {
    for (per in SOFA_PERIODS) {
      d <- sofa[get(group_col) == lvl & Period == per]
      a <- boot_auc(d$death_hosp, d$sofa)
      rows[[length(rows)+1]] <- data.table(group_type = group_col, group = lvl,
                                           period = per, n = nrow(d),
                                           auc = a[1], lo = a[2], hi = a[3])
    }
  }
  rbindlist(rows)
}

sofa_race   <- sofa_subgroup_auc("Race",     ETH_LEVELS)
sofa_gender <- sofa_subgroup_auc("Gender",   c("Male","Female"))
sofa_age    <- sofa_subgroup_auc("AgeGroup", AGE_LEVELS)

# ------------------------------- fig1D_sofa_full_temporal_trajectory.png ----
panel_traj <- function(df, title, pal) {
  df <- copy(df)[!is.na(auc)]
  df[, period := factor(period, levels = SOFA_PERIODS)]
  ggplot(df, aes(period, auc, group = group, color = group, fill = group)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, color = NA) +
    geom_line(linewidth = 0.6) + geom_point(size = 2) +
    scale_color_manual(values = pal, name = NULL) +
    scale_fill_manual(values = pal, guide = "none") +
    coord_cartesian(ylim = c(0.45, 0.80)) +
    labs(title = title, x = "Year Period", y = "AUROC (SOFA score)") +
    theme_bw(base_size = 9) +
    theme(legend.position = c(0.02, 0.02), legend.justification = c(0,0),
          legend.background = element_rect(fill = alpha("white", 0.7), color = NA),
          legend.key.size = unit(0.4, "cm"),
          axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0.5))
}

age_pal <- c(`16-44`="#d62728",`45-64`="#2ca02c",`65-79`="#ff7f0e",`80+`="#9467bd")
p1 <- panel_traj(sofa_race,   "Race/Ethnicity", eth_pal) |
      panel_traj(sofa_gender, "Gender",         c(Male="#1f77b4", Female="#ff7f0e")) |
      panel_traj(sofa_age,    "Age Group",      age_pal)
p1 <- p1 + plot_annotation(
  title = "MIMIC (Demo Data): SOFA Score AUROC — Full Temporal Trajectory",
  theme = theme(plot.title = element_text(hjust = 0.5, size = 11)))
save_png(p1, "fig1D_sofa_full_temporal_trajectory.png", 12, 3.5)

# ------------------------------------ fig1D_sofa_trajectory_heatmap.png ----
heat_df <- rbind(
  sofa_race[,   .(label = paste0("Race: ",   group),   period, auc)],
  sofa_gender[, .(label = paste0("Gender: ", group),   period, auc)],
  sofa_age[,    .(label = paste0("Age: ",    group),   period, auc)]
)[!is.na(auc)]
ord <- heat_df[, .(m = mean(auc, na.rm = TRUE)), by = label][order(-m), label]
heat_df[, label  := factor(label,  levels = ord)]
heat_df[, period := factor(period, levels = SOFA_PERIODS)]

p_hm <- ggplot(heat_df, aes(period, label, fill = auc)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.3f", auc)), size = 3) +
  scale_fill_gradientn(colours = c("#a50026","#f46d43","#fdae61","#fee08b",
                                   "#d9ef8b","#a6d96a","#1a9850"),
                       limits = c(0.50, 0.85), oob = scales::squish, name = "AUROC") +
  labs(title = "MIMIC (Demo Data): SOFA AUROC Trajectory — All Subgroups × All Year Periods",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 11))
save_png(p_hm, "fig1D_sofa_trajectory_heatmap.png", 8, 6.5)

# ---------------------------- fig1D_sofa_consecutive_yeargroup_drift.png ----
PAIRS <- list(c("2008 - 2010","2011 - 2013"),
              c("2011 - 2013","2014 - 2016"),
              c("2014 - 2016","2017 - 2019"))

drift_rows <- list()
for (gc in c("Race","Gender","AgeGroup")) {
  levels_use <- switch(gc, Race = ETH_LEVELS, Gender = c("Male","Female"), AgeGroup = AGE_LEVELS)
  for (pp in PAIRS) {
    for (lvl in levels_use) {
      d1 <- sofa[get(gc) == lvl & Period == pp[1]]
      d2 <- sofa[get(gc) == lvl & Period == pp[2]]
      a1 <- boot_auc(d1$death_hosp, d1$sofa)[1]
      a2 <- boot_auc(d2$death_hosp, d2$sofa)[1]
      pv <- delong_p(d1$death_hosp, d1$sofa, d2$death_hosp, d2$sofa)
      drift_rows[[length(drift_rows)+1]] <- data.table(
        group_type = gc, group = lvl,
        pair_lbl = paste(sub(" - .*","",pp[1]), "→", sub(" - .*","",pp[2])),
        delta = a2 - a1, p = pv)
    }
  }
}
drift <- rbindlist(drift_rows)
drift[, sig_dir := fifelse(is.na(p) | p >= 0.05, "ns",
                    fifelse(delta > 0, "up", "down"))]
drift[, group_type := factor(group_type, levels = c("Race","Gender","AgeGroup"),
                             labels = c("Race","Gender","Age"))]
drift[, group := factor(group, levels = c(rev(ETH_LEVELS), rev(c("Male","Female")), rev(AGE_LEVELS)))]

p_drift <- ggplot(drift, aes(delta, group, fill = sig_dir)) +
  geom_vline(xintercept = 0, color = "black", linewidth = 0.4) +
  geom_col(width = 0.7) +
  facet_grid(group_type ~ pair_lbl, scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = c(ns = "grey60", up = "#2ca02c", down = "#d62728"),
                    labels = c(ns = "Non-significant",
                               up = "Sig. increase (p<0.05)",
                               down = "Sig. decrease (p<0.05)"),
                    breaks = c("up","down","ns"),
                    name = NULL) +
  scale_x_continuous(limits = c(-0.15, 0.15), breaks = c(-0.10, 0, 0.10),
                     labels = c("-0.10","0","0.10")) +
  labs(title = "MIMIC (Demo Data): SOFA Drift — Consecutive Year-Period Comparisons",
       subtitle = "(* = p<0.05, green=improvement, red=decline)",
       x = expression(Delta * "AUROC"), y = NULL) +
  theme_bw(base_size = 9) +
  theme(strip.placement = "outside",
        strip.background.y = element_rect(fill = "grey95", color = NA),
        strip.background.x = element_rect(fill = "white", color = NA),
        strip.text.x = element_text(face = "bold"),
        legend.position = "bottom",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 8))
save_png(p_drift, "fig1D_sofa_consecutive_yeargroup_drift.png", 8, 9)

# ============================================ Sepsis intersectional figures ==

build_sepsis <- function(pat_path, demo_path, demo_join, period_col, periods, dataset_lbl) {
  pat <- as.data.table(arrow::read_csv_arrow(pat_path))
  setnames(pat, "patient_id", "Patient_ID")
  if (grepl("\\.parquet$", demo_path)) {
    demo <- as.data.table(arrow::read_parquet(demo_path,
              col_select = c("Patient_ID","Age","Gender","ethnicity","anchor_year_group")))
    demo <- unique(demo, by = "Patient_ID")
  } else {
    demo <- as.data.table(arrow::read_csv_arrow(demo_path))
  }
  setnames(demo, demo_join$id_col, "Patient_ID")
  d <- merge(pat, demo, by = "Patient_ID")
  d[, Race     := eth_norm(ethnicity)]
  d[, Gender   := gender_norm(Gender)]
  d[, AgeGroup := as.character(age_bin(as.numeric(Age)))]
  d[, Period   := factor(get(period_col), levels = as.character(periods))]
  d <- d[!is.na(Race) & !is.na(Gender) & !is.na(AgeGroup) & !is.na(Period)]
  d[, dataset := dataset_lbl]
  d[, .(Patient_ID, true_sepsis, mean_prob, Race, Gender, AgeGroup, Period, dataset)]
}

mimic <- build_sepsis(
  pat_path   = file.path(DATA, "patient_results/mimiciv_patient_results.csv"),
  demo_path  = file.path(DATA, "sepsis_prediction/mimiciv_model_shift_sepsis.parquet"),
  demo_join  = list(id_col = "Patient_ID"),
  period_col = "anchor_year_group", periods = PER_MIMIC, dataset_lbl = "MIMIC-IV")

eicu_old <- build_sepsis(
  file.path(DATA, "patient_results/eicu_2014_2015_patient_results.csv"),
  file.path(DATA, "sepsis_prediction/eicu_model_shift_sepsis_static.csv"),
  list(id_col = "Patient_ID"),
  "hospitaldischargeyear", PER_EICU, "eICU 2014-2015")

eicu_new <- build_sepsis(
  file.path(DATA, "patient_results/eicu_2020_2021_patient_results.csv"),
  file.path(DATA, "sepsis_prediction/eicu_new_model_shift_sepsis_static.csv"),
  list(id_col = "Patient_ID"),
  "hospitaldischargeyear", PER_EICU, "eICU 2020-2021")

eicu <- rbind(eicu_old, eicu_new)[, Period := factor(as.character(Period), levels = PER_EICU)]

intersect_auc <- function(d, periods) {
  rows <- list()
  combos <- unique(d[, .(Race, AgeGroup, Gender)])
  for (i in seq_len(nrow(combos))) {
    rc <- combos$Race[i]; ag <- combos$AgeGroup[i]; gd <- combos$Gender[i]
    for (per in periods) {
      sub <- d[Race == rc & AgeGroup == ag & Gender == gd & Period == per]
      if (nrow(sub) < 30) next
      a <- boot_auc(sub$true_sepsis, sub$mean_prob)
      rows[[length(rows)+1]] <- data.table(Race = rc, AgeGroup = ag, Gender = gd,
                                           Period = per, n = nrow(sub),
                                           auc = a[1], lo = a[2], hi = a[3])
    }
  }
  rbindlist(rows)
}

mimic_int <- intersect_auc(mimic, PER_MIMIC)
eicu_int  <- intersect_auc(eicu,  PER_EICU)

# ----------------------------------------- fig_intersectional_heatmap_*.png ----
heatmap_intersectional <- function(df, periods, title, fname) {
  df <- copy(df)[!is.na(auc)]
  df[, label := paste0(Race, " / ", AgeGroup, " / ", Gender)]
  ord <- df[, .(m = mean(auc, na.rm = TRUE)), by = label][order(-m), label]
  df[, label  := factor(label, levels = rev(ord))]
  df[, Period := factor(as.character(Period), levels = as.character(periods))]
  p <- ggplot(df, aes(Period, label, fill = auc)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.3f", auc)), size = 2.6) +
    scale_fill_gradientn(colours = c("#a50026","#f46d43","#fdae61","#fee08b",
                                     "#d9ef8b","#a6d96a","#1a9850"),
                         limits = c(0.45, 0.90), oob = scales::squish, name = "AUROC") +
    labs(title = title,
         subtitle = "(Patient-level, mean_prob · 95% CI · ordered by mean AUROC)",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0.5, size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8))
  save_png(p, fname, 7, max(5, 0.22 * length(levels(df$label)) + 1.5))
}

heatmap_intersectional(mimic_int, PER_MIMIC,
  "MIMIC-IV: Intersectional AUROC — Ethnicity × Age × Gender",
  "fig_intersectional_heatmap_mimic.png")
heatmap_intersectional(eicu_int, PER_EICU,
  "eICU Combined: Intersectional AUROC — Ethnicity × Age × Gender",
  "fig_intersectional_heatmap_eicu.png")

# --------------------------------------- fig_intersectional_linegraph_*.png ----
linegraph_intersectional <- function(df, periods, title, fname) {
  df <- copy(df)[!is.na(auc)]
  df[, Period := factor(as.character(Period), levels = as.character(periods))]
  df[, AgeGroup := factor(AgeGroup, levels = AGE_LEVELS)]
  df[, Race := factor(Race, levels = ETH_LEVELS)]
  df[, group_id := paste(Race, AgeGroup, Gender, sep = "|")]
  age_lty  <- c(`16-44`="solid",`45-64`="22",`65-79`="42",`80+`="12")
  gen_alpha <- c(Male = 0.92, Female = 0.52)
  race_pal  <- eth_pal
  p <- ggplot(df, aes(Period, auc, group = group_id, color = Race,
                      linetype = AgeGroup, alpha = Gender, fill = Race)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), color = NA, alpha = 0.10) +
    geom_line(linewidth = 0.55) + geom_point(size = 1.4) +
    scale_color_manual(values = race_pal, drop = FALSE, name = "Ethnicity") +
    scale_fill_manual(values = race_pal,  drop = FALSE, guide = "none") +
    scale_linetype_manual(values = age_lty, drop = FALSE, name = "Age group") +
    scale_alpha_manual(values = gen_alpha, name = "Gender") +
    facet_wrap(~ Race, ncol = 2) +
    coord_cartesian(ylim = c(0.30, 1.00)) +
    labs(title = title,
         subtitle = "(Patient-level mean probability · 95% CI shading · Line style = Age · Opacity = Gender)",
         x = "Time Period", y = "AUROC (patient-level, mean_prob)") +
    theme_bw(base_size = 9) +
    theme(legend.position = "bottom", legend.box = "horizontal",
          strip.background = element_rect(fill = "white", color = NA),
          strip.text = element_text(face = "bold"),
          axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0.5, size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8))
  save_png(p, fname, 11, 7.5)
}

linegraph_intersectional(mimic_int, PER_MIMIC,
  "MIMIC-IV (2008-2022): Intersectional Drift — AUROC by Time Period",
  "fig_intersectional_linegraph_mimic.png")
linegraph_intersectional(eicu_int, PER_EICU,
  "eICU Combined (2014-2021): Intersectional Drift — AUROC by Time Period",
  "fig_intersectional_linegraph_eicu.png")

# ----------------------------------------- fig_intersectional_summary_*.png ----
summary_intersectional <- function(df, periods, title, fname) {
  df <- copy(df)[!is.na(auc)]
  df[, Period := factor(as.character(Period), levels = as.character(periods))]
  df[, AgeGroup := factor(AgeGroup, levels = AGE_LEVELS)]
  df[, Race := factor(Race, levels = ETH_LEVELS)]
  df[, group_id := paste(Race, AgeGroup, Gender, sep = "|")]
  age_lty   <- c(`16-44`="solid",`45-64`="22",`65-79`="42",`80+`="12")
  gen_alpha <- c(Male = 0.95, Female = 0.55)
  p <- ggplot(df, aes(Period, auc, group = group_id, color = Race,
                      linetype = AgeGroup, alpha = Gender)) +
    geom_line(linewidth = 0.55) + geom_point(size = 1.4) +
    scale_color_manual(values = eth_pal, drop = FALSE, name = "Ethnicity") +
    scale_linetype_manual(values = age_lty, drop = FALSE, name = "Age group") +
    scale_alpha_manual(values = gen_alpha, name = "Gender") +
    coord_cartesian(ylim = c(0.40, 1.00)) +
    labs(title = title,
         subtitle = "(Color=Ethnicity · Line style=Age · Opacity=Gender)",
         x = "Time Period", y = "AUROC (patient-level)") +
    theme_bw(base_size = 9) +
    theme(legend.position = "right",
          legend.box = "vertical",
          axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0.5, size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8))
  save_png(p, fname, 9, 5)
}

summary_intersectional(mimic_int, PER_MIMIC,
  "MIMIC-IV (2008-2022): Intersectional AUROC Drift",
  "fig_intersectional_summary_mimic.png")
summary_intersectional(eicu_int, PER_EICU,
  "eICU Combined (2014-2021): Intersectional AUROC Drift",
  "fig_intersectional_summary_eicu.png")

message("\nAll 9 figures written to ", normalizePath(OUTDIR))
