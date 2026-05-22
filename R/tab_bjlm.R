# tab_bjlm.R
#
# Presents fixed effects from one or more bjlm_fit objects as a formatted
# table, with parameters on rows and models in columns.
#
# Requires: gt (for the rendered table), dplyr (for tidy construction).
# Falls back to knitr::kable() if gt is not installed.

# ---------------------------------------------------------------------------
#' Fixed-effects table for bjlm_fit objects
#'
#' Collects posterior summaries from one or more `bjlm_fit` objects and
#' displays them in a single table with parameters on rows and models in
#' columns. All parameters present in any model are shown; models that do not
#' include a given parameter display a dash.
#'
#' @param ... One or more `bjlm_fit` objects.
#' @param labels Character vector of column headers, one per model. If `NULL`
#'   (default) the deparsed call names are used.
#' @param digits Integer; number of decimal places (default `2`).
#' @param fmt Cell format: `"mean [CI]"` (default) shows mean and 95% credible
#'   interval; `"mean (SD)"` shows mean and posterior SD.
#' @param show_rhat Logical; append \eqn{\hat{R}} to each cell (default
#'   `FALSE`).
#'
#' @return A `gt_tbl` object (or a `knitr_kable` if **gt** is not installed).
#'
#' @examples
#' \dontrun{
#' tab_bjlm(fit1, fit2, labels = c("Model A", "Model B"))
#' }
#' @export
tab_bjlm <- function(...,
                     labels    = NULL,
                     digits    = 2,
                     fmt       = c("mean [CI]", "mean (SD)"),
                     show_rhat = FALSE) {

  fmt    <- match.arg(fmt)
  models <- list(...)
  n      <- length(models)

  if (n == 0L) stop("Supply at least one bjlm_fit object.")
  if (!all(sapply(models, function(m) inherits(m, "bjlm_fit") || inherits(m, "bipw_fit")))) {
    stop("All positional arguments must be bjlm_fit objects.")
  }

  # ---- Column labels --------------------------------------------------------
  if (is.null(labels)) {
    cl      <- match.call()
    cl_args <- as.list(cl)[-1]
    cl_args <- cl_args[!names(cl_args) %in% c("labels", "digits", "fmt", "show_rhat")]
    nm      <- sapply(cl_args, deparse)
    labels  <- if (length(nm) == n) nm else paste0("Model ", seq_len(n))
  }
  if (length(labels) != n) {
    stop("`labels` must have one entry per model (", n, " models supplied).")
  }

  # ---- Format one summary row into a single string -------------------------
  fmt_cell <- function(s, i) {
    m  <- round(as.numeric(s$mean[i]),  digits)
    lo <- round(as.numeric(s$q_lo[i]),  digits)
    hi <- round(as.numeric(s$q_hi[i]),  digits)
    sd <- round(as.numeric(s$sd[i]),    digits)
    rh <- round(as.numeric(s$rhat[i]),  3)

    cell <- if (fmt == "mean [CI]") {
      sprintf("%.*f [%.*f, %.*f]", digits, m, digits, lo, digits, hi)
    } else {
      sprintf("%.*f (%.*f)", digits, m, digits, sd)
    }
    if (show_rhat) cell <- paste0(cell, "  \u0052\u0302=", rh)
    cell
  }

  # ---- Extract and format parameters silently from each model --------------
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("Package 'posterior' is required for tab_bjlm. Install it with install.packages('posterior').")
  }

  cols <- lapply(models, function(fit) {
    draws <- fit$draws
    param_names <- c(fit$outcome_names, fit$propensity_names)
    
    # Exclude random effects u_... if present to keep table focus on fixed effects
    param_names <- param_names[!grepl("^u_", param_names)]
    
    if (length(param_names) == 0) return(character(0))
    
    sub <- posterior::subset_draws(draws, variable = param_names)
    s <- posterior::summarise_draws(sub,
      mean = mean,
      sd = stats::sd,
      q_lo = ~ stats::quantile(.x, probs = 0.025, names = FALSE),
      q_hi = ~ stats::quantile(.x, probs = 0.975, names = FALSE),
      rhat = posterior::rhat
    )
    
    vals <- vapply(seq_len(nrow(s)),
                   function(i) fmt_cell(s, i),
                   character(1))
    stats::setNames(vals, s$variable)
  })

  # ---- Union of parameters in natural order --------------------------------
  all_vars <- unique(unlist(lapply(cols, names)))

  # ---- Assemble wide data frame --------------------------------------------
  tbl <- data.frame(Parameter = all_vars, stringsAsFactors = FALSE)
  for (j in seq_len(n)) {
    tbl[[labels[j]]] <- cols[[j]][all_vars]   # NA for absent parameters
  }

  # ---- Parse "block_term" variable names -----------------------------------
  # e.g. "b0_(Intercept)" -> block = "b0", term = "(Intercept)"
  tbl$block <- sub("_.*", "", tbl$Parameter)
  tbl$term  <- sub("^[^_]+_", "", tbl$Parameter)
  solo      <- tbl$block == tbl$term
  tbl$term[solo] <- tbl$block[solo]

  # Map block prefixes to pretty labels
  pretty_block <- function(b) {
    if (b == "alpha") return("\u03B1 \u2013 Propensity score coefficients")
    if (b == "b0") return("\u03B2\u2080 \u2013 Outcome intercept and covariates")
    if (b == "b1") return("\u03B2\u2081 \u2013 Outcome initial slope")
    if (b == "sigma") return("\u03C3 \u2013 Residual standard deviation")
    if (b == "sigma_u") return("\u03C3_u \u2013 Subject random intercept SD")
    if (grepl("^delta([0-9]+)$", b)) {
       k <- sub("delta", "", b)
       return(sprintf("\u0394\u03B2 %s \u2013 Slope change at BP%s", k, k))
    }
    if (grepl("^omega([0-9]+)$", b)) {
       k <- sub("omega", "", b)
       return(sprintf("\u03C9%s \u2013 Transition point %s", k, k))
    }
    if (grepl("^rho([0-9]+)$", b)) {
       k <- sub("rho", "", b)
       return(sprintf("\u03C1%s \u2013 Sharpness %s", k, k))
    }
    if (grepl("^gamma_b1", b)) return("\u03B3 b1 \u2013 Inclusion (b1)")
    if (grepl("^gamma_delta([0-9]+)$", b)) {
       k <- sub("gamma_delta", "", b)
       return(sprintf("\u03B3%s \u2013 Inclusion (BP%s)", k, k))
    }
    b
  }
  tbl$block_label <- vapply(tbl$block, pretty_block, character(1))

  tbl_out <- tbl[, c("block_label", "term", labels)]

  # ---- Render --------------------------------------------------------------
  if (!requireNamespace("gt", quietly = TRUE)) {
    message("Install the 'gt' package for a richer table. Falling back to knitr::kable().")
    return(knitr::kable(tbl_out,
                        col.names = c("Block", "Parameter", labels),
                        row.names = FALSE,
                        align     = c("l", "l", rep("c", n))))
  }

  subtitle_text <- if (fmt == "mean [CI]") "Mean [95% credible interval]" else "Mean (posterior SD)"

  tbl_out |>
    gt::gt(groupname_col = "block_label", rowname_col = "term") |>
    gt::tab_header(title    = "Fixed effects",
                   subtitle = subtitle_text) |>
    gt::cols_align(align = "right", columns = dplyr::all_of(labels)) |>
    gt::cols_align(align = "left",  columns = "term") |>
    gt::tab_style(
      style     = gt::cell_text(weight = "bold"),
      locations = gt::cells_row_groups()
    ) |>
    gt::tab_style(
      style     = gt::cell_fill(color = "#efefef"),
      locations = gt::cells_row_groups()
    ) |>
    gt::sub_missing(missing_text = "\u2014") |>
    gt::opt_table_font(font = list(gt::google_font("IBM Plex Mono")), size = 13) |>
    gt::tab_options(table.width           = gt::pct(100),
                    row_group.border.top.width    = gt::px(2),
                    row_group.border.bottom.width = gt::px(1))
}
