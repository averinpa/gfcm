# Persistent server for the Petersen-Hansen partial-copula CI test (par_cop). Reads a data-file
# path per line on stdin, runs test_CI (B-spline quantile regression), prints the p-value for
# feature dimension q=2 (df=4) and q=3 (df=9), comma-separated, or "NA,NA" on error.
suppressMessages({library(quantreg); library(splines)})
source("vendor/par_cop/parCopCITest.R")
con <- file("stdin"); open(con)
while (TRUE) {
  path <- readLines(con, n = 1)
  if (length(path) == 0) break
  out <- tryCatch({
    d <- as.matrix(read.csv(path, header = FALSE))
    x <- d[, 1]; y <- d[, 2]
    z <- if (ncol(d) > 2) d[, 3:ncol(d), drop = FALSE] else NULL
    r <- test_CI(x, y, z, quantile_reg = "B-Spline", q = c(1, 2, 3))
    sprintf("%.8g,%.8g", r$p_value[2], r$p_value[3])
  }, error = function(e) "NA,NA")
  cat(out, "\n", sep=""); flush(stdout())
}
