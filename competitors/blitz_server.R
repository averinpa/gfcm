# Persistent BLITZ server: read a data-file path per line on stdin, run BLITZ, print p (or NA).
# Isolating BLITZ in this subprocess means a C++ crash kills only this process, not the study;
# the Python client detects the EOF and restarts the server, returning NaN for that call.
suppressMessages(library(BLITZ))
con <- file("stdin"); open(con)
while (TRUE) {
  path <- readLines(con, n = 1)
  if (length(path) == 0) break
  p <- tryCatch({
    d <- as.matrix(read.csv(path, header = FALSE))
    x <- d[, 1]; y <- d[, 2]
    z <- if (ncol(d) > 2) d[, 3:ncol(d), drop = FALSE] else matrix(0, nrow(d), 0)
    BLITZ(x, y, z)$p
  }, error = function(e) NA_real_)
  cat(sprintf("%.10g\n", p)); flush(stdout())
}
