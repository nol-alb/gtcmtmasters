library(MASS)

get_simpleratios <- function(x, threshold = .05) {
  output <- rep(NA_real_, length(x))
  
  
  i <- 1
  while(any(is.na(output))) {
    n <- 2^i - 1
    
    y <- log(x[is.na(output)] / n, 2)
    hits <- (abs(y) %% 1) < threshold | (abs(y) %% 1) > (1 - threshold)
    output[which(is.na(output))[hits]] <- 2^round(y[hits], 0) * n
    
    y <- log(x[is.na(output)] * n, 2)
    hits <- (abs(y) %% 1) < threshold | (abs(y) %% 1) > (1 - threshold)
    output[which(is.na(output))[hits]] <- 2^round(y[hits], 0) / n
    
    
    i <- i + 1
    
    if (i > 2) {
      i <- 1
      threshold <- threshold * 1.1
    }
  }
  output <- MASS::fractions(output)
  output <- lapply(strsplit(attr(output, 'fracs'), split = '/'),
                   function(n) {
                     n <- as.integer(n)
                     if (length(n == 1)) n <- c(n, 1L)
                     data.frame(Numerator = n[1], Denominator = n[2])
                   })
  do.call('rbind', output)
}

transcribe <- function(timestamps, threshold = .02) {
  timestamps - min(timestamps)
  
  durations <- diff(timestamps) 
  
  ratios <- (head(durations, -1) / tail(durations, -1)) # pairwise duration ratios
  
  simpleratios <- get_simpleratios(ratios, threshold = threshold)
  
  data.frame(Timestamp = timestamps,
             Duration = c(NA, durations),
             RawRatio = c(NA, NA, ratios),
             TranscribedRatio = c(NA,NA, do.call('paste', c(simpleratios, sep = '/'))))
 
}

timestamps <- c(1000, 1750, 2000, 2500, 3500, 4000)
timestamps <- c(timestamps, timestamps + 4000) 
#where is x?
transcribe(timestamps + rnorm(length(timestamps), 0, 10), threshold = .03)
