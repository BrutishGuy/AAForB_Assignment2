library(tune)    

METRIC <- "mae" # metric of interest
MAXIMIZE <- FALSE # we need to minimize the metric (depends on performance metric)

# resampling scheme: cross validation 5 times
cross_val_scheme <- vfold_cv(training(split_scheme), v = 5)

# performance metrics: you can define your own: see documentation
perf_metrics <- metric_set(mae, rmse, rsq_trad)

# Save the assessment set predictions
ctrl <- control_grid(save_pred = TRUE, verbose = TRUE)


