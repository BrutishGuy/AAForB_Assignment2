# clear environment
# rm(list = ls(all.names = TRUE))


# model building libaries
library(workflows)
library(tidymodels)
library(tictoc) # timing
library(ranger) # random forest

# visualize
library(naniar)
library(tidyverse)
library(vip) 


# load own functions
source("code/functions/invert_transformations_target.R")
source("code/functions/check_predictions.R")


# global parameters
source("code/models/global_params.R")

# seed for reproduceability
set.seed(100)

# run preprocessing steps
source("code/preprocessing/preprocessing.R")

# load cross validation scheme
source("code/models/cross_validation_scheme.R")


#----------------------------------------------------------
# Train Model and find optimal tuning parameters
#----------------------------------------------------------

rf_model <-
  rand_forest(mtry = tune(), 
              trees = tune(),
              min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")


# create workflow (add recipe and model)

# MIN PRICE
rf_wf_min_p <-
  workflow() %>% 
  add_recipe(recipe_steps_min_p(data=train_train)) %>% 
  add_model(rf_model)
rf_wf_min_p


# MAX PRICE
rf_wf_max_p <-
  workflow() %>% 
  add_recipe(recipe_steps_max_p(data=train_train)) %>% 
  add_model(rf_model)
rf_wf_max_p


rf_hypers <- 
  parameters(mtry(c(5,22)), 
             min_n(),
             trees(c(250,1500))) %>% 
  grid_max_entropy(size = 20)


tic("rf min price") # start time
# grid search
set.seed(100) # seed for reproduceability
rf_results_min_p <- 
  rf_wf_min_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = rf_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


tic("rf max price") # start time

# grid search
set.seed(100) # seed for reproduceability
rf_results_max_p <- rf_wf_max_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = rf_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


# have look at the tuning parameters
rf_results_min_p %>% autoplot()
rf_results_max_p %>% autoplot()

# have a look at the performance
rf_results_min_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

rf_results_max_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

# show performance

#  1) MIN PRICE
show_best(rf_results_min_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# 2) MAX PRICE
show_best(rf_results_max_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# select best values found for the hyper parameters

# 1) MIN PRICE
rf_params_min_p <-
  select_best(rf_results_min_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)
# 2) MAX PRICE
rf_params_max_p <-
  select_best(rf_results_max_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)


# get predictions using the best parameters found 

# 1) MIN PRICE
rf_pred_min_p <- 
  collect_predictions(rf_results_min_p) %>%
  inner_join(rf_params_min_p,by = c("mtry", "trees", "min_n"))

# 2) MAX PRICE
rf_pred_max_p <- 
  collect_predictions(rf_results_max_p) %>%
  inner_join(rf_params_max_p, by = c("mtry", "trees", "min_n"))


#---------------------------------------
# variable importance plot (vip)
#---------------------------------------

# fit our model on full training data

# 1) MIN PRICE
# fit model on train_train data using optimal 
# valus found by the cross validation scheme

set.seed(100) # seed for reproduceability

rf_fit_train_train_min_p <- rf_wf_min_p %>%
  finalize_workflow(rf_params_min_p) %>%
  fit(data = train_train)

# 2) MAX PRICE
# get_final_model is a custom functions (see code/functions/final_model)
set.seed(100) # seed for reproduceability

rf_fit_train_train_max_p <- rf_wf_max_p %>%
  finalize_workflow(rf_params_max_p) %>%
  fit(data = train_train)


# 1) MIN PRICE
p_vip_rf_min_p <- 
  vip(rf_fit_train_train_min_p$fit$fit$fit, 
      num_features = NUM_VIP, geom = "point",
      aesthetics = list(color = cbPalette[4], 
      fill = cbPalette[4])) + 
  THEME + ggtitle("Random Forest min. price")

p_vip_rf_min_p %>% ggplotly()


# 2) MAX PRICE
p_vip_rf_max_p <- 
  vip(rf_fit_train_train_max_p$fit$fit$fit, 
      num_features = NUM_VIP, geom = "point",
      aesthetics = list(color = cbPalette[4], 
      fill = cbPalette[4])) + 
  THEME + ggtitle("Random Forest max. price")

p_vip_rf_max_p %>% ggplotly()


# add figures togehter
(p_vip_rf_min_p + p_vip_rf_max_p )


#---------------------------------------
# Look at performance on validation data
#---------------------------------------

# 1) MIN PRICE

# perform transformation on train_val data using the train_train to estimate
train_train_rec_min_p <- prep(recipe_steps_min_p(data=train_train), training = train_train, fresh = TRUE, verbose = TRUE)
train_val_data_min_p  <- bake(train_train_rec_min_p, new_data = train_validation %>% select(-min_price,-max_price))

# make predictions on train_val data and add true values
rf_pred_val_min_p <- 
  predict(rf_fit_train_train_min_p, new_data = train_validation %>% select(-min_price,-max_price)) %>%
  mutate(actual = train_validation$min_price) %>% 
  dplyr::rename(fitted = .pred)


# rescale back to the orignal scale
# compute mean
mean_training_min_p <- train_train %>% 
  select(min_price) %>% log %>% pull %>% mean()
# compute sd
sd_training_min_p <- train_train %>% 
  select(min_price) %>% log %>% pull %>% sd()

# transform back to original unit

# add back mean and sd  + invert potentially others transformation such as log trans
rf_pred_val_min_p <- 
  invert_transformations_target(
    pred = rf_pred_val_min_p,
    mean = mean_training_min_p,
    sd = sd_training_min_p,
    # suppose you did a log transformation on the target 
    # you can give FUN = exp()
    FUN = exp
  ) %>%
  mutate(residual = actual - fitted_inverted)

# check performance metrics
rf_perf_val_min_p <- 
  perf_metrics(rf_pred_val_min_p,
  truth = actual, estimate = fitted_inverted)

rf_perf_val_min_p


# 2) MAX PRICE

# perform transformation on train_val data using the train_train to estimate
train_train_rec_max_p <- prep(recipe_steps_max_p(data=train_train), training = train_train, fresh = TRUE, verbose = TRUE)
train_val_data_max_p  <- bake(train_train_rec_max_p, new_data = train_validation %>% select(-min_price,-max_price))


# get prediction on validation data and add true values
rf_pred_val_max_p <- 
  predict(rf_fit_train_train_max_p, new_data = train_validation %>% select(-min_price,-max_price)) %>%
  mutate(actual = train_validation$max_price) %>% 
  dplyr::rename(fitted = .pred)


# rescale back to the orignal scale
mean_training_max_p <- 
  train_train %>% 
  select(max_price) %>% log %>% pull %>% mean()
sd_training_max_p <- 
  train_train %>% 
  select(max_price) %>% log %>% pull %>% sd()

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
rf_pred_val_max_p <- 
  invert_transformations_target(
    pred = rf_pred_val_max_p,
    mean = mean_training_max_p,
    sd = sd_training_max_p,
    # suppose you did a log transformation on the 
    # target you can give FUN = exp()
    FUN = exp
  ) %>% 
  mutate(residual = actual - fitted_inverted)


# save predictions on train_validation
tibble(
  ID = train_validation$id,
  MIN = rf_pred_val_min_p %>% select(fitted_inverted) %>% pull,
  MAX = rf_pred_val_max_p %>% select(fitted_inverted) %>% pull
) %>%
  write.csv(.,file = "output/predictions_train_val/rf_pred.csv", row.names=FALSE)


# check performance metrics
rf_perf_val_max_p <- 
  perf_metrics(rf_pred_val_max_p, 
  truth = actual, estimate = fitted_inverted)

rf_perf_val_max_p

# I think this is how the perfmance will be evaluated
(rf_perf_val_min_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull +
    rf_perf_val_max_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull)


# visualize predictions and residuals on validation data
rf_fitAct_min_p_val <- rf_pred_val_min_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_point() +
  geom_abline(col = "red",linetype = "dashed") +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Random Forest (validation) min. price") + THEME 

rf_fitAct_min_p_val %>% ggplotly


# MAX PRICE

rf_fitAct_max_p_val <- 
  rf_pred_val_max_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_abline(col = "red",linetype = "dashed") + 
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Random Forest (validation) : max. price") + THEME 

rf_fitAct_max_p_val %>% ggplotly
rf_fitAct_min_p_val + rf_fitAct_max_p_val


# residuals analysis: should not be a pattern in the residuals

# MIN PRICE

rf_resid_min_p_val <- 
  rf_pred_val_min_p %>% 
  ggplot(.,aes(x = seq(1:nrow(rf_pred_val_min_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Random Forest (validation) min. price") + THEME 

rf_resid_min_p_val %>% ggplotly


# MAX PRICE

rf_resid_max_p_val <- rf_pred_val_max_p %>% 
  ggplot(.,aes(x = seq(1:nrow(rf_pred_val_max_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Random Forest (validation) max. price") + THEME 

rf_resid_max_p_val %>% ggplotly


# everything together
(rf_fitAct_min_p_val + rf_fitAct_max_p_val)


# overview plot of everything
(rf_fitAct_min_p_val + rf_fitAct_max_p_val)/
  (rf_resid_min_p_val + rf_resid_max_p_val)


# visualize the distributions of the predictions on validation test set and compare it 
# with the actual values on the validation we have (this is just a check)

# distributions on training data
p_test_min_p_train_val <- 
  check_predictions(
  df=rf_pred_val_min_p,var = "actual",
  xlab = "Min. price: validation: actual") + THEME

p_test_max_p_train_val <- 
  check_predictions(
  df=rf_pred_val_max_p,var = "actual",
  xlab = "Max. price validation: actual") + THEME

# distribution of the predictions
p_rf_pred_min_p_train_val <- 
  check_predictions(
    df=rf_pred_val_min_p,var = "fitted_inverted",
    xlab = "Min. price validation: fitted") + THEME

p_rf_pred_max_p_train_val <- 
  check_predictions(
    df=rf_pred_val_max_p,var = "fitted_inverted",
    xlab = "Max. price validation: fitted") + THEME

((p_test_min_p_train_val + p_test_max_p_train_val)/
    (p_rf_pred_min_p_train_val + p_rf_pred_max_p_train_val))


#---------------------------------------
# Fit final model on all data
#---------------------------------------

# MIN PRICE
rf_fit_train_all_data_min_p <- 
  rf_wf_min_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(rf_params_min_p) %>%
  fit(data = df)



# save model
saveRDS(rf_fit_train_all_data_min_p, "code/models/save_models/rf_min_p.rds")

# MAX PRICE
rf_fit_train_all_data_max_p <- 
  rf_wf_max_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(rf_params_max_p) %>%
  fit(data = df)

# save model
saveRDS(rf_fit_train_all_data_max_p, "code/models/save_models/rf_max_p.rds")

