# clear environment
# rm(list = ls(all.names = TRUE))


# model building libaries
library(workflows)
library(tidymodels)
library(tictoc) # timing
library(xgboost) # xgboost

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

boost_model <-
  boost_tree(mtry = tune(), 
             trees = tune(), 
             min_n = tune(),
             tree_depth = tune(),
             learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


# create workflow (add recipe and model)

# MIN PRICE
boost_wf_min_p <-
  workflow() %>% 
  add_recipe(recipe_steps_min_p(data=train_train)) %>% 
  add_model(boost_model)
boost_wf_min_p


# MAX PRICE
boost_wf_max_p <-
  workflow() %>% 
  add_recipe(recipe_steps_max_p(data=train_train)) %>% 
  add_model(boost_model)
boost_wf_max_p


boost_hypers <- 
  parameters(mtry(c(5,20)), 
  min_n(), trees(c(250,1500)), 
  tree_depth(), learn_rate()) %>% 
  grid_max_entropy(size = 20)


tic("boost min price") # start time
# grid search
set.seed(100) # seed for reproduceability
boost_results_min_p <- 
  boost_wf_min_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = boost_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


tic("boost max price") # start time

# grid search
set.seed(100) # seed for reproduceability
boost_results_max_p <- boost_wf_max_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = boost_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


# have look at the tuning parameters
boost_results_min_p %>% autoplot()
boost_results_max_p %>% autoplot()

# have a look at the performance
boost_results_min_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

boost_results_max_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

# show performance

#  1) MIN PRICE
show_best(boost_results_min_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# 2) MAX PRICE
show_best(boost_results_max_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# select best values found for the hyper parameters

# 1) MIN PRICE
boost_params_min_p <-
  select_best(boost_results_min_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)
# 2) MAX PRICE
boost_params_max_p <-
  select_best(boost_results_max_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)


# get predictions using the best parameters found 

# 1) MIN PRICE
boost_pred_min_p <- 
  collect_predictions(boost_results_min_p) %>%
  inner_join(boost_params_min_p,by = c("mtry", "trees", "min_n"))

# 2) MAX PRICE
boost_pred_max_p <- 
  collect_predictions(boost_results_max_p) %>%
  inner_join(boost_params_max_p, by = c("mtry", "trees", "min_n"))


#---------------------------------------
# variable importance plot (vip)
#---------------------------------------

# fit our model on full training data

# 1) MIN PRICE
# fit model on train_train data using optimal 
# valus found by the cross validation scheme

set.seed(100) # seed for reproduceability

boost_fit_train_train_min_p <- boost_wf_min_p %>%
  finalize_workflow(boost_params_min_p) %>%
  fit(data = train_train)



# 2) MAX PRICE
# get_final_model is a custom functions (see code/functions/final_model)
set.seed(100) # seed for reproduceability

boost_fit_train_train_max_p <- boost_wf_max_p %>%
  finalize_workflow(boost_params_max_p) %>%
  fit(data = train_train)


# 1) MIN PRICE
p_vip_boost_min_p <- vip(boost_fit_train_train_min_p$fit$fit$fit, 
   num_features = NUM_VIP, geom = "point",
   aesthetics = list(color = cbPalette[4], 
   fill = cbPalette[4])) + THEME + ggtitle("Boosting min. price")

p_vip_boost_min_p %>% ggplotly()


# 2) MAX PRICE
p_vip_boost_max_p <- vip(boost_fit_train_train_max_p$fit$fit$fit, 
   num_features = NUM_VIP, geom = "point",
   aesthetics = list(color = cbPalette[4], 
   fill = cbPalette[4])) + THEME + ggtitle("Boosting max. price")

p_vip_boost_max_p %>% ggplotly()


# add figures togehter
(p_vip_boost_min_p + p_vip_boost_max_p )


#---------------------------------------
# Look at performance on validation data
#---------------------------------------

# 1) MIN PRICE

# perform transformation on train_val data using the train_train to estimate
# train_train_rec_min_p <- prep(recipe_steps_min_p(train_train), training = train_train, fresh = TRUE, verbose = TRUE)
# train_val_data_min_p  <- bake(train_train_rec_min_p, new_data = train_validation %>% select(-min_price,-max_price))

# get prediction on validation data and add true values
boost_pred_val_min_p <- 
  predict(boost_fit_train_train_min_p, new_data = train_validation %>% select(-min_price, -max_price)) %>%
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
boost_pred_val_min_p <- 
  invert_transformations_target(
    pred = boost_pred_val_min_p,
    mean = mean_training_min_p,
    sd = sd_training_min_p,
    # suppose you did a log transformation on the target 
    # you can give FUN = exp()
    FUN = exp
  ) %>%
  mutate(residual = actual - fitted_inverted)

# check performance metrics
boost_perf_val_min_p <- 
  perf_metrics(boost_pred_val_min_p,
  truth = actual, estimate = fitted_inverted)

boost_perf_val_min_p



# 2) MAX PRICE

# perform transformation on train_val data using the train_train to estimate
# train_train_rec_max_p <- prep(recipe_steps_max_p(data=train_train), training = train_train, fresh = TRUE, verbose = TRUE)
# train_val_data_max_p  <- bake(train_train_rec_max_p, new_data = train_validation %>% select(-min_price,-max_price))

# get prediction on validation data and add true values
boost_pred_val_max_p <- 
  predict(boost_fit_train_train_max_p, new_data = (train_validation %>% select(-min_price, -max_price))) %>%
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
boost_pred_val_max_p <- 
  invert_transformations_target(
    pred = boost_pred_val_max_p,
    mean = mean_training_max_p,
    sd = sd_training_max_p,
    # suppose you did a log transformation on the 
    # target you can give FUN = exp()
    FUN = exp) %>% 
  mutate(residual = actual - fitted_inverted)



# save predictions on train_validation
tibble(
  ID = train_validation$id,
  MIN = boost_pred_val_min_p %>% select(fitted_inverted) %>% pull,
  MAX = boost_pred_val_max_p %>% select(fitted_inverted) %>% pull
) %>%
  write.csv(.,file = "output/predictions_train_val/boost_pred.csv", row.names=FALSE)


# check performance metrics
boost_perf_val_max_p <- 
  perf_metrics(boost_pred_val_max_p,
  truth = actual, estimate = fitted_inverted)

boost_perf_val_max_p

# I think this is how the perfmance will be evaluated
(boost_perf_val_min_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull +
    boost_perf_val_max_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull)


# visualize predictions and residuals on validation data
boost_fitAct_min_p_val <- boost_pred_val_min_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_point() +
  geom_abline(col = "red",linetype = "dashed") +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Boosting (validation) min. price") + THEME 

boost_fitAct_min_p_val %>% ggplotly


# MAX PRICE

boost_fitAct_max_p_val <- 
  boost_pred_val_max_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_abline(col = "red",linetype = "dashed") + 
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Boosting (validation) : max. price") + THEME 

boost_fitAct_max_p_val %>% ggplotly
boost_fitAct_min_p_val + boost_fitAct_max_p_val


# residuals analysis: should not be a pattern in the residuals

# MIN PRICE

boost_resid_min_p_val <- 
  boost_pred_val_min_p %>% 
  ggplot(.,aes(x = seq(1:nrow(boost_pred_val_min_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Boosting (validation) min. price") + THEME 

boost_resid_min_p_val %>% ggplotly


# MAX PRICE

boost_resid_max_p_val <- boost_pred_val_max_p %>% 
  ggplot(.,aes(x = seq(1:nrow(boost_pred_val_max_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Boosting (validation) max. price") + THEME 

boost_resid_max_p_val %>% ggplotly


# everything together
(boost_fitAct_min_p_val + boost_fitAct_max_p_val)


# overview plot of everything
(boost_fitAct_min_p_val + boost_fitAct_max_p_val)/
  (boost_resid_min_p_val + boost_resid_max_p_val)


# visualize the distributions of the predictions on validation test set and compare it 
# with the actual values on the validation we have (this is just a check)

# distributions on training data
p_test_min_p_train_val <- 
  check_predictions(
    df=boost_pred_val_min_p,var = "actual",
    xlab = "Min. price: validation: actual") + THEME

p_test_max_p_train_val <- 
  check_predictions(
    df=boost_pred_val_max_p,var = "actual",
    xlab = "Max. price validation: actual") + THEME

# distribution of the predictions
p_boost_pred_min_p_train_val <- 
  check_predictions(
    df=boost_pred_val_min_p,var = "fitted_inverted",
    xlab = "Min. price validation: fitted") + THEME

p_boost_pred_max_p_train_val <- 
  check_predictions(
    df=boost_pred_val_max_p,var = "fitted_inverted",
    xlab = "Max. price validation: fitted") + THEME

((p_test_min_p_train_val + p_test_max_p_train_val)/
    (p_boost_pred_min_p_train_val + p_boost_pred_max_p_train_val))


#---------------------------------------
# Fit final model on all data
#---------------------------------------

# MIN PRICE
boost_fit_train_all_data_min_p <- 
  boost_wf_min_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(boost_params_min_p) %>%
  fit(data = df)

# save model
saveRDS(boost_fit_train_all_data_min_p, 
        "code/models/save_models/boost_min_p.rds")

# MAX PRICE
boost_fit_train_all_data_max_p <- 
  boost_wf_max_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(boost_params_max_p) %>%
  fit(data = df)

# save model
saveRDS(boost_fit_train_all_data_max_p, 
        "code/models/save_models/boost_max_p.rds")

