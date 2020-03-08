# clean environment
rm(list=ls())


# model building libaries
library(recipes)  # to define preprocessing steps
library(workflows)
library(tidymodels)
library(rsample) # to perform cross validation
library(tune)    # tuning
library(glmnet)
library(tictoc) # timing
library(rlang)

# visualize
library(naniar)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(plotly) # interactive vis
library(GGally)
library(vip) 

# parallel processing in case we want to speed up things (probably not necessary)
# I don't knwow whether this works on a mac
# library(doParallel) 


# load own functions
source("code/functions/final_model.R")
source("code/functions/predict_new_data.R")
source("code/functions/untransform_target.R")


# Global parameters ---------------------------------------
# colors
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

THEME <- theme_minimal()
LEGEND <- theme(legend.title = element_blank())
# seed for reproduceability
set.seed(100)

# Parallel Processing
# parallel::detectCores(logical = TRUE)
#cl <- makeCluster(2)


METRIC <- "mae" # metric of interest
MAXIMIZE <- FALSE # we need to minimize the metric (depends on performance metric)
NUM_VIP <- 10 # look at the 10 most important variables



# import data and rename id
df <- read.csv("data/train.csv",sep = ";") %>% dplyr::rename(
  id = ï..id) 

# split in train and validation
# on this traning data I use cross validation to tune the hyper parameters
# the validation data is used as sort of safeguard to prevent ourselfves from overfitting
# 0.75 training, 0.25 validation
split_scheme <- initial_split(df, prop = 3/4)

# check distributions
# training min price
p_train_min_p <- split_scheme %>%
  training() %>% select(min_price) %>% 
  ggplot(.,aes(min_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "max. price") + ggtitle("Train") + THEME

# training max price
p_train_max_p <- split_scheme %>%
  training() %>% select(max_price) %>% 
  ggplot(.,aes(max_price)) + 
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "min. price") + ggtitle("Train") + THEME


# validation min price
p_val_min_p <- split_scheme %>%
  testing() %>% select(min_price) %>% 
  ggplot(.,aes(min_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "max. price") + ggtitle("Validation") + THEME

# validation max price
p_val_max_p <- split_scheme %>%
  testing() %>% select(max_price) %>% 
  ggplot(.,aes(max_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "min price") + ggtitle("Validation") + THEME 


# make figure (this uses the libraray patchwork ==> super cool library)
(p_train_min_p + p_train_max_p)/
  (p_val_min_p + p_val_max_p)


# create recips: this is the preprocessing pipeline
# will be applied to every fold in the cross validation and also to the validation
# and test set
# the steps are performed in order so be careful
# also when you do transformation on the target variable (such as standardizing) don't
# foget to backtransform to the orignal space as the predictions are expected in the
# orignal uinits


# create a seperate model for 
#  1) min price
#  2) max price

# preprocessing steps min price
recipe_steps_min_p <- training(split_scheme) %>%
  recipe(min_price ~., data = df) %>%
  # feature we remove 
  step_rm(name, base_name, gpu, cpu_details, id, max_price) %>%
  # log transformation on target since it's right skewed
  step_log(all_outcomes()) %>%
  # k nearest neightbors for imputation of missing values in the feature space
  step_knnimpute(all_predictors()) %>%
  # to make data more noramlly dist.
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  # for previously unseen factors
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  # one hot encoding
  step_dummy(all_nominal()) %>%
  # normalize data (mean = 0 and sd 1)
  step_normalize(all_numeric()) %>%
  step_nzv(all_predictors())



# preprocessing steps max price
recipe_steps_max_p <- training(split_scheme) %>%
  recipe(max_price ~., data = df) %>%
  # feature we remove 
  step_rm(name, base_name, gpu, cpu_details, id, min_price) %>%
  # log transformation on target since it's right skewed
  step_log(all_outcomes()) %>%
  # k nearest neightbors for imputation of missing values in the feature space
  step_knnimpute(all_predictors()) %>%
  # to make data more noramlly dist.
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  # for previously unseen factors
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  # one hot encoding
  step_dummy(all_nominal()) %>%
  # normalize data (mean = 0 and sd 1)
  step_normalize(all_numeric()) %>%
  step_nzv(all_predictors())
  
# quick check data prep if it works 
check_prep  <- prep(recipe_steps_min_p, 
                    training = training(split_scheme), verbose = TRUE)
check_juice <- juice(check_prep) 

colnames(check_juice)
ncol(check_juice)
nrow(check_juice)


# mean zero and std 1
sapply(check_juice,mean)
sapply(check_juice,sd)


# resampling scheme: cross validation 5 times
cross_val_scheme <- vfold_cv(training(split_scheme), v = 5)


# performance metrics: you can define your own: see documentation
perf_metrics <- metric_set(mae, rmse, rsq)

# Save the assessment set predictions
ctrl <- control_grid(save_pred = TRUE, verbose = TRUE)


#=====================================================#
#  Model Building
#=====================================================#

#----------------------------------------------------------
# glmnet (Linear regression with a penalty): Bencmark model
#----------------------------------------------------------


# specify model: this will be the same for the min price and max price
glmnet_model <- linear_reg(
  mode    = "regression", 
  penalty = tune(), # tuning parameter 1 how hard you want to penalize the coefficients
  mixture = tune()  # tunng parameter 2 (mixture between lasso, ridge and lasso)
) %>%
  set_engine("glmnet")


# create workflow (add recipe and model)

# MIN PRICE
glmnet_wf_min_p <-
  workflow() %>% 
  add_recipe(recipe_steps_min_p) %>% 
  add_model(glmnet_model)
glmnet_wf_min_p


# MAX PRICE
glmnet_wf_max_p <-
  workflow() %>% 
  add_recipe(recipe_steps_max_p) %>% 
  add_model(glmnet_model)
glmnet_wf_max_p


# Save the assessment set predictions
ctrl <- control_grid(save_pred = TRUE, verbose = TRUE)


# set grid for parameters to look at (same for min and max price)
# you can increase the size, but the model fitting will be slower
# since you will test more values
glmnet_hypers <- parameters(penalty(), mixture()) %>%
  grid_max_entropy(size = 10) 

# visualize the parameters of glmnet
ggplot(glmnet_hypers,aes(x = penalty, y = mixture)) + geom_point()+
  scale_x_log10() + THEME

tic("glmnet min price") # start time
# grid search
glmnnet_results_min_p <- glmnet_wf_min_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = glmnet_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


tic("glmnet max price") # start time
# grid search
glmnnet_results_max_p <- glmnet_wf_max_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = glmnet_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


# have look at the tuning parameters
glmnnet_results_min_p %>% autoplot()
glmnnet_results_max_p %>% autoplot()

# have a look at the performance
glmnnet_results_min_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

glmnnet_results_max_p %>%
  collect_metrics() %>%
  filter(.metric == METRIC)

# show performance

#  1) MIN PRICE
show_best(glmnnet_results_min_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# 2) MAX PRICE
show_best(glmnnet_results_max_p, 
          n = 10,
          metric = METRIC, 
          maximize = MAXIMIZE)

# select best values found for the hyper parameters

# 1) MIN PRICE
glmnet_params_min_p <-
  select_best(glmnnet_results_min_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)
# 2) MAX PRICE
glmnet_params_max_p <-
  select_best(glmnnet_results_max_p, 
              metric = METRIC, 
              maximize = MAXIMIZE)


# get predictions using the best parameters found 

# 1) MIN PRICE
glmnnet_pred_min_p <- collect_predictions(glmnnet_results_min_p) %>%
  inner_join(glmnet_params_min_p, by = c("penalty", "mixture"))

# 2) MAX PRICE
glmnnet_pred_max_p <- collect_predictions(glmnnet_results_max_p) %>%
  inner_join(glmnet_params_max_p, by = c("penalty", "mixture"))


# fit our model on full training data

# 1) MIN PRICE
# get_final_model is a custom functions (see code/functions/final_model)
glmnnet_fit_train_min_p <- get_final_model(recipe_steps = recipe_steps_min_p,
                           data = training(split_scheme),
                           model = glmnet_model,
                           target = "min_price",
                           params = glmnet_params_min_p)

# 2) MAX PRICE
# get_final_model is a custom functions (see code/functions/final_model)
glmnnet_fit_train_max_p <- get_final_model(recipe_steps = recipe_steps_max_p,
                           data = training(split_scheme),
                           model = glmnet_model,
                           target = "max_price",
                           params = glmnet_params_max_p)


# just a cool visual plot how the coefficients shrink accoridng to penalty
# abolsute higher coefficients are more important features
# this model was also discussed in class


# 1) MIN PRICE

tidy_coefs_min_p <-
  broom::tidy(glmnnet_fit_train_min_p) %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::select(-step,-dev.ratio)

# get the lambda closest to tune's optimal choice
delta_min_p <- abs(tidy_coefs_min_p$lambda - glmnet_params_min_p$penalty)
lambda_opt_min_p <- tidy_coefs_min_p$lambda[which.min(delta_min_p)]


# Keep the large absolute values
label_coefs_min_p <-
  tidy_coefs_min_p %>%
  mutate(abs_estimate = abs(estimate)) %>%
  # only add labels with a coefficient with an absolute value higer than .1
  dplyr::filter(abs_estimate >= .10) %>%
  distinct(term) %>%
  inner_join(tidy_coefs_min_p, by = "term") %>%
  dplyr::filter(lambda == lambda_opt_min_p)


# plot the paths and hightlist the large values
# the vertical lines indicates the optimal value
p_tidy_coefs_min_p <- tidy_coefs_min_p %>%
  ggplot(aes(x = lambda, y = estimate, group = term, col = term, label = term)) +
  geom_vline(xintercept = lambda_opt_min_p, lty = 2) + THEME + 
  labs(x = "Penalty", y  ="Coefficient estimate") +  ggtitle("Minimum price") +
  geom_line(alpha = 1) +
  theme(legend.position = "none") + 
  scale_x_log10() +
  ggrepel::geom_text_repel(data = label_coefs_min_p, aes(label = term, x = .05))

p_tidy_coefs_min_p


# 2) MAX PRICE

tidy_coefs_max_p <-
  broom::tidy(glmnnet_fit_train_max_p) %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::select(-step,-dev.ratio) 

# get the lambda closest to tune's optimal choice
delta_max_p <- abs(tidy_coefs_max_p$lambda - glmnet_params_max_p$penalty)
lambda_opt_max_p <- tidy_coefs_max_p$lambda[which.min(delta_max_p)]


# Keep the large values
label_coefs_max_p <-
  tidy_coefs_max_p %>%
  mutate(abs_estimate = abs(estimate)) %>%
  # only add labels with a coefficient with an absolute value higer than .1
  dplyr::filter(abs_estimate >= .1) %>%
  distinct(term) %>%
  inner_join(tidy_coefs_max_p, by = "term") %>%
  dplyr::filter(lambda == lambda_opt_max_p)


# plot the paths and hightlist the large values
p_tidy_coefs_max_p <- tidy_coefs_max_p %>%
  ggplot(aes(x = lambda, y = estimate, group = term, col = term, label = term)) +
  geom_vline(xintercept = lambda_opt_max_p, lty = 2) + THEME + 
  labs(x = "Penalty", y  ="Coefficient estimate") + ggtitle("Maximum price") +
  geom_line(alpha = 1) +
  theme(legend.position = "none") +
  scale_x_log10() +
  ggrepel::geom_text_repel(data = label_coefs_max_p, aes(label = term, x = .005)) 
  

(p_tidy_coefs_min_p + p_tidy_coefs_max_p)

# variable importance plot

# MIN PRICE
p_vip_glmnet_min_p <- vip(glmnnet_fit_train_min_p, num_features = NUM_VIP,
  lambda = glmnet_params_min_p$penalty, geom = "point",
  aesthetics = list(color = cbPalette[4], fill = cbPalette[4])) +
  THEME + ggtitle("Linear model (glmnet) min. price")

p_vip_glmnet_min_p %>% ggplotly()


# MAX PRICE
p_vip_glmnet_max_p <- vip(glmnnet_fit_train_max_p, num_features = NUM_VIP,
  lambda = glmnet_params_max_p$penalty, geom = "point",
  aesthetics = list(color = cbPalette[4], fill = cbPalette[4])) + THEME + 
  ggtitle("Linear model (glmnet) max. price") 

p_vip_glmnet_max_p %>% ggplotly()


# add figures togehter
(p_vip_glmnet_min_p+p_vip_glmnet_max_p )

# Look at performance on validation data
#---------------------------------------

# 1) MIN PRICE

# get prediction on validation data and add true values
glmnet_pred_val_min_p <- prediction_new_data(recipe_steps = recipe_steps_min_p,
  data = testing(split_scheme),model = glmnnet_fit_train_min_p) %>%
  mutate(actual = testing(split_scheme)$min_price) %>% 
  dplyr::rename(fitted = .pred) %>%
  select(fitted,actual) 


# rescale back to the orignal scale
# compute mean
mean_training_min_p <- training(split_scheme) %>% 
  select(min_price) %>% log %>% pull %>% mean()
# compute sd
sd_training_min_p <- training(split_scheme) %>% 
  select(min_price) %>% log %>% pull %>% sd()

# transform back to original unit

# add back mean and sd  + invert potentially others transformation such as log trans
glmnet_pred_val_min_p <- untransform_target(pred = glmnet_pred_val_min_p,
                   mean = mean_training_min_p,
                   sd = sd_training_min_p,
                   # suppose you did a log transformation on the target 
                   # you can give FUN = exp()
                   FUN = exp) %>%
                   mutate(residual = actual - fitted_unscaled)


# check performance metrics
glmnet_perf_val_min_p <- perf_metrics(glmnet_pred_val_min_p,
             truth = actual, estimate = fitted_unscaled)

glmnet_perf_val_min_p

# 2) MAX PRICE

# get prediction on validation data and add true values
glmnet_pred_val_max_p <- prediction_new_data(recipe_steps = recipe_steps_max_p,
  data = testing(split_scheme),model = glmnnet_fit_train_max_p) %>%
  mutate(actual = testing(split_scheme)$max_price) %>% 
  dplyr::rename(fitted = .pred) %>%
  select(fitted,actual) 


# rescale back to the orignal scale
mean_training_max_p <- training(split_scheme) %>% 
  select(max_price) %>% log %>% pull %>% mean()
sd_training_max_p <- training(split_scheme) %>% 
  select(max_price) %>% log %>% pull %>% sd()

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
glmnet_pred_val_max_p <- untransform_target(pred = glmnet_pred_val_max_p,
                   mean = mean_training_max_p,
                   sd = sd_training_max_p,
                   # suppose you did a log transformation on the 
                   # target you can give FUN = exp()
                   FUN = exp) %>% 
                   mutate(residual = actual - fitted_unscaled)


# check performance metrics
glmnet_perf_val_max_p <- perf_metrics(glmnet_pred_val_max_p,
                                      truth = actual, estimate = fitted_unscaled)

glmnet_perf_val_max_p

# I think this is how the perfmance will be evaluated
(glmnet_perf_val_min_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull +
glmnet_perf_val_max_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull)
  

# visualize predictions and residuals on validation data

glmnet_fitAct_min_p_val <- glmnet_pred_val_min_p %>% 
  ggplot(.,aes(x = actual, y = fitted_unscaled)) + 
  geom_point() +
  geom_abline(col = "red",linetype = "dashed") +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Linear Model (validation) min. price") + THEME 

glmnet_fitAct_min_p_val %>% ggplotly


# MAX PRICE

glmnet_fitAct_max_p_val <- glmnet_pred_val_max_p %>% ggplot(.,aes(x = actual, y = fitted_unscaled)) + 
  geom_abline(col = "red",linetype = "dashed") + 
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Linear Model (validation) : max. price") + THEME 

glmnet_fitAct_max_p_val %>% ggplotly
glmnet_fitAct_min_p_val + glmnet_fitAct_max_p_val


# residuals analysis: should not be a pattern in the residuals

# MIN PRICE

glmnet_resid_min_p_val <- glmnet_pred_val_min_p %>% 
  ggplot(.,aes(x = seq(1:nrow(glmnet_pred_val_min_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Linear Model (validation) min. price") + THEME 

glmnet_resid_min_p_val %>% ggplotly


# MAX PRICE

glmnet_resid_max_p_val <- glmnet_pred_val_max_p %>% 
  ggplot(.,aes(x = seq(1:nrow(glmnet_pred_val_max_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Linear Model (validation) max. price") + THEME 

glmnet_resid_max_p_val %>% ggplotly


# everything together
(glmnet_fitAct_min_p_val + glmnet_fitAct_max_p_val)


# overview plot of everything
(glmnet_fitAct_min_p_val + glmnet_fitAct_max_p_val)/
(glmnet_resid_min_p_val + glmnet_resid_max_p_val)

# to save a figure (this is just a test): it will automatically save the last
# figure you plotted
ggsave("output/glmnet_predictions_validations.jpeg", 
       device = "jpeg", width = 8, height = 5)



#----------------------------------------------------------
# Boosting
#----------------------------------------------------------

boost_model <-
  boost_tree(mtry = tune(), 
             trees = tune(), 
             min_n = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


# create workflow (add recipe and model)

# MIN PRICE
boost_wf_min_p <-
  workflow() %>% 
  add_recipe(recipe_steps_min_p) %>% 
  add_model(boost_model)
glmnet_wf_min_p


# MAX PRICE
boost_wf_max_p <-
  workflow() %>% 
  add_recipe(recipe_steps_max_p) %>% 
  add_model(boost_model)
glmnet_wf_max_p


boost_hypers <- parameters(mtry(c(5,20)), 
  min_n(),trees(c(500,2000))) %>% 
  grid_max_entropy(size = 10)


tic("boost min price") # start time
# grid search
boost_results_min_p <- boost_wf_min_p %>%
  tune_grid( 
    resamples = cross_val_scheme, # cross validation scheme
    grid = boost_hypers,         # hyper parameter values to test
    metrics = perf_metrics,       # metrics to compute
    control = ctrl                # save predictions and performance metrics
  )
toc() # end timing


tic("boost max price") # start time

# grid search
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
boost_pred_min_p <- collect_predictions(boost_results_min_p) %>%
  inner_join(boost_params_min_p, by = c("mtry", "trees", "min_n"))

# 2) MAX PRICE
boost_pred_max_p <- collect_predictions(boost_results_max_p) %>%
  inner_join(boost_params_max_p, by = c("mtry", "trees", "min_n"))


# fit our model on full training data

# 1) MIN PRICE
# get_final_model is a custom functions (see code/functions/final_model)
boost_fit_train_min_p <- get_final_model(recipe_steps = recipe_steps_min_p,
                                           data = training(split_scheme),
                                           model = boost_model,
                                           target = "min_price",
                                           params = boost_params_min_p)

# 2) MAX PRICE
# get_final_model is a custom functions (see code/functions/final_model)
boost_fit_train_max_p <- get_final_model(recipe_steps = recipe_steps_max_p,
                                           data = training(split_scheme),
                                           model = boost_model,
                                           target = "max_price",
                                           params = boost_params_max_p)

# variable importance plot (vip)

# 1) MIN PRICE
p_vip_boost_min_p <- vip(boost_fit_train_min_p, num_features = NUM_VIP, geom = "point",
  aesthetics = list(color = cbPalette[4], 
  fill = cbPalette[4])) + THEME + ggtitle("Boosting min. price")

p_vip_boost_min_p %>% ggplotly()


# 2) MAX PRICE
p_vip_boost_max_p <- vip(boost_fit_train_max_p, num_features = NUM_VIP, 
  geom = "point", aesthetics = list(color = cbPalette[4], 
  fill = cbPalette[4])) + THEME + ggtitle("Boosting max. price") 

p_vip_boost_max_p %>% ggplotly()


# add figures togehter
(p_vip_boost_min_p + p_vip_boost_max_p )


# Look at performance on validation data
#---------------------------------------

# 1) MIN PRICE

# get prediction on validation data and add true values
# prediction new data is custom function: see code/functions/predict_new_data.R
boost_pred_val_min_p <- prediction_new_data(recipe_steps = recipe_steps_min_p,
  data = testing(split_scheme),model = boost_fit_train_min_p) %>%
  mutate(actual = testing(split_scheme)$min_price) %>% 
  dplyr::rename(fitted = .pred) %>%
  select(fitted,actual) 

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
# untransform_target is a custom function see code/functions/untransform_target
boost_pred_val_min_p <- untransform_target(pred = boost_pred_val_min_p,
                        mean = mean_training_min_p,
                        sd = sd_training_min_p,
                        # suppose you did a log transformation on the target 
                        # you can give FUN = exp
                        FUN = exp) %>%
                        mutate(residual = actual - fitted_unscaled)

# check performance metrics
boost_perf_val_min_p <- perf_metrics(boost_pred_val_min_p,
                        truth = actual, estimate = fitted_unscaled)

boost_perf_val_min_p

# 2) MAX PRICE

# get prediction on validation data and add true values
# prediction new data is custom function: see code/functions/predict_new_data.R
boost_pred_val_max_p <- prediction_new_data(recipe_steps = recipe_steps_max_p,
  data = testing(split_scheme), model = boost_fit_train_max_p) %>%
  mutate(actual = testing(split_scheme)$max_price) %>% 
  dplyr::rename(fitted = .pred) %>%
  select(fitted,actual) 

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
# untransform_target is a custom function see code/functions/untransform_target
boost_pred_val_max_p <- untransform_target(pred = boost_pred_val_max_p,
                        mean = mean_training_max_p,
                        sd = sd_training_max_p,
                       # suppose you did a log transformation on the target 
                       # you can give FUN = exp()
                        FUN = exp)  %>%
                        mutate(residual = actual - fitted_unscaled)
  

# check performance metrics
boost_perf_val_max_p <- perf_metrics(boost_pred_val_max_p,
                                      truth = actual, estimate = fitted_unscaled)

boost_perf_val_max_p

# I think this is how the perfmance will be evaluated
(boost_perf_val_min_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull +
    boost_perf_val_max_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull)



# visualize predictions and residuals on validation data

boost_fitAct_min_p_val <- boost_pred_val_min_p %>% 
  ggplot(.,aes(x = actual, y = fitted_unscaled)) + 
  geom_abline(col = "red",linetype = "dashed") +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Boosting (validation) min. price") + THEME 

boost_fitAct_min_p_val %>% ggplotly


# MAX PRICE

boost_fitAct_max_p_val <- boost_pred_val_max_p %>% 
  ggplot(.,aes(x = actual, y = fitted_unscaled)) + 
  geom_abline(col = "red",linetype = "dashed") + 
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Boosting (validation) : max. price") + THEME 

boost_fitAct_max_p_val %>% ggplotly
boost_fitAct_min_p_val + boost_fitAct_max_p_val


# residuals analysis: should not be a pattern in the residuals

# MIN PRICE

boost_resid_min_p_val <- boost_pred_val_min_p %>% 
  ggplot(.,aes(x = seq(1:nrow(boost_pred_val_min_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Boosting (validation) min. price") + THEME 

boost_resid_min_p_val %>% ggplotly


# maximum price

boost_resid_max_p_val <- boost_pred_val_max_p %>% 
  ggplot(.,aes(x = seq(1:nrow(boost_pred_val_max_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Boosting (validation) max. price") + THEME 

boost_resid_max_p_val %>% ggplotly


(boost_fitAct_min_p_val + boost_fitAct_max_p_val)

# everything together
(boost_fitAct_min_p_val + boost_fitAct_max_p_val)/
  (boost_resid_min_p_val + boost_resid_max_p_val)

# to save a figure (this is just a test): it will automatically save the last
# figure you plotted
ggsave("output/boosting_predictions_validations.jpeg", 
       device = "jpeg", width = 8, height = 5)

#=====================================================#
# predictions on test data
#=====================================================#

# read in test data
test <- read.csv("data/train.csv",sep = ";") %>% dplyr::rename(
  id = ï..id) 


# compute mean and sd on all data
mean_allData_min_p <- df %>% select(min_price) %>% pull %>% mean()
sd_allData_min_p <- df %>% select(min_price) %>% pull %>% sd()

# compute mean and sd on all data
mean_allData_max_p <- df %>% select(max_price) %>% pull %>% mean()
sd_allData_max_p <- df %>% select(max_price) %>% pull %>% sd()

#------------------------------------------------------------------------------
# # SUBMISSION FOR GLMNET
#------------------------------------------------------------------------------

# fit our model on full training data

# 1) min price

# get_final_model is a custom functions (see code/functions/final_model)
glmnet_fit_allData_min_p <- get_final_model(recipe_steps = recipe_steps_min_p,
                             data = df,
                             model = glmnet_model,
                             target = "min_price",
                             # best params found on the cross validation grid search
                             params = glmnet_params_min_p) 

# 2) MAX PRICE
glmnet_fit_allData_max_p <- get_final_model(recipe_steps = recipe_steps_max_p,
                             data = df,
                             model = glmnet_model,
                             target = "max_price",
                             # best params found on the cross validation grid search
                             params = glmnet_params_max_p) 


# 1) MIN PRICE

# get prediction on validation data and add true values
# prediction new data is custom function: see code/functions/predict_new_data.R
glmnet_pred_test_min_p <- prediction_new_data(recipe_steps = recipe_steps_min_p,
  data = test, model = glmnet_fit_allData_min_p) %>% dplyr::rename(fitted = .pred)

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
# untransform_target is a custom function see code/functions/untransform_target
glmnet_pred_test_min_p <- untransform_target(pred = glmnet_pred_test_min_p,
                                           mean = mean_allData_min_p,
                                           sd = sd_allData_min_p,
                                           # this is just identity mapping, 
                                           # suppose you did a log transformation on the target you can give FUN = exp()
                                           FUN = function(x) x*1) 


# 2) MAX PRICE

# get prediction on validation data and add true values
glmnet_pred_test_max_p <- prediction_new_data(recipe_steps = recipe_steps_max_p,
  data = test, model = glmnet_fit_allData_max_p) %>% dplyr::rename(fitted = .pred)

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
glmnet_pred_test_max_p <- untransform_target(pred = glmnet_pred_test_max_p,
                                           mean = mean_allData_max_p,
                                           sd = sd_allData_max_p,
                                           # this is just identity mapping, 
                                           # suppose you did a log transformation on the target you can give FUN = exp()
                                           FUN = function(x) x*1) 


# create a tibble and save it
test_sub_glmnet <- tibble(ID = test$id,
       MIN = glmnet_pred_test_min_p$fitted_unscaled,
       MAX = glmnet_pred_test_max_p$fitted_unscaled)  

# predictions where the mimimum is greater than the maximum
test_sub_glmnet %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# take the min and max for each row
test_sub_glmnet_update <- tibble(
  MIN = apply(test_sub_glmnet, 1,min), # take min 
  MAX = apply(test_sub_glmnet, 1,max)) # take max 


# predictions where the mimimum is greater than the maximum
# there should be no cases
test_sub_glmnet_update %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# write submission 
test_sub_glmnet_update %>% 
  write.csv("submission/glmnet.csv", row.names=FALSE)



#------------------------------------------------------------------------------
# SUBMISSION FOR BOOSTING
#------------------------------------------------------------------------------
# fit our model on full training data

# 1) min price
# train on all data (df)
# get_final_model is a custom functions (see code/functions/final_model)
boost_fit_allData_min_p <- get_final_model(recipe_steps = recipe_steps_min_p,
                                            data = df,
                                            model = boost_model,
                                            target = "min_price",
                                            # best params found on the cross validation grid search
                                            params = boost_params_min_p) 

# 2) MAX PRICE
# train on all data (df)
boost_fit_allData_max_p <- get_final_model(recipe_steps = recipe_steps_max_p,
                                            data = df,
                                            model = boost_model,
                                            target = "max_price",
                                            # best params found on the cross validation grid search
                                            params = boost_params_max_p) 


# 1) MIN PRICE

# get prediction on validation data and add true values
# prediction new data is custom function: see code/functions/predict_new_data.R
boost_pred_test_min_p <- prediction_new_data(recipe_steps = recipe_steps_min_p,
  data = test, model = boost_fit_allData_min_p) %>% 
  dplyr::rename(fitted = .pred)

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
# untransform_target is a custom function see code/functions/untransform_target
boost_pred_test_min_p <- untransform_target(pred = boost_pred_test_min_p,
                                             mean = mean_allData_min_p,
                                             sd = sd_allData_min_p,
                                             # this is just identity mapping, 
                                             # suppose you did a log transformation on the target you can give FUN = exp()
                                             FUN = exp) 


# 2) MAX PRICE

# get prediction on validation data and add true values
boost_pred_test_max_p <- prediction_new_data(recipe_steps = recipe_steps_max_p,
  data = test, model = boost_fit_allData_max_p) %>% 
  dplyr::rename(fitted = .pred)

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
boost_pred_test_max_p <- untransform_target(pred = boost_pred_test_max_p,
                                             mean = mean_allData_max_p,
                                             sd = sd_allData_max_p,
                                             # this is just identity mapping, 
                                             # suppose you did a log transformation on the target you can give FUN = exp()
                                             FUN = exp) 


# create a tibble and save it
test_sub_boost <- tibble(ID = test$id,
                          MIN = boost_pred_test_min_p$fitted_unscaled,
                          MAX = boost_pred_test_max_p$fitted_unscaled)  

# predictions where the mimimum is greater than the maximum
test_sub_boost %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# take the min and max for each row
test_sub_boost_update <- tibble(
  MIN = apply(test_sub_boost, 1,min), # take min 
  MAX = apply(test_sub_boost, 1,max)) # take max 


# predictions where the mimimum is greater than the maximum
# there should be no cases
test_sub_boost_update %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# write submission 
test_sub_boost_update %>% 
  write.csv("submission/boosting.csv", row.names=FALSE)



 
