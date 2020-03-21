# clear environment
rm(list = ls(all.names = TRUE))


# model building libaries
library(workflows)
library(tidymodels)
library(glmnet) # lasso, ridge or elastic net
library(tictoc) # timing

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
  add_recipe(recipe_steps_min_p(train_train)) %>% 
  add_model(glmnet_model)


# MAX PRICE
glmnet_wf_max_p <-
  workflow() %>% 
  add_recipe(recipe_steps_max_p(train_train)) %>% 
  add_model(glmnet_model)
glmnet_wf_max_p


# set grid for parameters to look at (same for min and max price)
# you can increase the size, but the model fitting will be slower
# since you will test more values
glmnet_hypers <- parameters(penalty(), mixture()) %>%
  grid_max_entropy(size = 20) 

# visualize the parameters of glmnet
ggplot(glmnet_hypers,aes(x = penalty, y = mixture)) + geom_point()+
  scale_x_log10() + THEME

tic("glmnet min price") # start time
# grid search
set.seed(100) # seed for reproduceability
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
set.seed(100) # seed for reproduceability
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

# performance for best tuning parameters

# 1) MIN PRICE
collect_metrics(glmnnet_results_min_p) %>%
  inner_join(glmnet_params_min_p, by = c("penalty", "mixture"))

# 2) MAX PRICE
collect_metrics(glmnnet_results_max_p) %>%
  inner_join(glmnet_params_max_p, by = c("penalty", "mixture"))



# just a cool visual plot how the coefficients shrink accoridng to penalty
# abolsute higher coefficients are more important features
# this model was also discussed in class

# fit our model on full training data (train_train)

# 1) MIN PRICE
# fit model on train_train data using optimal valus found by the cross validation scheme
glmnet_fit_train_train_min_p <- glmnet_wf_min_p %>%
  finalize_workflow(glmnet_params_min_p) %>%
  fit(data = train_train)


tidy_coefs_min_p <-
  glmnet_fit_train_train_min_p$fit$fit$fit %>%
  broom::tidy() %>%
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
p_tidy_coefs_min_p <- 
  tidy_coefs_min_p %>%
  ggplot(aes(x = lambda, y = estimate, group = term, col = term, label = term)) +
  geom_vline(xintercept = lambda_opt_min_p, lty = 2) + THEME + 
  labs(x = "Penalty", y  ="Coefficient estimate") +  ggtitle("Minimum price") +
  geom_line(alpha = 1) +
  theme(legend.position = "none") + 
  scale_x_log10() +
  ggrepel::geom_text_repel(data = label_coefs_min_p, aes(label = term, x = .05))

p_tidy_coefs_min_p


# 2) MAX PRICE
glmnet_fit_train_train_max_p <- 
  glmnet_wf_max_p %>%
  finalize_workflow(glmnet_params_max_p) %>%
  fit(data = train_train)

tidy_coefs_max_p <-
  broom::tidy(glmnet_fit_train_train_max_p$fit$fit$fit) %>%
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
p_tidy_coefs_max_p <- 
  tidy_coefs_max_p %>%
  ggplot(aes(x = lambda, y = estimate, group = term, col = term, label = term)) +
  geom_vline(xintercept = lambda_opt_max_p, lty = 2) + THEME + 
  labs(x = "Penalty", y  ="Coefficient estimate") + ggtitle("Maximum price") +
  geom_line(alpha = 1) +
  theme(legend.position = "none") +
  scale_x_log10() +
  ggrepel::geom_text_repel(data = label_coefs_max_p, aes(label = term, x = .005)) 


(p_tidy_coefs_min_p + p_tidy_coefs_max_p)

#---------------------------------------
# variable importance plot
#---------------------------------------

# MIN PRICE
p_vip_glmnet_min_p <- vip(glmnet_fit_train_train_min_p$fit$fit$fit, num_features = NUM_VIP,
  lambda = glmnet_params_min_p$penalty, geom = "point",
  aesthetics = list(color = cbPalette[4], fill = cbPalette[4])) +
  THEME + ggtitle("Linear model (glmnet) min. price")

p_vip_glmnet_min_p %>% ggplotly()


# MAX PRICE
p_vip_glmnet_max_p <- vip(glmnet_fit_train_train_max_p$fit$fit$fit, num_features = NUM_VIP,
  lambda = glmnet_params_max_p$penalty, geom = "point",
  aesthetics = list(color = cbPalette[4], fill = cbPalette[4])) + THEME + 
  ggtitle("Linear model (glmnet) max. price") 

p_vip_glmnet_max_p %>% ggplotly()


# add figures togehter
(p_vip_glmnet_min_p + p_vip_glmnet_max_p )



#---------------------------------------
# Look at performance on validation data
#---------------------------------------

# 1) MIN PRICE

# perform transformation on train_val data using the train_train to estimate
# train_train_rec_min_p <- prep(recipe_steps_min_p(train_train), training = train_train, fresh = TRUE, verbose = TRUE)
# train_val_data_min_p  <- bake(train_train_rec_min_p, new_data = train_validation %>% select(-min_price,-max_price))


# make predictions on train_val data and add true values
glmnet_pred_val_min_p <- 
  predict(glmnet_fit_train_train_min_p, new_data = train_validation %>% select(-min_price, -max_price)) %>%
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
glmnet_pred_val_min_p <- 
  invert_transformations_target(
    pred = glmnet_pred_val_min_p,
    mean = mean_training_min_p,
    sd = sd_training_min_p,
  # suppose you did a log transformation on the target 
  # you can give FUN = exp()
    FUN = exp
    ) %>%
  mutate(residual = actual - fitted_inverted)


# check performance metrics
glmnet_perf_val_min_p <- 
  perf_metrics(glmnet_pred_val_min_p,
  truth = actual, estimate = fitted_inverted)

glmnet_perf_val_min_p

# 2) MAX PRICE

# perform transformation on train_val data using the train_train to estimate
# train_train_rec_max_p <- prep(recipe_steps_max_p(train_train), training = train_train, fresh = TRUE, verbose = TRUE)
# train_val_data_max_p  <- bake(train_train_rec_max_p, new_data = train_validation %>% select(-min_price,-max_price))


# get prediction on validation data and add true values
glmnet_pred_val_max_p <- 
  predict(glmnet_fit_train_train_max_p, new_data = train_validation %>% select(-min_price, -max_price)) %>%
  mutate(actual = train_validation$max_price) %>% 
  dplyr::rename(fitted = .pred)


# rescale back to the orignal scale
mean_training_max_p <- train_train %>% 
  select(max_price) %>% log %>% pull %>% mean()
sd_training_max_p <- train_train %>% 
  select(max_price) %>% log %>% pull %>% sd()

# transform back to original unit
# add back mean and sd  + invert potentially others transformation such as log trans
glmnet_pred_val_max_p <- 
  invert_transformations_target(
    pred = glmnet_pred_val_max_p,
    mean = mean_training_max_p,
    sd = sd_training_max_p,
    # suppose you did a log transformation on the 
    # target you can give FUN = exp()
    FUN = exp) %>% 
  mutate(residual = actual - fitted_inverted)



# save predictions on train_validation
tibble(
   ID = train_validation$id,
   MIN = glmnet_pred_val_min_p %>% select(fitted_inverted) %>% pull,
   MAX = glmnet_pred_val_max_p %>% select(fitted_inverted) %>% pull
   ) %>%
  write.csv(.,file = "output/predictions_train_val/glmnet_pred.csv", row.names=FALSE)


# check performance metrics
glmnet_perf_val_max_p <- perf_metrics(glmnet_pred_val_max_p,
  truth = actual, estimate = fitted_inverted)

glmnet_perf_val_max_p

# I think this is how the perfmance will be evaluated
(glmnet_perf_val_min_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull +
    glmnet_perf_val_max_p %>% filter(.metric=="mae") %>% select(.estimate) %>% pull)


# visualize predictions and residuals on validation data
glmnet_fitAct_min_p_val <- glmnet_pred_val_min_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_point() +
  geom_abline(col = "red",linetype = "dashed") +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Linear Model (validation) min. price") + THEME 

glmnet_fitAct_min_p_val %>% ggplotly


# 2) MAX PRICE

glmnet_fitAct_max_p_val <- glmnet_pred_val_max_p %>% 
  ggplot(.,aes(x = actual, y = fitted_inverted)) + 
  geom_abline(col = "red",linetype = "dashed") + 
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') +
  geom_point() + labs(x = "Actual Values", y = "Fitted values") + 
  ggtitle("Linear Model (validation) : max. price") + THEME 

glmnet_fitAct_max_p_val %>% ggplotly
glmnet_fitAct_min_p_val + glmnet_fitAct_max_p_val


# residuals analysis: should not be a pattern in the residuals

# 1) MIN PRICE

glmnet_resid_min_p_val <- glmnet_pred_val_min_p %>% 
  ggplot(.,aes(x = seq(1:nrow(glmnet_pred_val_min_p)), y = residual)) + 
  geom_hline(yintercept=0, col = "red",linetype = "dashed") +
  geom_point() + labs(y = "Residuals", x = "Obserations nr.") + 
  ggtitle("Linear Model (validation) min. price") + THEME 

glmnet_resid_min_p_val %>% ggplotly


# 2) MAX PRICE

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


# visualize the distributions of the predictions on validation test set and compare it 
# with the actual values on the validation we have (this is just a check)

# distributions on training data
p_test_min_p_train_val <- 
  check_predictions(
  df=glmnet_pred_val_min_p,var = "actual",
  xlab = "Min. price: validation: actual") + THEME

p_test_max_p_train_val <- 
  check_predictions(
  df=glmnet_pred_val_max_p,var = "actual",
  xlab = "Max. price validation: actual") + THEME

# distribution of the predictions
p_glmnet_pred_min_p_train_val <- 
  check_predictions(
    df=glmnet_pred_val_min_p,var = "fitted_inverted",
    xlab = "Min. price validation: fitted") + THEME

p_glmnet_pred_max_p_train_val <- 
  check_predictions(
    df=glmnet_pred_val_max_p,var = "fitted_inverted",
    xlab = "Max. price validation: fitted") + THEME

((p_test_min_p_train_val + p_test_max_p_train_val)/
    (p_glmnet_pred_min_p_train_val + p_glmnet_pred_max_p_train_val))


#---------------------------------------
# Fit final model on all data
#---------------------------------------

# MIN PRICE
glmnet_fit_train_all_data_min_p <- glmnet_wf_min_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(glmnet_params_min_p) %>%
  fit(data = df)

# save model
saveRDS(glmnet_fit_train_all_data_min_p, "code/models/save_models/glmnet_min_p.rds")

# MAX PRICE
glmnet_fit_train_all_data_max_p <- glmnet_wf_max_p %>%
  update_recipe(recipe_steps_max_p(df)) %>%
  finalize_workflow(glmnet_params_max_p) %>%
  fit(data = df)

# save model
saveRDS(glmnet_fit_train_all_data_max_p, "code/models/save_models/glmnet_max_p.rds")



