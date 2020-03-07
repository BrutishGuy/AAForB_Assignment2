#' Creates new lagged by means of lags of a given timeserie
#' 
#' @param recipe_steps: A recipe
#' @param data: tibble containing date and consumption 
#' @param model parsnip model
#' @return return a tibble with the predictions and true values

prediction_new_data <- function(recipe_steps, data, model){
  
  # prep  data
  prep_data <- recipe_steps %>% 
    prep(data) %>%
    bake(data) 
  
  # get predictions
  model %>% predict(new_data = prep_data)
  
}