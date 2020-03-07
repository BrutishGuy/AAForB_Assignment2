#' returns the fitted model on the data
#' 
#' @param recipe A recipe
#' @param data Tibble containg date and consumption feature
#' @param model Parsnip model
#' @param target String indicating the target variable
#' @param params Tibble containing the parameter values for the hyper parameters
#' @return 


get_final_model <- function(recipe_steps, data, model, target, params){
  
  # prep data
  prep_data <- prep(recipe_steps)
  # select model with chosen parameters
  mod_params <- finalize_model(model, params)
  # fit model
  mod_params %>% fit(as.formula(paste(target,"~.")),data = bake(prep_data, new_data = data))
  
}







