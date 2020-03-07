#' reverses transformations
#' 
#' @param pred: transformed predictions
#' @param mean: numeric, mean 
#' @param sd: numeric, standard devation 
#' @param FUN: function nverse transformation (default is just multiply everyting with 1)
#' @return return a tibble with the predictions in the scaled and orignal units

untransform_target <- function(pred, mean = 0, sd = 1, FUN = function(x) x*1){
  
  pred %>% mutate(
    fitted_unscaled  = (fitted * sd + mean) %>% FUN
  )
}

