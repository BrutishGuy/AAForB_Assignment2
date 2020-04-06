#' Check predictions
#' 
#' @param df: Tibble with the MIN and MAX price
#' @param var: String indicating the MIN OR MAX price 
#' @param xlab: String indicating the label of the x axis
#' @return ggplot with the distributions of the MIN or MAX price

check_predictions <- function(df, var = "MIN", xlab = ""){
  
  
  df %>% select(var) %>% ggplot(., aes(x=!!ensym(var))) + 
    geom_histogram(aes(y=..density..),
    bins = 30, color = cbPalette[4], fill="white") +
    geom_density(alpha=.2, fill=cbPalette[7],color = cbPalette[7]) + 
    labs(x = xlab) 
}