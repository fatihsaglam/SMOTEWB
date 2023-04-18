#' @title  Random Undersampling (RUS)
#'
#' @description Resampling with RUS.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#'
#' @details
#' Random Undersampling (RUS) is a method of removing negative
#' samples until balance is achieved.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @examples
#'
#' set.seed(1)
#' x <- rbind(matrix(rnorm(2000, 3, 1), ncol = 2, nrow = 1000),
#'            matrix(rnorm(100, 5, 1), ncol = 2, nrow = 50))
#' y <- as.factor(c(rep("negative", 1000), rep("positive", 50)))
#'
#' plot(x, col = y)
#'
#' # resampling
#' m <- RUS(x = x, y = y)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname RUS
#' @export

RUS <- function(x, y) {
  x <- as.matrix(x)

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  class_names <- as.character(unique(y))
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,,drop = FALSE]
  x_neg <- x[y == class_neg,,drop = FALSE]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  n_remove <- (n_neg - n_pos)

  i_remove <- sample(1:n_neg, n_remove)

  x_neg_new <- x_neg[-i_remove,,drop = FALSE]

  x_new <- rbind(
    x_pos,
    x_neg_new
  )
  y_new <- c(
    rep(class_pos, n_pos),
    rep(class_neg, n_pos)
  )
  y_new <- factor(y_new, levels = levels(y), labels = levels(y))

  return(list(
    x_new = x_new,
    y_new = y_new
  ))
}

